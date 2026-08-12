"""Auditable CLI for the preregistered evidence-integrity benchmark."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

from dp_indicator.agents.claim_verifier import ClaimVerifier
from dp_indicator.benchmarks.evidence_integrity import (
    DEFAULT_HOLISTIC_BATCH_SIZE,
    EvidenceIntegrityRunner,
    MAX_HOLISTIC_BATCH_SIZE,
    PROVIDER_MODEL_NORMALIZATION_RULE_VERSION,
    TERMINAL_FAILURE_LATCH_POLICY_VERSION,
    TRANSIENT_TRANSPORT_EXCEPTION_NAMES,
    TRANSIENT_TRANSPORT_RETRY_POLICY_VERSION,
    _normalized_provider_model,
    build_split_variants,
    extract_source_units,
    sample_paired_variants,
    select_coverage_aware_split,
)
from dp_indicator.benchmarks import evidence_integrity as runner_module
from dp_indicator.benchmarks import evidence_integrity_stats as stats_module
from dp_indicator.benchmarks.evidence_integrity_stats import (
    binary_metrics,
    build_bundles,
    bundle_bootstrap_difference,
    bundle_macro_f1_difference,
    bundle_metrics,
    bundle_permutation_test,
    cluster_bootstrap_difference,
    cluster_permutation_test,
    compute_split_manifest_hash,
    exact_mcnemar,
    holm_adjust,
    macro_f1_difference,
    majority_vote_predictions,
)
from dp_indicator.core.llm import ChatRawResult, LLMClient


SEED = 20260728
CORRUPTION_TYPES = (
    "invalid_id",
    "citation_swap",
    "causal_overclaim",
    "entity_swap",
)
JUDGED_CONDITIONS = ("full", "holistic")
ALL_CONDITIONS = ("full", "holistic", "id_only")
DEFAULT_MODEL = "bh:glm-5.1"
DEFAULT_TARGET = "Kv1.3"
SCHEMA_VERSION = 1
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(?:accesskey|api[_-]?key|authorization)\b"
    r"\s*[\"']?\s*[:=]\s*[\"']?\s*(?:bearer\s+)?"
    r"(?!\[REDACTED\])\S+"
)


class IncompleteRun(RuntimeError):
    """Raised when confirmatory artifacts are incomplete."""


class ConfirmationLockMismatch(RuntimeError):
    """Raised when the frozen confirmation contract changed."""


class SecretScanFailed(RuntimeError):
    """Raised when a benchmark artifact may contain a credential."""


class PaidExecutionBlocked(RuntimeError):
    """Raised before any paid request when safety prerequisites fail."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _normalized_model(value: object) -> str:
    return _normalized_provider_model(value)


def _atomic_write(path: Path, payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        text = ""
    current_secret = os.environ.get("BH_API_KEY", "")
    reasons = []
    if current_secret and current_secret in text:
        reasons.append("current_api_key_value")
    if _CREDENTIAL_ASSIGNMENT.search(text):
        reasons.append("credential_assignment_pattern")
    if reasons:
        error = SecretScanFailed(
            f"Unsafe content blocked before write: {path}"
        )
        error.artifact_path = path
        error.reasons = reasons
        raise error
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    _atomic_write(path, payload)


def _read_json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IncompleteRun(f"Unreadable benchmark artifact: {path}") from error
    return value


def _manifest_fingerprint(manifest: dict) -> str:
    return _sha256(manifest)


def _validate_manifest_integrity(manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise IncompleteRun("Manifest integrity check requires an object")
    declared = manifest.get("manifest_sha256")
    without_declared = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    if (
        not isinstance(declared, str)
        or declared != _sha256(without_declared)
        or manifest.get("source_units_sha256")
        != _sha256(manifest.get("source_units"))
        or manifest.get("split_sha256") != _sha256(manifest.get("split"))
        or not manifest.get("integrity", {}).get("ok")
    ):
        raise IncompleteRun("Manifest integrity check failed")
    holistic_batch_size = manifest.get("protocol", {}).get(
        "holistic_batch_size"
    )
    if (
        isinstance(holistic_batch_size, bool)
        or not isinstance(holistic_batch_size, int)
        or not 1 <= holistic_batch_size <= MAX_HOLISTIC_BATCH_SIZE
    ):
        raise IncompleteRun(
            "Manifest holistic_batch_size must be between 1 and 5"
        )
    split = manifest.get("split", {})
    pilot = split.get("pilot_ids")
    confirmation = split.get("confirmation_ids")
    if (
        not isinstance(pilot, list)
        or not isinstance(confirmation, list)
        or len(pilot) != len(set(pilot))
        or len(confirmation) != len(set(confirmation))
        or set(pilot).intersection(confirmation)
    ):
        raise IncompleteRun("Manifest integrity check failed for split")


def _sample_by_type(
    variants: list[dict],
    *,
    pairs_per_type: int,
    seed: int,
) -> dict[str, list[dict]]:
    sampled = sample_paired_variants(
        variants,
        pairs_per_type=pairs_per_type,
        seed=seed,
    )
    output = {kind: [] for kind in CORRUPTION_TYPES}
    for item in sampled:
        kind = str(
            item.get("paired_corruption_type")
            or item.get("variant_type")
        )
        if kind not in output:
            raise ValueError(f"Unexpected paired corruption type: {kind}")
        output[kind].append(item)
    expected = pairs_per_type * 2
    if any(len(output[kind]) != expected for kind in CORRUPTION_TYPES):
        raise ValueError("Paired sample is incomplete")
    return output


def _split_manifest(split: dict, split_name: str) -> dict:
    value = {
        "name": split_name,
        "pilot_ids": list(split["pilot_ids"]),
        "confirmation_ids": list(split["confirmation_ids"]),
    }
    value["split_sha256"] = compute_split_manifest_hash(value)
    return value


def _build_split_bundles(
    variants: list[dict],
    split: dict,
    split_name: str,
    seed: int,
) -> list[dict]:
    return build_bundles(
        variants,
        _split_manifest(split, split_name),
        seed=seed,
    )


def prepare_benchmark(
    project_root: Path,
    output_dir: Path | None = None,
) -> dict:
    """Freeze constructibility-aware splits, pair samples, and bundles."""
    project_root = Path(project_root)
    output_dir = Path(
        output_dir or project_root / "evidence_integrity_benchmark"
    )
    units = extract_source_units(project_root / "repeat_runs")
    preparation_status = {
        "ready": False,
        "source_count": len(units),
        "requested_source_count": 160,
        "source_shortfall": max(0, 160 - len(units)),
        "split_counts": {},
        "yield_counts": {},
        "blocked_types": {},
    }
    _write_json(
        output_dir / "PREPARATION_SHORTFALL.json",
        preparation_status,
    )
    try:
        split = select_coverage_aware_split(units, 40, 120, SEED)
        preparation_status["split_counts"] = {
            "pilot": len(split["pilot_ids"]),
            "confirmation": len(split["confirmation_ids"]),
        }
        split_variants = build_split_variants(units, split, seed=SEED)
    except Exception:
        _write_json(
            output_dir / "PREPARATION_SHORTFALL.json",
            preparation_status,
        )
        raise
    split_payload = {}
    for split_name, pair_count in (("pilot", 10), ("confirmation", 30)):
        data = split_variants[split_name]
        variants = data["variants"]
        preparation_status["yield_counts"][split_name] = dict(
            data["yield_report"].get("counts", {})
        )
        preparation_status["blocked_types"][split_name] = list(
            data["yield_report"].get("blocked_types", [])
        )
        if not data["yield_report"].get("all_ready"):
            blocked = ", ".join(data["yield_report"].get("blocked_types", []))
            _write_json(
                output_dir / "PREPARATION_SHORTFALL.json",
                preparation_status,
            )
            raise IncompleteRun(
                f"{split_name} mutation-yield shortfall: {blocked}"
            )
        try:
            sampled = _sample_by_type(
                variants,
                pairs_per_type=pair_count,
                seed=SEED,
            )
            bundles = _build_split_bundles(
                variants, split, split_name, SEED
            )
        except Exception:
            _write_json(
                output_dir / "PREPARATION_SHORTFALL.json",
                preparation_status,
            )
            raise
        split_payload[split_name] = {
            "source_ids": list(data["source_ids"]),
            "test1": sampled,
            "bundles": bundles,
            "yield_report": data["yield_report"],
            "attrition_report": data["attrition_report"],
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "seed": SEED,
            "pilot_sources": 40,
            "confirmation_sources": 120,
            "pairs": {"pilot": 10, "confirmation": 30},
            "bundles": {"pilot": 9, "confirmation": 30},
            "conditions": list(ALL_CONDITIONS),
            "judged_repetitions": 3,
            "model": DEFAULT_MODEL,
            "base_url": "https://open.bohrium.com/openapi/v1",
            "timeout_seconds": 180,
            "max_request_retries": 1,
            "batch_size": 20,
            "holistic_batch_size": DEFAULT_HOLISTIC_BATCH_SIZE,
            "verifier_timeout_seconds": 120,
            "verifier_max_concurrency": 3,
            "target": DEFAULT_TARGET,
            "primary_endpoint": "full_minus_id_only_macro_f1",
            "alpha": 0.05,
            "clean_fpr_max": 0.15,
        },
        "source_units": units,
        "source_units_sha256": _sha256(units),
        "split": split,
        "split_sha256": _sha256(split),
        "splits": split_payload,
        "protocol_deviations": [],
        "integrity": {
            "ok": (
                len(split["pilot_ids"]) == 40
                and len(split["confirmation_ids"]) == 120
                and set(split["pilot_ids"]).isdisjoint(
                    split["confirmation_ids"]
                )
            )
        },
    }
    manifest["manifest_sha256"] = _sha256(manifest)
    _write_json(output_dir / "MANIFEST.json", manifest)
    preparation_status["ready"] = True
    _write_json(
        output_dir / "PREPARATION_SHORTFALL.json",
        preparation_status,
    )
    return manifest


def _bundle_items(bundles: list[dict]) -> list[dict]:
    return [
        dict(member)
        for bundle in bundles
        for member in bundle.get("members", [])
    ]


def _scopes(manifest: dict, split_name: str):
    split_data = manifest["splits"][split_name]
    for kind in CORRUPTION_TYPES:
        yield "test1", kind, split_data["test1"][kind]
    bundles = split_data.get("bundles", [])
    if bundles:
        yield "test2", "bundles", _bundle_items(bundles)


def _enrich_predictions(
    predictions: list[dict],
    variants: list[dict],
) -> list[dict]:
    by_id = {str(item["variant_id"]): item for item in variants}
    output = []
    for prediction in predictions:
        variant = by_id[str(prediction["variant_id"])]
        record = {
            **prediction,
            "gold": variant["gold"],
            "variant_type": variant["variant_type"],
        }
        for key in ("pair_id", "pair_role", "paired_corruption_type"):
            if key in variant:
                record[key] = variant[key]
        output.append(record)
    return sorted(
        output,
        key=lambda item: (
            str(item.get("pair_id", "")),
            str(item.get("pair_role", "")),
            str(item["variant_id"]),
        ),
    )


def _latest_predictions(
    run_root: Path,
    condition: str,
    split_name: str,
    variants: list[dict],
) -> dict[int, list[dict]]:
    repetitions = (1, 2, 3) if condition in JUDGED_CONDITIONS else (1,)
    output = {}
    for repetition in repetitions:
        records = []
        rep_root = (
            run_root
            / "attempts"
            / condition
            / split_name
            / f"rep_{repetition:02d}"
        )
        for batch_dir in sorted(rep_root.glob("batch_*")):
            paths = sorted(batch_dir.glob("batch_result_*.json"))
            if not paths:
                raise IncompleteRun(f"Missing batch result: {batch_dir}")
            final = _read_json(paths[-1])
            if not final.get("complete") or final.get("test_mode") not in (
                True,
                False,
            ):
                raise IncompleteRun(f"Incomplete batch result: {paths[-1]}")
            records.extend(final["normalized_predictions"])
        if len(records) != len(variants):
            raise IncompleteRun(
                f"{condition} repetition {repetition} is incomplete"
            )
        output[repetition] = _enrich_predictions(records, variants)
    return output


def _actual_provider_models(
    run_root: Path,
    condition: str,
    split_name: str,
    repetition: int,
    *,
    fallback: str,
) -> list[str]:
    if condition == "id_only":
        return [fallback]
    models = []
    rep_root = (
        run_root / "attempts" / condition / split_name
        / f"rep_{repetition:02d}"
    )
    for path in sorted(rep_root.glob("batch_*/request_*.json")):
        record = _read_json(path)
        if record.get("request_succeeded"):
            models.append(_normalized_model(record.get("provider_model")))
    return sorted(set(models)) if models else [fallback]


def _attempt_tree_evidence(
    run_root: Path,
    condition: str,
    split_name: str,
    repetition: int,
) -> dict:
    rep_root = (
        Path(run_root) / "attempts" / condition / split_name
        / f"rep_{repetition:02d}"
    )
    batch_dirs = sorted(rep_root.glob("batch_*"))
    if not batch_dirs:
        raise IncompleteRun(f"Missing runner attempts: {rep_root}")
    entries = []
    runner_fingerprints = set()
    final_predictions = []
    for batch_dir in batch_dirs:
        request_paths = sorted(batch_dir.glob("request_*.json"))
        for path in request_paths:
            record = _read_json(path)
            fingerprint = record.get("fingerprint")
            if not isinstance(fingerprint, str) or not fingerprint:
                raise IncompleteRun(f"Runner request lacks fingerprint: {path}")
            runner_fingerprints.add(fingerprint)
            entries.append({
                "path": path.relative_to(rep_root).as_posix(),
                "kind": "request",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        result_paths = sorted(batch_dir.glob("batch_result_*.json"))
        if not result_paths:
            raise IncompleteRun(f"Missing final batch artifact: {batch_dir}")
        final_path = result_paths[-1]
        final = _read_json(final_path)
        fingerprint = final.get("fingerprint")
        if (
            not isinstance(fingerprint, str)
            or not fingerprint
            or final.get("complete") is not True
            or not isinstance(final.get("normalized_predictions"), list)
        ):
            raise IncompleteRun(
                f"Invalid final runner batch artifact: {final_path}"
            )
        runner_fingerprints.add(fingerprint)
        final_predictions.extend(final["normalized_predictions"])
        entries.append({
            "path": final_path.relative_to(rep_root).as_posix(),
            "kind": "final_batch_result",
            "sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        })
    if len(runner_fingerprints) != 1:
        raise IncompleteRun("Runner fingerprint mismatch across attempt tree")
    runner_fingerprint = next(iter(runner_fingerprints))
    canonical = {
        "runner_fingerprint": runner_fingerprint,
        "entries": entries,
    }
    return {
        **canonical,
        "attempt_tree_hash": _sha256(canonical),
        "final_predictions": final_predictions,
    }


class _FakeVerifier:
    async def ground_existing_claims(self, claims, cited_metadata, target):
        output = []
        for claim in claims:
            evidence_id = claim["evidence_ids"][0]
            abstract = str(cited_metadata.get(evidence_id, {}).get("abstract", ""))
            statement = str(claim["text"])
            supported = bool(abstract) and statement.casefold() in abstract.casefold()
            output.append(
                {
                    **claim,
                    "verdict": "supported" if supported else "unsupported",
                    "evidence_results": [
                        {
                            "evidence_id": evidence_id,
                            "verdict": "supported" if supported else "unsupported",
                            "reason": "deterministic offline judge",
                            "quote": statement if supported else "",
                        }
                    ],
                }
            )
        return output


class _FakeHolisticLLM:
    api_key = ""
    model = "deterministic-offline-judge"
    base_url = "offline://"
    timeout = 0

    async def chat_raw(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        marker = "Items:\n"
        payload = prompt.split(marker, 1)[1].split(
            "\n\nReturn JSON:", 1
        )[0]
        items = json.loads(payload)
        results = []
        for item in items:
            statement = str(item["statement"])
            abstract = str(item.get("abstract", ""))
            supported = bool(abstract) and statement.casefold() in abstract.casefold()
            results.append(
                {
                    "variant_id": item["variant_id"],
                    "verdict": "supported" if supported else "unsupported",
                    "reason": "deterministic offline judge",
                    "quote": statement if supported else "",
                }
            )
        content = json.dumps({"items": results}, ensure_ascii=False)
        return ChatRawResult(
            content=content,
            raw_response={"offline": True, "items": results},
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            provider_model=self.model,
        )


async def _execute_split(
    manifest: dict,
    split_name: str,
    output_dir: Path,
    *,
    verifier: object,
    llm: object,
    test_mode: bool,
) -> None:
    fingerprint = _manifest_fingerprint(manifest)
    for test_name, scope, variants in _scopes(manifest, split_name):
        run_root = output_dir / "runs" / test_name / scope
        runner = EvidenceIntegrityRunner(
            variants={split_name: variants},
            output_dir=run_root,
            fingerprint=f"{fingerprint}:{test_name}:{scope}",
            verifier=verifier,
            llm=llm,
            target=manifest["protocol"].get("target", DEFAULT_TARGET),
            batch_size=int(manifest["protocol"].get("batch_size", 20)),
            holistic_batch_size=int(
                manifest["protocol"].get(
                    "holistic_batch_size",
                    DEFAULT_HOLISTIC_BATCH_SIZE,
                )
            ),
            max_request_retries=int(
                manifest["protocol"].get("max_request_retries", 1)
            ),
            test_mode=test_mode,
        )
        for condition in ALL_CONDITIONS:
            await runner.run_condition(condition, split_name)
            repetitions = _latest_predictions(
                run_root, condition, split_name, variants
            )
            for repetition, records in repetitions.items():
                path = (
                    output_dir
                    / "predictions"
                    / split_name
                    / test_name
                    / scope
                    / condition
                    / f"rep_{repetition:02d}.json"
                )
                requested_model = _normalized_model(
                    getattr(llm, "model", "")
                )
                attempt_evidence = _attempt_tree_evidence(
                    run_root,
                    condition,
                    split_name,
                    repetition,
                )
                safe_records = [
                    {
                        key: record[key]
                        for key in (
                            "variant_id",
                            "source_id",
                            "prediction",
                            "schema_failure",
                        )
                    }
                    for record in records
                ]
                payload = {
                    "condition": condition,
                    "split": split_name,
                    "test": test_name,
                    "scope": scope,
                    "repetition": repetition,
                    "test_mode": test_mode,
                    "manifest_sha256": fingerprint,
                    "external_scope_fingerprint": (
                        f"{fingerprint}:{test_name}:{scope}"
                    ),
                    "runner_fingerprint": attempt_evidence[
                        "runner_fingerprint"
                    ],
                    "attempt_tree_hash": attempt_evidence[
                        "attempt_tree_hash"
                    ],
                    "attempt_file_hashes": attempt_evidence["entries"],
                    "requested_model": requested_model,
                    "provider_models": _actual_provider_models(
                        run_root,
                        condition,
                        split_name,
                        repetition,
                        fallback=requested_model,
                    ),
                    "expected_variant_ids": sorted(
                        str(item["variant_id"]) for item in variants
                    ),
                    "frozen_scope_sha256": _sha256(variants),
                    "holistic_batch_size": int(
                        manifest["protocol"].get(
                            "holistic_batch_size",
                            DEFAULT_HOLISTIC_BATCH_SIZE,
                        )
                    ),
                    "records": safe_records,
                }
                payload["artifact_sha256"] = _sha256(payload)
                _write_json(path, payload)


def run_dry_run(output_dir: Path) -> None:
    """Run deterministic fake judges without constructing a network client."""
    output_dir = Path(output_dir)
    manifest = _read_json(output_dir / "MANIFEST.json")
    _validate_manifest_integrity(manifest)
    for split_name in ("pilot", "confirmation"):
        asyncio.run(
            _execute_split(
                manifest,
                split_name,
                output_dir / "dry_run",
                verifier=_FakeVerifier(),
                llm=_FakeHolisticLLM(),
                test_mode=True,
            )
        )
    completion = _completion_payload(
        output_dir,
        manifest,
        kind="dry_run",
        artifact_root=output_dir / "dry_run" / "predictions",
        secret_scan_ok=True,
    )
    _write_json(output_dir / "DRY_RUN_COMPLETE.json", completion)
    try:
        require_clean_secret_scan(
            output_dir,
            secret_values=(),
            report_name="DRY_RUN_SECRET_SCAN.json",
        )
    except SecretScanFailed:
        failed = {
            **completion,
            "complete": False,
            "secret_scan_ok": False,
        }
        failed.pop("completion_sha256", None)
        failed["completion_sha256"] = _sha256(failed)
        _write_json(output_dir / "DRY_RUN_COMPLETE.json", failed)
        raise


def _tree_hash(root: Path) -> dict:
    root = Path(root)
    files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".tmp" not in path.name
    ] if root.exists() else []
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in files
    ]
    return {
        "file_count": len(entries),
        "tree_sha256": _sha256(entries),
    }


def _prepared_execution_contract(manifest: dict) -> dict:
    lock_basis = _confirmation_lock_payload(manifest)
    return {
        "manifest_sha256": _manifest_fingerprint(manifest),
        "source_units_sha256": manifest.get("source_units_sha256"),
        "split_sha256": manifest.get("split_sha256"),
        "pilot_dataset_sha256": _sha256(manifest["splits"]["pilot"]),
        "confirmation_dataset_sha256": _sha256(
            manifest["splits"]["confirmation"]
        ),
        "prompt_contract": lock_basis["prompt_contract"],
        "code_hashes": lock_basis["code_hashes"],
        "runtime": lock_basis["runtime"],
        "inference_contract": lock_basis["inference_contract"],
    }


def _completion_payload(
    output_dir: Path,
    manifest: dict,
    *,
    kind: str,
    artifact_root: Path,
    secret_scan_ok: bool,
) -> dict:
    tree = _tree_hash(artifact_root)
    payload = {
        "kind": kind,
        "complete": tree["file_count"] > 0,
        "manifest_sha256": _manifest_fingerprint(manifest),
        "execution_contract_sha256": _sha256(
            _prepared_execution_contract(manifest)
        ),
        "holistic_batch_size": int(
            manifest.get("protocol", {}).get(
                "holistic_batch_size",
                DEFAULT_HOLISTIC_BATCH_SIZE,
            )
        ),
        "artifact_root": Path(artifact_root).relative_to(output_dir).as_posix(),
        "artifact_file_count": tree["file_count"],
        "artifact_tree_sha256": tree["tree_sha256"],
        "integrity_ok": True,
        "secret_scan_ok": bool(secret_scan_ok),
    }
    payload["completion_sha256"] = _sha256(payload)
    return payload


def _validate_completion(
    output_dir: Path,
    manifest: dict,
    filename: str,
    expected_kind: str,
) -> None:
    path = output_dir / filename
    if not path.is_file():
        raise PaidExecutionBlocked(
            "Confirmation requires completed dry-run and pilot prerequisites"
        )
    value = _read_json(path)
    declared = value.get("completion_sha256")
    unhashed = {
        key: item for key, item in value.items()
        if key != "completion_sha256"
    }
    artifact_root = output_dir / str(value.get("artifact_root", ""))
    tree = _tree_hash(artifact_root)
    if (
        declared != _sha256(unhashed)
        or value.get("kind") != expected_kind
        or value.get("manifest_sha256") != _manifest_fingerprint(manifest)
        or value.get("execution_contract_sha256")
        != _sha256(_prepared_execution_contract(manifest))
        or value.get("holistic_batch_size")
        != int(
            manifest.get("protocol", {}).get(
                "holistic_batch_size",
                DEFAULT_HOLISTIC_BATCH_SIZE,
            )
        )
        or value.get("complete") is not True
        or value.get("integrity_ok") is not True
        or value.get("secret_scan_ok") is not True
        or value.get("artifact_file_count") != tree["file_count"]
        or value.get("artifact_tree_sha256") != tree["tree_sha256"]
    ):
        raise PaidExecutionBlocked(
            "Confirmation dry-run/pilot prerequisite integrity failed"
        )


def _require_confirmation_prerequisites(
    output_dir: Path,
    manifest: dict,
) -> None:
    _validate_completion(
        output_dir, manifest, "DRY_RUN_COMPLETE.json", "dry_run"
    )
    _validate_completion(
        output_dir, manifest, "PILOT_COMPLETE.json", "pilot"
    )


def _confirmation_lock_payload(manifest: dict) -> dict:
    protocol = manifest.get("protocol", {})
    confirmation = manifest["splits"]["confirmation"]
    prompt_contract = {
        "runner_prompt_version": getattr(
            runner_module, "_PROMPT_VERSION", ""
        ),
        "holistic_system_sha256": _sha256(
            runner_module.HOLISTIC_SYSTEM
        ),
        "holistic_user_template_sha256": _sha256(
            inspect.getsource(runner_module._holistic_messages)
        ),
        "full_system_user_template_sha256": _sha256(
            inspect.getsource(ClaimVerifier._ground_claims)
        ),
    }
    code_files = {
        "benchmark_cli_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "benchmark_runner_sha256": hashlib.sha256(
            Path(runner_module.__file__).read_bytes()
        ).hexdigest(),
        "benchmark_stats_sha256": hashlib.sha256(
            Path(stats_module.__file__).read_bytes()
        ).hexdigest(),
        "claim_verifier_sha256": hashlib.sha256(
            Path(inspect.getsourcefile(ClaimVerifier)).read_bytes()
        ).hexdigest(),
        "llm_sha256": hashlib.sha256(
            Path(inspect.getsourcefile(LLMClient)).read_bytes()
        ).hexdigest(),
    }
    return {
        "manifest_sha256": _manifest_fingerprint(manifest),
        "dataset_sha256": _sha256(manifest["splits"]["confirmation"]),
        "prompt_contract": prompt_contract,
        "code_hashes": code_files,
        "runtime": {
            "requested_model": protocol.get("model"),
            "normalized_requested_model": _normalized_model(
                protocol.get("model")
            ),
            "provider_model_normalization_rule_version": (
                PROVIDER_MODEL_NORMALIZATION_RULE_VERSION
            ),
            "transient_transport_retry_policy_version": (
                TRANSIENT_TRANSPORT_RETRY_POLICY_VERSION
            ),
            "transient_transport_exception_classes": list(
                TRANSIENT_TRANSPORT_EXCEPTION_NAMES
            ),
            "terminal_failure_latch_policy_version": (
                TERMINAL_FAILURE_LATCH_POLICY_VERSION
            ),
            "base_url": protocol.get("base_url"),
            "timeout_seconds": protocol.get("timeout_seconds"),
            "max_request_retries": protocol.get("max_request_retries"),
            "batch_size": protocol.get("batch_size"),
            "holistic_batch_size": protocol.get(
                "holistic_batch_size",
                DEFAULT_HOLISTIC_BATCH_SIZE,
            ),
            "judged_repetitions": protocol.get("judged_repetitions"),
            "verifier_timeout_seconds": protocol.get(
                "verifier_timeout_seconds"
            ),
            "verifier_max_concurrency": protocol.get(
                "verifier_max_concurrency"
            ),
        },
        "manifests": {
            "split_sha256": manifest.get("split_sha256"),
            "confirmation_variants_sha256": _sha256(confirmation),
            "sample_manifest_sha256": _sha256(confirmation.get("test1")),
            "bundle_manifest_sha256": _sha256(confirmation.get("bundles")),
            "expected_prediction_scope_sha256": _sha256(
                _expected_prediction_scope(manifest, "confirmation")
            ),
        },
        "inference_contract": {
            "primary_endpoint": protocol.get("primary_endpoint"),
            "test2_primary_endpoint": "full_minus_id_only_bundle_macro_f1",
            "secondary_endpoint": "full_minus_holistic_macro_f1",
            "alpha": protocol.get("alpha", 0.05),
            "publication_gate": {
                "positive_effect": True,
                "ci_excludes_zero": True,
                "permutation_p_below": 0.05,
                "clean_fpr_max": protocol.get("clean_fpr_max", 0.15),
                "integrity_required": True,
                "critical_deviation_forbidden": True,
            },
            "seed": protocol.get("seed", SEED),
        },
    }


def _expected_prediction_scope(
    manifest: dict,
    split_name: str,
) -> list[dict]:
    output = []
    for test_name, scope, variants in _scopes(manifest, split_name):
        expected_ids = sorted(str(item["variant_id"]) for item in variants)
        for condition in ALL_CONDITIONS:
            repetitions = (
                (1, 2, 3) if condition in JUDGED_CONDITIONS else (1,)
            )
            for repetition in repetitions:
                output.append({
                    "path": (
                        f"{split_name}/{test_name}/{scope}/{condition}/"
                        f"rep_{repetition:02d}.json"
                    ),
                    "condition": condition,
                    "repetition": repetition,
                    "expected_variant_ids": expected_ids,
                })
    return output


def _validate_existing_confirmation_lock(
    output_dir: Path,
    manifest: dict,
) -> dict:
    path = Path(output_dir) / "CONFIRMATION_LOCK.json"
    if not path.is_file():
        raise ConfirmationLockMismatch(
            "CONFIRMATION_LOCK.json is required for reporting"
        )
    observed = _read_json(path)
    expected = _confirmation_lock_payload(manifest)
    if observed != expected:
        raise ConfirmationLockMismatch(
            "Confirmation lock is stale or does not match current contract"
        )
    return observed


def _validate_completed_prediction_scope(
    output_dir: Path,
    manifest: dict,
    lock: dict,
) -> None:
    expected = _expected_prediction_scope(manifest, "confirmation")
    if (
        lock.get("manifests", {}).get(
            "expected_prediction_scope_sha256"
        )
        != _sha256(expected)
    ):
        raise ConfirmationLockMismatch(
            "Confirmation lock prediction scope mismatch"
        )
    root = Path(output_dir) / "predictions"
    observed_paths = sorted(
        path.relative_to(root).as_posix()
        for path in root.glob("confirmation/**/*.json")
        if path.is_file()
    ) if root.exists() else []
    expected_paths = sorted(item["path"] for item in expected)
    if observed_paths != expected_paths:
        raise IncompleteRun(
            "Completed confirmation prediction scope is missing or has extras"
        )


def _require_confirmation_lock(
    output_dir: Path,
    manifest: dict,
) -> None:
    path = output_dir / "CONFIRMATION_LOCK.json"
    expected = _confirmation_lock_payload(manifest)
    if path.exists():
        if _read_json(path) != expected:
            raise ConfirmationLockMismatch(
                "Confirmation lock does not match frozen protocol"
            )
        return
    _write_json(path, expected)


def _validate_paid_preflight(manifest: dict, split_name: str) -> None:
    """Validate every paid scope before constructing the transport client."""
    try:
        protocol = manifest["protocol"]
        split = manifest["split"]
        split_data = manifest["splits"][split_name]
        source_ids = list(
            split[
                "pilot_ids"
                if split_name == "pilot"
                else "confirmation_ids"
            ]
        )
        expected_sources = int(
            protocol[
                "pilot_sources"
                if split_name == "pilot"
                else "confirmation_sources"
            ]
        )
        expected_pairs = int(protocol["pairs"][split_name])
        expected_bundles = int(protocol["bundles"][split_name])
    except (KeyError, TypeError, ValueError) as error:
        raise PaidExecutionBlocked(
            "Paid preflight completeness metadata is invalid"
        ) from error
    if (
        len(source_ids) != expected_sources
        or len(source_ids) != len(set(source_ids))
        or not manifest.get("integrity", {}).get("ok")
    ):
        raise PaidExecutionBlocked(
            "Paid preflight completeness gate failed for source split"
        )
    test1 = split_data.get("test1")
    if not isinstance(test1, dict) or set(test1) != set(CORRUPTION_TYPES):
        raise PaidExecutionBlocked(
            "Paid preflight completeness gate failed for Test 1 scopes"
        )
    for kind in CORRUPTION_TYPES:
        records = test1[kind]
        if not isinstance(records, list) or len(records) != expected_pairs * 2:
            raise PaidExecutionBlocked(
                "Paid preflight completeness gate failed for paired samples"
            )
        pair_roles: dict[str, set[str]] = {}
        for record in records:
            if not isinstance(record, dict):
                raise PaidExecutionBlocked(
                    "Paid preflight completeness gate found malformed pair"
                )
            pair_id = str(record.get("pair_id", ""))
            role = str(record.get("pair_role", ""))
            if not pair_id or role not in {"clean_control", "corruption"}:
                raise PaidExecutionBlocked(
                    "Paid preflight completeness gate found malformed pair"
                )
            pair_roles.setdefault(pair_id, set()).add(role)
        if (
            len(pair_roles) != expected_pairs
            or any(
                roles != {"clean_control", "corruption"}
                for roles in pair_roles.values()
            )
        ):
            raise PaidExecutionBlocked(
                "Paid preflight completeness gate found incomplete pairs"
            )
    bundles = split_data.get("bundles")
    if not isinstance(bundles, list) or len(bundles) != expected_bundles:
        raise PaidExecutionBlocked(
            "Paid preflight completeness gate failed for bundles"
        )
    bundle_sources = [
        str(member.get("source_id", ""))
        for bundle in bundles
        if isinstance(bundle, dict)
        for member in bundle.get("members", [])
        if isinstance(member, dict)
    ]
    if (
        len(bundle_sources) != expected_bundles * 4
        or len(bundle_sources) != len(set(bundle_sources))
        or any(not source_id for source_id in bundle_sources)
    ):
        raise PaidExecutionBlocked(
            "Paid preflight completeness gate failed for bundle members"
        )


async def _run_paid_lifecycle(
    manifest: dict,
    split_name: str,
    output_dir: Path,
    *,
    api_key: str,
) -> None:
    """Create, use, and close the async transport on one event loop."""
    model = str(manifest["protocol"].get("model", DEFAULT_MODEL))
    llm = LLMClient(
        model=model,
        api_key=api_key,
        base_url=str(
            manifest["protocol"].get(
                "base_url", "https://open.bohrium.com/openapi/v1"
            )
        ),
        timeout=float(
            manifest["protocol"].get("timeout_seconds", 180)
        ),
    )
    primary_error: BaseException | None = None
    try:
        verifier = ClaimVerifier(
            llm,
            task="verifier",
            call_timeout_seconds=float(
                manifest["protocol"].get(
                    "verifier_timeout_seconds", 120
                )
            ),
            max_concurrency=int(
                manifest["protocol"].get(
                    "verifier_max_concurrency", 3
                )
            ),
        )
        await _execute_split(
            manifest,
            split_name,
            output_dir,
            verifier=verifier,
            llm=llm,
            test_mode=False,
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            await llm.aclose()
        except BaseException as close_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"LLM client close also failed: {close_error!r}"
            )
            raise primary_error from close_error


def run_paid(
    split_name: str,
    output_dir: Path,
    *,
    test_mode: bool = False,
) -> None:
    """Run a paid split only after all pre-call safety gates pass."""
    if split_name not in {"pilot", "confirmation"}:
        raise ValueError("Paid split must be pilot or confirmation")
    if test_mode or os.environ.get("EVIDENCE_INTEGRITY_TEST_MODE"):
        raise PaidExecutionBlocked("Paid commands reject test mode")
    api_key = os.environ.get("BH_API_KEY", "")
    if not api_key:
        raise PaidExecutionBlocked("BH_API_KEY is required for paid commands")
    output_dir = Path(output_dir)
    manifest = _read_json(output_dir / "MANIFEST.json")
    try:
        _validate_manifest_integrity(manifest)
    except IncompleteRun as error:
        raise PaidExecutionBlocked(
            "Paid preflight manifest integrity check failed"
        ) from error
    split_data = manifest["splits"][split_name]
    if not split_data.get("yield_report", {}).get("all_ready", True):
        raise PaidExecutionBlocked("Mutation-yield completeness gate failed")
    repository_scan = _repository_secret_scan(
        Path(__file__).resolve().parent,
        output_dir,
        secret_value=api_key,
    )
    _write_json(
        output_dir / f"PRE_{split_name.upper()}_SECRET_SCAN.json",
        repository_scan,
    )
    if not repository_scan["ok"]:
        raise PaidExecutionBlocked(
            "Repository-wide pre-call secret scan failed"
        )
    if split_name == "pilot":
        _validate_completion(
            output_dir, manifest, "DRY_RUN_COMPLETE.json", "dry_run"
        )
    if split_name == "confirmation":
        _require_confirmation_prerequisites(output_dir, manifest)
    _validate_paid_preflight(manifest, split_name)
    if split_name == "confirmation":
        _require_confirmation_lock(output_dir, manifest)
    asyncio.run(
        _run_paid_lifecycle(
            manifest,
            split_name,
            output_dir,
            api_key=api_key,
        )
    )
    if split_name == "pilot":
        scan = require_clean_secret_scan(
            output_dir,
            secret_values=(api_key,),
            report_name="PILOT_SECRET_SCAN.json",
        )
        _write_json(
            output_dir / "PILOT_COMPLETE.json",
            _completion_payload(
                output_dir,
                manifest,
                kind="pilot",
                artifact_root=output_dir / "predictions" / "pilot",
                secret_scan_ok=scan["ok"],
            ),
        )


def scan_artifacts(
    output_dir: Path,
    *,
    secret_values: tuple[str, ...] = (),
    excluded_names: tuple[str, ...] = (),
) -> dict:
    output_dir = Path(output_dir)
    hits = []
    excluded = set(excluded_names)
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name in excluded or ".tmp" in path.name:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        reasons = []
        for secret in secret_values:
            if secret and secret in text:
                reasons.append("current_api_key_value")
        if _CREDENTIAL_ASSIGNMENT.search(text):
            reasons.append("credential_assignment_pattern")
        if reasons:
            hits.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "reasons": sorted(set(reasons)),
                }
            )
    return {
        "ok": not hits,
        "files_scanned": sum(
            1
            for path in output_dir.rglob("*")
            if path.is_file() and path.name not in excluded
        ),
        "hits": hits,
    }


def require_clean_secret_scan(
    output_dir: Path,
    *,
    secret_values: tuple[str, ...] = (),
    report_name: str = "SECRET_SCAN.json",
) -> dict:
    result = scan_artifacts(
        output_dir,
        secret_values=secret_values,
        excluded_names=(report_name,),
    )
    _write_json(Path(output_dir) / report_name, result)
    if not result["ok"]:
        raise SecretScanFailed("Secret scan found a blocked artifact")
    return result


def _repository_secret_scan(
    project_root: Path,
    output_dir: Path,
    *,
    secret_value: str,
) -> dict:
    hits = []
    scanned = 0
    artifact_suffixes = {
        ".json", ".md", ".txt", ".log", ".env", ".yaml", ".yml"
    }
    for path in sorted(
        item for item in Path(project_root).rglob("*") if item.is_file()
    ):
        parts = set(path.parts)
        if "__pycache__" in parts or ".git" in parts:
            continue
        try:
            relative_output = path.relative_to(output_dir)
            if (
                "attempts" in relative_output.parts
                and path.name.startswith(("request_", "batch_result_"))
            ):
                continue
        except ValueError:
            pass
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        scanned += 1
        reasons = []
        if secret_value and secret_value in text:
            reasons.append("current_api_key_value")
        if (
            path.suffix.casefold() in artifact_suffixes
            and _CREDENTIAL_ASSIGNMENT.search(text)
        ):
            reasons.append("credential_assignment_pattern")
        if reasons:
            hits.append({
                "path": path.relative_to(project_root).as_posix(),
                "reasons": sorted(set(reasons)),
            })
    return {"ok": not hits, "files_scanned": scanned, "hits": hits}


def evaluate_publication_gate(
    *,
    effect: float,
    ci_low: float,
    ci_high: float,
    permutation_p: float,
    clean_fpr: float,
    integrity_ok: bool,
    critical_deviation: bool,
) -> dict:
    checks = {
        "positive_primary_effect": effect > 0,
        "confidence_interval_excludes_zero": ci_low > 0 and ci_high > 0,
        "permutation_p_below_0_05": permutation_p < 0.05,
        "clean_fpr_at_most_15_percent": clean_fpr <= 0.15,
        "dataset_and_secret_integrity": bool(integrity_ok),
        "no_critical_protocol_deviation": not critical_deviation,
    }
    return {
        "main_text_eligible": all(checks.values()),
        "checks": checks,
    }


def _load_prediction(
    output_dir: Path,
    split_name: str,
    test_name: str,
    scope: str,
    condition: str,
    repetition: int,
    manifest: dict,
) -> list[dict]:
    path = (
        output_dir
        / "predictions"
        / split_name
        / test_name
        / scope
        / condition
        / f"rep_{repetition:02d}.json"
    )
    value = _read_json(path)
    if not isinstance(value, dict):
        raise IncompleteRun(f"Invalid prediction artifact: {path}")
    declared_hash = value.get("artifact_sha256")
    unhashed = {
        key: item
        for key, item in value.items()
        if key != "artifact_sha256"
    }
    if declared_hash != _sha256(unhashed):
        raise IncompleteRun(f"Prediction artifact hash mismatch: {path}")
    if test_name == "test1":
        frozen = manifest["splits"][split_name]["test1"][scope]
    elif test_name == "test2" and scope == "bundles":
        frozen = _bundle_items(
            manifest["splits"][split_name]["bundles"]
        )
    else:
        raise IncompleteRun("Prediction scope is not frozen in manifest")
    fingerprint = _manifest_fingerprint(manifest)
    requested_model = _normalized_model(
        manifest["protocol"].get("model", "")
    )
    expected_ids = sorted(str(item["variant_id"]) for item in frozen)
    expected_metadata = {
        "condition": condition,
        "split": split_name,
        "test": test_name,
        "scope": scope,
        "repetition": repetition,
        "manifest_sha256": fingerprint,
        "external_scope_fingerprint": f"{fingerprint}:{test_name}:{scope}",
        "requested_model": requested_model,
        "expected_variant_ids": expected_ids,
        "frozen_scope_sha256": _sha256(frozen),
        "holistic_batch_size": int(
            manifest["protocol"].get(
                "holistic_batch_size",
                DEFAULT_HOLISTIC_BATCH_SIZE,
            )
        ),
    }
    if any(value.get(key) != expected for key, expected in expected_metadata.items()):
        raise IncompleteRun(
            f"Prediction artifact fingerprint/scope mismatch: {path}"
        )
    provider_models = value.get("provider_models")
    if (
        not isinstance(provider_models, list)
        or not provider_models
        or any(
            _normalized_model(item) != requested_model
            for item in provider_models
        )
    ):
        raise IncompleteRun(f"Prediction provider model mismatch: {path}")
    run_root = output_dir / "runs" / test_name / scope
    attempt_evidence = _attempt_tree_evidence(
        run_root, condition, split_name, repetition
    )
    if (
        value.get("runner_fingerprint")
        != attempt_evidence["runner_fingerprint"]
        or value.get("attempt_tree_hash")
        != attempt_evidence["attempt_tree_hash"]
        or value.get("attempt_file_hashes")
        != attempt_evidence["entries"]
    ):
        raise IncompleteRun(
            f"Prediction attempt tree or runner fingerprint mismatch: {path}"
        )
    if value.get("test_mode"):
        raise IncompleteRun("Test-mode predictions cannot enter final report")
    records = value.get("records")
    if not isinstance(records, list):
        raise IncompleteRun(f"Invalid prediction artifact: {path}")
    by_frozen = {str(item["variant_id"]): item for item in frozen}
    observed_ids = [
        str(item.get("variant_id", ""))
        for item in records
        if isinstance(item, dict)
    ]
    if (
        len(observed_ids) != len(records)
        or len(observed_ids) != len(set(observed_ids))
        or sorted(observed_ids) != expected_ids
    ):
        raise IncompleteRun(
            f"Prediction IDs missing, extra, or duplicated: {path}"
        )
    joined = []
    final_by_id = {}
    for final_record in attempt_evidence["final_predictions"]:
        if not isinstance(final_record, dict):
            raise IncompleteRun("Runner final prediction is malformed")
        final_id = str(final_record.get("variant_id", ""))
        if not final_id or final_id in final_by_id:
            raise IncompleteRun(
                "Runner final predictions have missing or duplicate IDs"
            )
        final_by_id[final_id] = final_record
    if sorted(final_by_id) != expected_ids:
        raise IncompleteRun(
            "Runner final prediction IDs differ from frozen scope"
        )
    for record in records:
        variant = by_frozen[str(record["variant_id"])]
        final_record = final_by_id[str(record["variant_id"])]
        normalized_fields = (
            "variant_id",
            "source_id",
            "prediction",
            "schema_failure",
        )
        if (
            record.get("source_id") != variant.get("source_id")
            or record.get("prediction") not in {"supported", "unsupported"}
            or not isinstance(record.get("schema_failure"), bool)
            or "gold" in record
            or "variant_type" in record
            or any(
                record.get(key) != final_record.get(key)
                for key in normalized_fields
            )
        ):
            raise IncompleteRun(
                "Prediction content does not match frozen manifest or "
                f"validated runner final labels: {path}"
            )
        item = {
            **record,
            "gold": variant["gold"],
            "variant_type": variant["variant_type"],
        }
        for key in ("pair_id", "pair_role", "paired_corruption_type"):
            if key in variant:
                item[key] = variant[key]
        joined.append(item)
    return joined


def _voted_scope(
    output_dir: Path,
    split_name: str,
    test_name: str,
    scope: str,
    condition: str,
    manifest: dict,
) -> dict:
    if condition == "id_only":
        records = _load_prediction(
            output_dir, split_name, test_name, scope, condition, 1, manifest
        )
        return {"records": records, "disagreement_rate": 0.0}
    repetitions = {
        repetition: _load_prediction(
            output_dir,
            split_name,
            test_name,
            scope,
            condition,
            repetition,
            manifest,
        )
        for repetition in (1, 2, 3)
    }
    return majority_vote_predictions(repetitions)


def _compute_complete_results(output_dir: Path, manifest: dict) -> dict:
    split_name = "confirmation"
    expected_pairs = int(manifest["protocol"]["pairs"][split_name])
    condition_records = {condition: [] for condition in ALL_CONDITIONS}
    stability = {}
    for condition in ALL_CONDITIONS:
        rates = []
        for kind in CORRUPTION_TYPES:
            voted = _voted_scope(
                output_dir, split_name, "test1", kind, condition, manifest
            )
            records = voted["records"]
            if len(records) != expected_pairs * 2:
                raise IncompleteRun(
                    f"{condition}/{kind} confirmation count mismatch"
                )
            condition_records[condition].extend(records)
            rates.append(voted["disagreement_rate"])
        stability[condition] = round(sum(rates) / len(rates), 6)
    full = condition_records["full"]
    holistic = condition_records["holistic"]
    id_only = condition_records["id_only"]
    full_metrics = binary_metrics(full)
    primary_effect = macro_f1_difference(full, id_only)
    bootstrap = cluster_bootstrap_difference(
        full, id_only, seed=SEED, draws=10_000
    )
    permutation = cluster_permutation_test(
        full, id_only, seed=SEED, draws=100_000
    )
    holistic_bootstrap = cluster_bootstrap_difference(
        full, holistic, seed=SEED, draws=10_000
    )
    holistic_permutation = cluster_permutation_test(
        full, holistic, seed=SEED, draws=100_000
    )
    full_id_mcnemar = exact_mcnemar(full, id_only)
    full_holistic_mcnemar = exact_mcnemar(full, holistic)
    test1 = {
        "metrics": {
            condition: binary_metrics(records)
            for condition, records in condition_records.items()
        },
        "full_minus_id_only": {
            "macro_f1_difference": primary_effect,
            "mcnemar": full_id_mcnemar,
            "bootstrap": bootstrap,
            "permutation": permutation,
        },
        "full_minus_holistic": {
            "macro_f1_difference": macro_f1_difference(full, holistic),
            "mcnemar": full_holistic_mcnemar,
            "bootstrap": holistic_bootstrap,
            "permutation": holistic_permutation,
        },
    }
    bundles = manifest["splits"][split_name].get("bundles", [])
    expected_bundles = int(manifest["protocol"]["bundles"][split_name])
    if len(bundles) != expected_bundles:
        raise IncompleteRun("Confirmation bundle manifest is incomplete")
    split_manifest = _split_manifest(manifest["split"], split_name)
    test2 = {}
    bundle_predictions = {}
    for condition in ALL_CONDITIONS:
        voted = _voted_scope(
            output_dir,
            split_name,
            "test2",
            "bundles",
            condition,
            manifest,
        )
        test2[condition] = bundle_metrics(
            voted["records"], bundles, split_manifest
        )
        bundle_predictions[condition] = voted["records"]
        stability[f"{condition}_test2"] = voted["disagreement_rate"]
    test2_primary_bootstrap = bundle_bootstrap_difference(
        bundle_predictions["full"],
        bundle_predictions["id_only"],
        bundles,
        split_manifest,
        seed=SEED,
        draws=10_000,
    )
    test2_primary_permutation = bundle_permutation_test(
        bundle_predictions["full"],
        bundle_predictions["id_only"],
        bundles,
        split_manifest,
        seed=SEED,
        draws=100_000,
    )
    test2_secondary_bootstrap = bundle_bootstrap_difference(
        bundle_predictions["full"],
        bundle_predictions["holistic"],
        bundles,
        split_manifest,
        seed=SEED,
        draws=10_000,
    )
    test2_secondary_permutation = bundle_permutation_test(
        bundle_predictions["full"],
        bundle_predictions["holistic"],
        bundles,
        split_manifest,
        seed=SEED,
        draws=100_000,
    )
    test2["full_minus_id_only"] = {
        "macro_f1_difference": bundle_macro_f1_difference(
            bundle_predictions["full"],
            bundle_predictions["id_only"],
            bundles,
            split_manifest,
        ),
        "bootstrap": test2_primary_bootstrap,
        "permutation": test2_primary_permutation,
    }
    test2["full_minus_holistic"] = {
        "macro_f1_difference": bundle_macro_f1_difference(
            bundle_predictions["full"],
            bundle_predictions["holistic"],
            bundles,
            split_manifest,
        ),
        "bootstrap": test2_secondary_bootstrap,
        "permutation": test2_secondary_permutation,
    }
    secondary_p_values = holm_adjust({
        "test1_full_id_mcnemar": full_id_mcnemar["p_value"],
        "test1_full_holistic_mcnemar": full_holistic_mcnemar["p_value"],
        "test1_full_holistic_permutation": holistic_permutation["p_value"],
        "test2_full_holistic_permutation": (
            test2_secondary_permutation["p_value"]
        ),
    })
    return {
        "primary": {
            "effect": primary_effect,
            "bootstrap": bootstrap,
            "permutation": permutation,
        },
        "full_metrics": full_metrics,
        "test1": test1,
        "test2": test2,
        "secondary_p_values": secondary_p_values,
        "stability": stability,
    }


def _chinese_report(results: dict) -> str:
    primary = results["primary"]
    bootstrap = primary["bootstrap"]
    gate = results["publication_gate"]
    destination = "正文" if gate["main_text_eligible"] else "补充材料"
    effect_pp = primary["effect_percentage_points"]
    ci_low_pp = bootstrap["ci_low"] * 100
    ci_high_pp = bootstrap["ci_high"] * 100
    return (
        "# 证据完整性基准结果\n\n"
        "在预注册的独立确认集中，完整 Claim Grounding 相对于仅检查文献"
        f"标识符的对照，其宏平均 F1 差值为 {effect_pp:.1f} 个百分点"
        f"（95% 聚类自助法置信区间 {ci_low_pp:.1f}–"
        f"{ci_high_pp:.1f} 个百分点；聚类置换检验 "
        f"p={primary['permutation']['p_value']:.4f}）。完整条件的干净"
        f"样本假阳性率为 {results['full_metrics']['clean_false_positive_rate']:.1%}。"
        f"\n\n依据预注册发表门槛，本结果应进入{destination}。重复判断先经"
        "多数投票聚合，未被作为独立统计观测；整体判断按每批至多 "
        f"{results['runtime']['holistic_batch_size']} 项执行。\n"
    )


def build_final_report(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    manifest = _read_json(output_dir / "MANIFEST.json")
    _validate_manifest_integrity(manifest)
    confirmation_lock = _validate_existing_confirmation_lock(
        output_dir, manifest
    )
    _validate_completed_prediction_scope(
        output_dir, manifest, confirmation_lock
    )
    secret_scan = require_clean_secret_scan(
        output_dir,
        secret_values=(os.environ.get("BH_API_KEY", ""),),
    )
    results = _compute_complete_results(output_dir, manifest)
    results["primary"]["effect_percentage_points"] = round(
        results["primary"]["effect"] * 100,
        6,
    )
    for section in ("test1", "test2"):
        for comparison in (
            "full_minus_id_only",
            "full_minus_holistic",
        ):
            value = results.get(section, {}).get(comparison)
            if (
                isinstance(value, dict)
                and isinstance(
                    value.get("macro_f1_difference"), (int, float)
                )
            ):
                value["effect_percentage_points"] = round(
                    value["macro_f1_difference"] * 100,
                    6,
                )
    deviations = manifest.get("protocol_deviations", [])
    critical = any(
        isinstance(item, dict) and item.get("critical") is True
        for item in deviations
    )
    primary = results["primary"]
    results["publication_gate"] = evaluate_publication_gate(
        effect=primary["effect"],
        ci_low=primary["bootstrap"]["ci_low"],
        ci_high=primary["bootstrap"]["ci_high"],
        permutation_p=primary["permutation"]["p_value"],
        clean_fpr=results["full_metrics"]["clean_false_positive_rate"],
        integrity_ok=bool(
            manifest.get("integrity", {}).get("ok") and secret_scan["ok"]
        ),
        critical_deviation=critical,
    )
    results["protocol_deviations"] = deviations
    results["runtime"] = confirmation_lock["runtime"]
    results["secret_scan"] = {
        "ok": secret_scan["ok"],
        "hits": secret_scan["hits"],
    }
    try:
        _write_json(output_dir / "FINAL_RESULTS.json", results)
        _atomic_write(
            output_dir / "MAIN_TEXT_RESULTS_ZH.md",
            _chinese_report(results).encode("utf-8"),
        )
    except SecretScanFailed as error:
        relative = Path(error.artifact_path).relative_to(output_dir).as_posix()
        blocked = {
            "ok": False,
            "files_scanned": 0,
            "hits": [{
                "path": relative,
                "reasons": list(error.reasons),
            }],
            "repository": {"ok": False, "files_scanned": 0, "hits": []},
        }
        _write_json(output_dir / "POST_FINAL_SECRET_SCAN.json", blocked)
        raise
    artifact_post = scan_artifacts(
        output_dir,
        secret_values=(os.environ.get("BH_API_KEY", ""),),
        excluded_names=("POST_FINAL_SECRET_SCAN.json",),
    )
    repository_post = _repository_secret_scan(
        Path(__file__).resolve().parent,
        output_dir,
        secret_value=os.environ.get("BH_API_KEY", ""),
    )
    post = {
        **artifact_post,
        "ok": artifact_post["ok"] and repository_post["ok"],
        "repository": repository_post,
    }
    _write_json(output_dir / "POST_FINAL_SECRET_SCAN.json", post)
    if not post["ok"]:
        raise SecretScanFailed("Post-final secret scan found a blocked artifact")
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the evidence-integrity benchmark"
    )
    parser.add_argument(
        "command",
        choices=("prepare", "dry-run", "pilot", "confirm", "report"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "evidence_integrity_benchmark",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = Path(__file__).resolve().parent
    if args.command == "prepare":
        prepare_benchmark(project_root, args.output_dir)
    elif args.command == "dry-run":
        run_dry_run(args.output_dir)
    elif args.command == "pilot":
        run_paid("pilot", args.output_dir)
    elif args.command == "confirm":
        run_paid("confirmation", args.output_dir)
    else:
        build_final_report(args.output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        IncompleteRun,
        ConfirmationLockMismatch,
        SecretScanFailed,
        PaidExecutionBlocked,
    ) as error:
        print(f"Benchmark blocked: {error}", file=sys.stderr)
        raise SystemExit(2)
