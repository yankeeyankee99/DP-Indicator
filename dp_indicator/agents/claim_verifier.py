"""Claim-level grounding for evidence-linked biomedical hypotheses.

This DP-Indicator fix10 module treats
model-produced descriptions as claims to verify, never as source quotations.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
from typing import Iterable


def _stable_claim_id(hypothesis_id: str, origin: dict, text: str) -> str:
    payload = "|".join(
        (
            hypothesis_id,
            str(origin.get("kind", "")),
            str(origin.get("axis_index", "")),
            str(origin.get("step_index", "")),
            str(origin.get("bucket", "")),
            str(origin.get("item_index", "")),
            text.strip(),
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"CLM-{digest}"


def build_claim_candidates(hypotheses: list[dict]) -> list[dict]:
    """Collect auditable claim candidates from causal steps and mappings."""
    candidates: list[dict] = []
    for hypothesis in hypotheses:
        hid = str(
            hypothesis.get("hypothesis_id")
            or hypothesis.get("indication")
            or ""
        )
        chain = hypothesis.get("causal_chain", {})
        if isinstance(chain, dict) and "mechanism_axes" in chain:
            for axis_index, axis in enumerate(chain.get("mechanism_axes", [])):
                for step_index, step in enumerate(axis.get("steps", [])):
                    text = str(step.get("mechanism", "")).strip()
                    if not text:
                        continue
                    origin = {
                        "kind": "causal_step",
                        "axis_index": axis_index,
                        "step_index": step_index,
                        "layer": str(
                            step.get("layer", step.get("level", ""))
                        ),
                    }
                    candidates.append(
                        {
                            "claim_id": _stable_claim_id(hid, origin, text),
                            "hypothesis_id": hid,
                            "origin": origin,
                            "text": text,
                            "expected_relation": "support",
                            "evidence_ids": [
                                str(evidence_id)
                                for evidence_id in step.get(
                                    "evidence_ids",
                                    [],
                                )
                                if evidence_id
                            ],
                        }
                    )

        mapping = hypothesis.get("evidence_mapping", {})
        if not isinstance(mapping, dict):
            continue
        relations = {
            "positive_evidence": "support",
            "indirect_evidence": "support",
            "contradicting_evidence": "contradict",
        }
        limits = {
            "positive_evidence": 12,
            "indirect_evidence": 3,
            "contradicting_evidence": 5,
        }
        for bucket, expected_relation in relations.items():
            for item_index, item in enumerate(
                mapping.get(bucket, [])[: limits[bucket]]
            ):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("rationale", "")).strip()
                evidence_id = item.get("id") or item.get("evidence_id")
                if not text or not evidence_id:
                    continue
                origin = {
                    "kind": "evidence_mapping",
                    "bucket": bucket,
                    "item_index": item_index,
                }
                candidates.append(
                    {
                        "claim_id": _stable_claim_id(hid, origin, text),
                        "hypothesis_id": hid,
                        "origin": origin,
                        "text": text,
                        "expected_relation": expected_relation,
                        "evidence_ids": [str(evidence_id)],
                    }
                )
    return candidates


def _normalize_for_matching(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def quote_is_in_source(quote: str, source: str) -> bool:
    """Return true only when a non-empty quote occurs in the source text."""
    normalized_quote = _normalize_for_matching(quote)
    normalized_source = _normalize_for_matching(source)
    return bool(
        normalized_quote
        and normalized_source
        and normalized_quote in normalized_source
    )


def validate_grounding_record(record: dict, source: str) -> dict:
    """Enforce source availability and deterministic quote provenance."""
    checked = copy.deepcopy(record)
    checked.setdefault("reason", "")
    checked["quote_verified"] = False
    if not str(source).strip():
        checked["verdict"] = "unverifiable"
        checked["reason"] = "source_unavailable"
        return checked

    verdict = str(checked.get("verdict", "unverifiable")).lower()
    quote = str(checked.get("quote", ""))
    if verdict in {"supported", "partial", "contradicted"}:
        if quote_is_in_source(quote, source):
            checked["quote_verified"] = True
        else:
            checked["verdict"] = "unsupported"
            checked["reason"] = "quote_not_found_in_source"
    return checked


def aggregate_claim_verdict(records: Iterable[dict]) -> str:
    """Aggregate pair-level verdicts without hiding conflicting evidence."""
    verdicts = {
        str(record.get("verdict", "unverifiable")).lower()
        for record in records
    }
    has_support = bool(verdicts & {"supported", "partial"})
    has_contradiction = "contradicted" in verdicts
    if has_support and has_contradiction:
        return "mixed"
    if "supported" in verdicts:
        return "supported"
    if "partial" in verdicts:
        return "partial"
    if has_contradiction:
        return "contradicted"
    if "unsupported" in verdicts:
        return "unsupported"
    return "unverifiable"


def aggregate_parent_claims(claims: list[dict]) -> list[dict]:
    """Aggregate atomic verdicts back to each original claim/evidence pair."""
    groups: dict[tuple[str, str, str], dict] = {}
    for claim in claims:
        hypothesis_id = str(claim.get("hypothesis_id", ""))
        parent_claim_id = str(
            claim.get("parent_claim_id") or claim.get("claim_id") or ""
        )
        for result in claim.get("evidence_results", []):
            if result.get("evidence_role") == "bridge_evidence":
                continue
            evidence_id = str(result.get("evidence_id", ""))
            if not evidence_id:
                continue
            key = (hypothesis_id, parent_claim_id, evidence_id)
            group = groups.setdefault(
                key,
                {
                    "hypothesis_id": hypothesis_id,
                    "parent_claim_id": parent_claim_id,
                    "evidence_id": evidence_id,
                    "origin": copy.deepcopy(claim.get("origin", {})),
                    "expected_relation": claim.get(
                        "expected_relation",
                        "support",
                    ),
                    "results": [],
                },
            )
            group["results"].append(
                {
                    "claim_id": claim.get("claim_id", ""),
                    **copy.deepcopy(result),
                }
            )

    aggregates = []
    weights = {
        "supported": 1.0,
        "partial": 0.5,
        "unsupported": 0.0,
        "contradicted": 0.0,
    }
    for group in groups.values():
        results = group.pop("results")
        verifiable = [
            result
            for result in results
            if result.get("verdict") in weights
        ]
        unverifiable_count = sum(
            1
            for result in results
            if result.get("verdict") == "unverifiable"
        )
        coverage = (
            sum(weights[result["verdict"]] for result in verifiable)
            / len(verifiable)
            if verifiable
            else None
        )
        has_contradiction = any(
            result.get("verdict") == "contradicted"
            for result in results
        )
        if not verifiable:
            decision = "unverifiable"
        elif has_contradiction:
            decision = "remove"
        elif coverage >= 0.60:
            decision = "retain_supported"
        elif coverage >= 0.30:
            decision = "retain_partial"
        else:
            decision = "remove"

        verified_spans = []
        seen_spans = set()
        for result in results:
            if result.get("quote_verified") is not True:
                continue
            quote = str(result.get("quote", ""))
            span_key = (group["evidence_id"], quote)
            if not quote or span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            verified_spans.append(
                {
                    "claim_id": result.get("claim_id", ""),
                    "evidence_id": group["evidence_id"],
                    "quote": quote,
                    "quote_verified": True,
                }
            )

        aggregates.append(
            {
                **group,
                "coverage": coverage,
                "verifiable_count": len(verifiable),
                "unverifiable_count": unverifiable_count,
                "unverifiable_ratio": (
                    unverifiable_count / len(results) if results else 1.0
                ),
                "has_contradiction": has_contradiction,
                "decision": decision,
                "verified_spans": verified_spans,
            }
        )
    return aggregates


def aggregate_bridge_claims(
    claims: list[dict],
    retrieval: dict,
) -> list[dict]:
    """Project each selected bridge set as one parent-level aggregate."""
    known_parents = {
        (
            str(claim.get("hypothesis_id", "")),
            str(claim.get("parent_claim_id") or claim.get("claim_id") or ""),
        )
        for claim in claims
    }
    aggregates: list[dict] = []
    for group in (retrieval or {}).get("groups", []) or []:
        if not isinstance(group, dict):
            continue
        hypothesis_id = str(group.get("hypothesis_id", ""))
        parent_claim_id = str(group.get("parent_claim_id", ""))
        if (hypothesis_id, parent_claim_id) not in known_parents:
            continue
        selected = group.get("selected") or {}
        if not isinstance(selected, dict):
            continue
        evidence_ids = list(
            dict.fromkeys(
                str(evidence_id)
                for evidence_id in selected.get("evidence_ids", [])
                if evidence_id
            )
        )
        verified_spans = []
        seen_spans: set[tuple[str, str, str]] = set()
        for span in selected.get("verified_spans", []) or []:
            if not isinstance(span, dict):
                continue
            span_key = (
                str(span.get("claim_id", "")),
                str(span.get("evidence_id", "")),
                str(span.get("quote", "")),
            )
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            normalized = copy.deepcopy(span)
            normalized["evidence_role"] = "bridge_evidence"
            verified_spans.append(normalized)
        coverage = selected.get("coverage")
        aggregates.append(
            {
                "aggregate_kind": "bridge_set",
                "hypothesis_id": hypothesis_id,
                "parent_claim_id": parent_claim_id,
                "origin": copy.deepcopy(group.get("origin", {})),
                "evidence_ids": evidence_ids,
                "coverage": (
                    round(float(coverage), 6)
                    if isinstance(coverage, (int, float))
                    else coverage
                ),
                "decision": selected.get("decision", "remove"),
                "verified_spans": verified_spans,
            }
        )
    return aggregates


def merge_bridge_grounding(claims: list[dict], retrieval: dict) -> None:
    """Attach selected bridge judgments while preserving direct judgments."""
    claims_by_id = {
        str(claim.get("claim_id", "")): claim
        for claim in claims
        if claim.get("claim_id")
    }
    verdict_rank = {
        "unverifiable": 0,
        "unsupported": 1,
        "partial": 2,
        "supported": 3,
    }
    for group in (retrieval or {}).get("groups", []) or []:
        if not isinstance(group, dict):
            continue
        selected = group.get("selected") or {}
        if selected.get("decision") not in {
            "retain_supported",
            "retain_partial",
        }:
            continue
        selected_ids = {
            str(evidence_id)
            for evidence_id in selected.get("evidence_ids", [])
            if evidence_id
        }
        for grounded in group.get("grounded_claims", []) or []:
            if not isinstance(grounded, dict):
                continue
            claim = claims_by_id.get(str(grounded.get("claim_id", "")))
            if claim is None:
                continue
            bridge_results = []
            for result in grounded.get("evidence_results", []) or []:
                if (
                    not isinstance(result, dict)
                    or str(result.get("evidence_id", "")) not in selected_ids
                ):
                    continue
                normalized = copy.deepcopy(result)
                normalized["evidence_role"] = "bridge_evidence"
                bridge_results.append(normalized)
            if not bridge_results:
                continue
            existing = claim.setdefault("evidence_results", [])
            seen = {
                (
                    str(item.get("evidence_id", "")),
                    str(item.get("verdict", "")),
                    str(item.get("quote", "")),
                )
                for item in existing
                if isinstance(item, dict)
            }
            for result in bridge_results:
                key = (
                    str(result.get("evidence_id", "")),
                    str(result.get("verdict", "")),
                    str(result.get("quote", "")),
                )
                if key not in seen:
                    existing.append(result)
                    seen.add(key)

            direct_verdict = str(claim.get("verdict", "unverifiable")).lower()
            if direct_verdict != "unverifiable":
                continue
            best = max(
                (
                    str(result.get("verdict", "unverifiable")).lower()
                    for result in bridge_results
                    if result.get("quote_verified") is True
                    or result.get("verdict") in {"unsupported", "unverifiable"}
                ),
                key=lambda verdict: verdict_rank.get(verdict, 0),
                default="unverifiable",
            )
            claim["verdict"] = best


def _hypothesis_steps(hypothesis: dict) -> list[dict]:
    chain = hypothesis.get("causal_chain", {})
    if not isinstance(chain, dict):
        return []
    if "mechanism_axes" in chain:
        return [
            step
            for axis in chain.get("mechanism_axes", [])
            for step in axis.get("steps", [])
            if isinstance(step, dict)
        ]
    return [value for value in chain.values() if isinstance(value, dict)]


def recheck_sanitized_hypotheses(
    hypotheses: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Apply final deterministic invariants without mutating caller data."""
    checked = copy.deepcopy(hypotheses)
    audit: list[dict] = []
    for hypothesis in checked:
        hid = str(
            hypothesis.get("hypothesis_id")
            or hypothesis.get("indication")
            or ""
        )
        for step in _hypothesis_steps(hypothesis):
            verified_ids = {
                str(span.get("evidence_id", ""))
                for span in step.get("verified_spans", [])
                if span.get("quote_verified") is True
                and span.get("evidence_id")
            }
            cited_ids = {
                str(evidence_id)
                for evidence_id in step.get("evidence_ids", [])
                if evidence_id
            }
            if step.get("status") == "supported" and not (
                cited_ids & verified_ids
            ):
                step["status"] = "inferred"
                audit.append(
                    {
                        "hypothesis_id": hid,
                        "action": "downgrade_step",
                        "location": str(
                            step.get("layer", step.get("level", ""))
                        ),
                        "reason": "no_verified_supporting_span",
                    }
                )
    return checked, audit


class ClaimVerifier:
    """Decompose model output into claims and ground each claim in sources."""

    def __init__(
        self,
        llm: object,
        task: str = "reasoner",
        call_timeout_seconds: float = 120,
        max_concurrency: int = 3,
    ):
        self.llm = llm
        self.task = task
        self.call_timeout_seconds = call_timeout_seconds
        self.max_concurrency = max(1, max_concurrency)

    async def _structured(self, messages: list[dict], **kwargs):
        return await asyncio.wait_for(
            self.llm.structured(messages, **kwargs),
            timeout=self.call_timeout_seconds,
        )

    async def verify(
        self,
        hypotheses: list[dict],
        cited_metadata: dict[str, dict],
        target: str = "",
    ) -> dict:
        candidates = build_claim_candidates(hypotheses)
        claims = await self._decompose_candidates(candidates, target)
        await self._ground_claims(claims, cited_metadata, target)
        return {
            "claims": claims,
            "by_hypothesis": self._group_by_hypothesis(claims),
        }

    async def ground_existing_claims(
        self,
        claims: list[dict],
        cited_metadata: dict[str, dict],
        target: str = "",
    ) -> list[dict]:
        """Ground already-atomic claims without decomposing or mutating them."""
        grounded = copy.deepcopy(claims)
        await self._ground_claims(grounded, cited_metadata, target)
        return grounded

    async def _decompose_candidates(
        self,
        candidates: list[dict],
        target: str,
    ) -> list[dict]:
        if not candidates:
            return []
        decomposed_by_id = {}
        chunk_size = 20
        chunks = [
            candidates[start : start + chunk_size]
            for start in range(0, len(candidates), chunk_size)
        ]
        semaphore = asyncio.Semaphore(self.max_concurrency)
        completed = 0

        async def decompose_chunk(chunk: list[dict]) -> dict[str, list[str]]:
            nonlocal completed
            payload = [
                {
                    "candidate_id": candidate["claim_id"],
                    "text": candidate["text"],
                }
                for candidate in chunk
            ]
            prompt = f"""## Task
Split each biomedical statement into the smallest independently verifiable
claims. Preserve the original meaning and wording as closely as possible.
Do not add mechanisms, diseases, outcomes, or certainty.

Target: {target}
Candidates:
{json.dumps(payload, ensure_ascii=False)}

Return JSON:
{{"items": [{{"candidate_id": "...", "claims": ["...", "..."]}}]}}"""
            async with semaphore:
                try:
                    result, _ = await self._structured(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "You decompose biomedical statements for "
                                    "evidence verification. Return JSON only."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=2048,
                        task=self.task,
                        temperature=0,
                    )
                except Exception:
                    result = {}
            chunk_results: dict[str, list[str]] = {}
            if isinstance(result, dict):
                for item in result.get("items", []):
                    if not isinstance(item, dict):
                        continue
                    candidate_id = str(item.get("candidate_id", ""))
                    texts = [
                        str(text).strip()
                        for text in item.get("claims", [])
                        if str(text).strip()
                    ]
                    if candidate_id and texts:
                        chunk_results[candidate_id] = texts
            completed += 1
            print(
                "  [heartbeat] ClaimVerifier: decomposed "
                f"{completed}/{len(chunks)} claim batches",
                flush=True,
            )
            return chunk_results

        for chunk_results in await asyncio.gather(
            *[decompose_chunk(chunk) for chunk in chunks]
        ):
            decomposed_by_id.update(chunk_results)

        claims: list[dict] = []
        for candidate in candidates:
            texts = decomposed_by_id.get(
                candidate["claim_id"],
                [candidate["text"]],
            )
            for index, text in enumerate(texts):
                claim = copy.deepcopy(candidate)
                digest = hashlib.sha256(
                    (
                        candidate["claim_id"]
                        + "|"
                        + str(index)
                        + "|"
                        + text
                    ).encode("utf-8")
                ).hexdigest()[:16]
                claim["claim_id"] = f"CLM-{digest}"
                claim["parent_claim_id"] = candidate["claim_id"]
                claim["text"] = text
                claims.append(claim)
        return claims

    async def _ground_claims(
        self,
        claims: list[dict],
        cited_metadata: dict[str, dict],
        target: str,
    ) -> None:
        claims_by_evidence: dict[str, list[dict]] = {}
        for claim in claims:
            claim["evidence_results"] = []
            for evidence_id in claim.get("evidence_ids", []):
                metadata = cited_metadata.get(evidence_id, {})
                source = str(metadata.get("abstract", ""))
                if not source:
                    claim["evidence_results"].append(
                        validate_grounding_record(
                            {
                                "evidence_id": evidence_id,
                                "verdict": "unverifiable",
                                "quote": "",
                                "reason": "source_unavailable",
                            },
                            source,
                        )
                    )
                    continue
                claims_by_evidence.setdefault(evidence_id, []).append(claim)

        semaphore = asyncio.Semaphore(self.max_concurrency)
        completed_grounding = 0
        grounding_batches = [
            (evidence_id, source_claims[start : start + 6])
            for evidence_id, source_claims in claims_by_evidence.items()
            for start in range(0, len(source_claims), 6)
        ]

        async def ground_one(evidence_id: str, source_claims: list[dict]):
            nonlocal completed_grounding
            metadata = cited_metadata.get(evidence_id, {})
            source = str(metadata.get("abstract", ""))
            async def request(batch: list[dict], retry: bool = False):
                payload = [
                    {
                        "claim_id": claim["claim_id"],
                        "claim": claim["text"],
                        "expected_relation": claim["expected_relation"],
                    }
                    for claim in batch
                ]
                retry_instruction = (
                    "This is a retry for previously omitted claim IDs. "
                    "Return one item for every ID.\n\n"
                    if retry
                    else ""
                )
                prompt = f"""## Task
{retry_instruction}
Judge each claim only against the supplied article abstract. Do not use
outside knowledge. For supported, partial, or contradicted verdicts, copy the
smallest exact quote from the abstract that proves the judgment.

Target: {target}
Evidence ID: {evidence_id}
Title: {metadata.get("title", "")}
Abstract:
{source}

Claims:
{json.dumps(payload, ensure_ascii=False)}

Allowed verdicts: supported, partial, unsupported, contradicted.
Return JSON:
{{"items": [{{"claim_id": "...", "verdict": "...", "quote": "...", "reason": "..."}}]}}"""
                try:
                    result, _ = await self._structured(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "You are a conservative biomedical claim "
                                    "grounding reviewer. Return JSON only."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=1024 if retry else 2048,
                        task=self.task,
                        temperature=0,
                    )
                except Exception:
                    result = {}
                return (
                    result.get("items", [])
                    if isinstance(result, dict)
                    else []
                )

            async with semaphore:
                returned = await request(source_claims)
                expected_ids = {
                    claim["claim_id"] for claim in source_claims
                }
                returned_by_id = {
                    str(item.get("claim_id", "")): item
                    for item in returned
                    if isinstance(item, dict)
                    and str(item.get("claim_id", "")) in expected_ids
                }
                missing_claims = [
                    claim
                    for claim in source_claims
                    if claim["claim_id"] not in returned_by_id
                ]
                if missing_claims:
                    retry_items = await request(missing_claims, retry=True)
                    missing_ids = {
                        claim["claim_id"] for claim in missing_claims
                    }
                    returned_by_id.update(
                        {
                            str(item.get("claim_id", "")): item
                            for item in retry_items
                            if isinstance(item, dict)
                            and str(item.get("claim_id", ""))
                            in missing_ids
                        }
                    )
            checked_records = []
            for claim in source_claims:
                raw = returned_by_id.get(claim["claim_id"])
                if not isinstance(raw, dict):
                    raw = {
                        "verdict": "unverifiable",
                        "quote": "",
                        "reason": "grounding_result_missing",
                    }
                record = {
                    "evidence_id": evidence_id,
                    "verdict": raw.get("verdict", "unverifiable"),
                    "quote": raw.get("quote", ""),
                    "reason": raw.get("reason", ""),
                }
                checked_records.append(
                    validate_grounding_record(record, source)
                )
            completed_grounding += 1
            if (
                completed_grounding % 5 == 0
                or completed_grounding == len(grounding_batches)
            ):
                print(
                    "  [heartbeat] ClaimVerifier: grounded "
                    f"{completed_grounding}/{len(grounding_batches)} "
                    "claim batches",
                    flush=True,
                )
            return source_claims, checked_records

        grounded_groups = await asyncio.gather(
            *[
                ground_one(evidence_id, source_claims)
                for evidence_id, source_claims in grounding_batches
            ]
        )
        for source_claims, records in grounded_groups:
            for claim, record in zip(source_claims, records):
                claim["evidence_results"].append(record)

        for claim in claims:
            claim["verdict"] = aggregate_claim_verdict(
                claim["evidence_results"]
            )

    @staticmethod
    def _group_by_hypothesis(claims: list[dict]) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = {}
        for claim in claims:
            grouped.setdefault(claim["hypothesis_id"], []).append(claim)
        return grouped
