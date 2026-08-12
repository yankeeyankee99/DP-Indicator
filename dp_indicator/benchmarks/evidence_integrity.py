"""Frozen source-unit construction for the evidence-integrity benchmark.

This module is deliberately isolated from the production pipeline. It only
reads archived stage artifacts and creates deterministic in-memory manifests.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import os
import random
import re
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx

from dp_indicator.agents.claim_verifier import ClaimVerifier
from dp_indicator.core.llm import ProviderHTTPStatusError


MUTATIONS = (
    "invalid_id",
    "citation_swap",
    "causal_overclaim",
    "entity_swap",
)
RESERVED_BENCHMARK_PMID = "PMID:00000000"
MIN_MEANINGFUL_PILOT = 20
MAX_CONTENT_TOKEN_OVERLAP = 0.20
MAX_DONOR_DIAGNOSTIC_SAMPLES_PER_REASON = 3
PILOT_PAIRS_PER_CORRUPTION = 10
CONFIRMATION_PAIRS_PER_CORRUPTION = 30
PILOT_BUNDLE_COUNT = 9
CONFIRMATION_BUNDLE_COUNT = 30
BUNDLE_SOURCE_COUNT = 4
DEFAULT_HOLISTIC_BATCH_SIZE = 5
MAX_HOLISTIC_BATCH_SIZE = 5
HOLISTIC_SYSTEM = (
    "Judge whether each biomedical statement is directly supported by only "
    "the supplied source. Return JSON only; do not use outside knowledge."
)
JUDGED_CONDITIONS = ("full", "holistic")
_GROUNDING_VERDICTS = {
    "supported",
    "partial",
    "mixed",
    "unsupported",
    "contradicted",
    "unverifiable",
}
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(accesskey|api[_-]?key|authorization)\b\s*[:=]\s*"
    r"(?:bearer\s+)?[^\s,;\"'}]+"
)

_DIRECTION_ANCHOR_FAMILIES = {
    "decrease": (
        "attenuated", "attenuates", "alleviated", "alleviates",
        "ameliorated", "ameliorates", "improved", "improves", "reduced",
        "reduces", "reduce", "reducing", "reduction", "decreased",
        "decreases", "lower", "downregulated", "inhibits", "inhibited",
        "inhibiting", "inhibition", "suppresses", "suppressed", "blocks",
        "blocked", "impairs", "impaired", "prevented", "prevents",
        "low levels",
    ),
    "increase": (
        "exacerbated", "exacerbates", "worsened", "worsens", "increased",
        "increases", "increase", "increasing", "elevated", "higher",
        "upregulated", "promotes", "promoted", "promoting", "activation",
        "activates", "activated", "enhances", "enhanced", "caused",
        "causes", "high levels",
    ),
}

_OVERCLAIM_ANCHORS = (
    {
        "canonical": "randomized phase III result",
        "aliases": (
            "randomized phase III",
            "randomised phase III",
            "randomized controlled trial",
            "randomised controlled trial",
            "randomized trial",
            "randomised trial",
            "randomized study",
            "randomised study",
            "randomized phase 3",
            "randomised phase 3",
            "phase III randomized",
            "phase III randomised",
            "phase 3 randomized",
            "phase 3 randomised",
            "phase III trial",
            "phase 3 trial",
            "pivotal phase III",
        ),
    },
    {
        "canonical": "approved treatment status",
        "aliases": (
            "regulatory approval",
            "regulatory approved",
            "approved treatment",
            "approved therapy",
            "approved indication",
            "marketing authorization",
            "marketing authorisation",
            "FDA approved",
            "EMA approved",
        ),
    },
)

_ENTITY_RULES = (
    {
        "canonical": "Kv1.3",
        "aliases": ("Kv1.3", "KCNA3"),
        "replacement": "Nav1.5",
        "replacement_aliases": ("Nav1.5", "SCN5A"),
    },
    {
        "canonical": "KCa3.1",
        "aliases": ("KCa3.1", "KCNN4", "IKCa1"),
        "replacement": "Nav1.5",
        "replacement_aliases": ("Nav1.5", "SCN5A"),
    },
    {
        "canonical": "macrophage",
        "aliases": ("macrophage", "macrophages", "RAW264.7"),
        "replacement": "neutrophil",
        "replacement_aliases": ("neutrophil", "neutrophils"),
    },
    {
        "canonical": "microglia",
        "aliases": ("microglia", "microglial"),
        "replacement": "astrocyte",
        "replacement_aliases": ("astrocyte", "astrocytes", "astrocytic"),
    },
    {
        "canonical": "T cell",
        "aliases": ("T cell", "T cells", "T-cell", "T-cells"),
        "replacement": "hepatocyte",
        "replacement_aliases": ("hepatocyte", "hepatocytes"),
    },
    {
        "canonical": "B cell",
        "aliases": ("B cell", "B cells", "B-cell", "B-cells"),
        "replacement": "hepatocyte",
        "replacement_aliases": ("hepatocyte", "hepatocytes"),
    },
    {
        "canonical": "IL-6",
        "aliases": ("IL-6", "IL6", "interleukin-6"),
        "replacement": "IL-10",
        "replacement_aliases": ("IL-10", "IL10", "interleukin-10"),
    },
    {
        "canonical": "TNF-α",
        "aliases": ("TNF-α", "TNFα", "TNF-alpha"),
        "replacement": "IL-10",
        "replacement_aliases": ("IL-10", "IL10", "interleukin-10"),
    },
    {
        "canonical": "NF-κB",
        "aliases": ("NF-κB", "NFκB", "NF-kB"),
        "replacement": "Wnt",
        "replacement_aliases": ("Wnt", "WNT"),
    },
    {
        "canonical": "ERK1/2",
        "aliases": ("ERK1/2", "ERK1", "ERK2", "ERK"),
        "replacement": "Wnt",
        "replacement_aliases": ("Wnt", "WNT"),
    },
)

_FIXED_ANCHOR_PHRASES = tuple(
    dict.fromkeys(
        phrase
        for rule in _ENTITY_RULES
        for phrase in rule["aliases"]
    )
) + tuple(
    dict.fromkeys(
        phrase
        for family in _DIRECTION_ANCHOR_FAMILIES.values()
        for phrase in family
    )
)

_CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "the", "to", "was",
    "were", "with", "while",
}


class MutationIneligible(Exception):
    """Expected failure when a corruption cannot be proven safe."""

    def __init__(
        self,
        operator: str,
        reason: str,
        details: dict | None = None,
    ) -> None:
        super().__init__(f"{operator}: {reason}")
        self.operator = operator
        self.reason = reason
        self.details = copy.deepcopy(details or {})


class FingerprintMismatch(RuntimeError):
    """Raised when an attempt ledger cannot be safely resumed."""


class ConcurrentRunError(RuntimeError):
    """Raised when another runner owns the condition/split lock."""


class IncompleteCondition(RuntimeError):
    """Raised when a condition lacks a complete technical result."""


class MalformedResponseError(ValueError):
    """Raised internally when a provider returns malformed JSON."""


class ProviderModelMismatch(ValueError):
    """Raised when the provider did not execute the frozen requested model."""


class BenchmarkTerminalFailure(RuntimeError):
    """Raised when a benchmark request is blocked by a terminal latch."""


PROVIDER_MODEL_NORMALIZATION_RULE_VERSION = "bohrium-routing-v1"


def _normalized_provider_model(value: object) -> str:
    """Remove only documented, case-sensitive Bohrium routing prefixes.

    ``bh:`` may precede ``BohrClaw/``. Prefix and model comparisons remain
    case-sensitive; no arbitrary prefix, suffix, or surrounding whitespace is
    accepted.
    """
    if not isinstance(value, str):
        return ""
    normalized = value
    if normalized.startswith("bh:"):
        normalized = normalized[3:]
    if normalized.startswith("BohrClaw/"):
        normalized = normalized[len("BohrClaw/"):]
    return normalized


def _normalize_for_matching(value: Any) -> str:
    """Match the production quote rule: collapse whitespace and case-fold."""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _normalize_evidence_id(value: Any) -> str:
    return re.sub(r"\s+", "", str(value)).upper()


def _phrase_pattern(phrase: str) -> re.Pattern:
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    return re.compile(
        rf"(?<![\w]){escaped}(?![\w])",
        flags=re.IGNORECASE,
    )


def _phrase_matches(text: str, phrases: tuple[str, ...]) -> list[str]:
    return [
        phrase
        for phrase in phrases
        if _phrase_pattern(phrase).search(str(text))
    ]


def _replace_token_phrase(
    text: str,
    original: str,
    replacement: str,
) -> tuple[str, re.Match] | None:
    match = _phrase_pattern(original).search(text)
    if match is None:
        return None
    changed = text[: match.start()] + replacement + text[match.end() :]
    return changed, match


def _content_tokens(text: str) -> set[str]:
    tokens = {
        token.casefold()
        for token in re.findall(
            r"[^\W_]+(?:[.+/-][^\W_]+)*",
            str(text),
            flags=re.UNICODE,
        )
    }
    return {
        token
        for token in tokens
        if len(token) >= 3 and token not in _CONTENT_STOPWORDS
    }


def _normalize_topic(value: Any) -> str:
    separated = re.sub(r"[-_/]+", " ", str(value))
    return " ".join(sorted(_content_tokens(separated)))


def _extract_essential_anchors(statement: str) -> list[dict]:
    anchors: list[dict] = []
    for rule in _ENTITY_RULES:
        matches = _phrase_matches(statement, rule["aliases"])
        if matches:
            anchors.append(
                {
                    "kind": "entity",
                    "canonical": rule["canonical"],
                    "aliases": list(rule["aliases"]),
                    "matched": matches,
                }
            )
    for family_name, phrases in _DIRECTION_ANCHOR_FAMILIES.items():
        matches = _phrase_matches(statement, phrases)
        if matches:
            anchors.append(
                {
                    "kind": "direction",
                    "canonical": family_name,
                    "aliases": list(phrases),
                    "matched": matches,
                }
            )
    for numeric in re.findall(
        r"(?<![\w])\d+(?:\.\d+)?(?:\s*%|\s*(?:mg|µg|ug|ng|mm|µm|um|nm))?",
        statement,
        flags=re.IGNORECASE,
    ):
        normalized = re.sub(r"\s+", "", numeric).casefold()
        anchors.append(
            {
                "kind": "numeric",
                "canonical": normalized,
                "aliases": [normalized],
                "matched": [numeric],
            }
        )
    return anchors


def _citation_profile(unit: dict) -> dict:
    donor_text = f"{unit.get('title', '')} {unit.get('abstract', '')}"
    return {
        "anchors": _extract_essential_anchors(
            str(unit.get("statement") or "")
        ),
        "statement_tokens": _content_tokens(
            str(unit.get("statement") or "")
        ),
        "donor_text": donor_text,
        "donor_tokens": _content_tokens(donor_text),
        "normalized_topic": _normalize_topic(
            unit.get("hypothesis_id", "")
        ),
        "present_fixed_phrases": {
            phrase
            for phrase in _FIXED_ANCHOR_PHRASES
            if _phrase_pattern(phrase).search(donor_text)
        },
    }


def _numeric_anchor_matches(anchor: dict, text: str) -> list[str]:
    parsed = re.fullmatch(
        r"(\d+(?:\.\d+)?)(%|mg|µg|ug|ng|mm|µm|um|nm)?",
        str(anchor["canonical"]),
        flags=re.IGNORECASE,
    )
    if parsed is None:
        raise ValueError(
            f"Malformed numeric anchor: {anchor['canonical']}"
        )
    number, unit = parsed.groups()
    suffix = rf"\s*{re.escape(unit)}" if unit else ""
    pattern = re.compile(
        rf"(?<![\w.]){re.escape(number)}{suffix}(?![\w])",
        flags=re.IGNORECASE,
    )
    return [match.group(0) for match in pattern.finditer(str(text))]


def _anchor_checks(
    anchors: list[dict],
    donor_profile: dict,
) -> list[dict]:
    checks = []
    present_fixed = donor_profile["present_fixed_phrases"]
    for anchor in anchors:
        if anchor["kind"] == "numeric":
            present_aliases = _numeric_anchor_matches(
                anchor,
                donor_profile["donor_text"],
            )
        else:
            present_aliases = [
                alias
                for alias in anchor["aliases"]
                if alias in present_fixed
            ]
        checks.append(
            {
                "kind": anchor["kind"],
                "canonical": anchor["canonical"],
                "aliases": copy.deepcopy(anchor["aliases"]),
                "present": bool(present_aliases),
                "present_aliases": present_aliases,
            }
        )
    return checks


def _assess_citation_donor(
    unit: dict,
    donor: dict,
    *,
    source_profile: dict | None = None,
    donor_profile: dict | None = None,
) -> dict:
    """Conservatively prove that a donor cannot directly support the quote."""
    source_profile = source_profile or _citation_profile(unit)
    donor_profile = donor_profile or _citation_profile(donor)
    anchors = source_profile["anchors"]
    anchor_checks = _anchor_checks(anchors, donor_profile)
    source_tokens = source_profile["statement_tokens"]
    donor_tokens = donor_profile["donor_tokens"]
    union = source_tokens | donor_tokens
    overlap = (
        len(source_tokens & donor_tokens) / len(union)
        if union
        else 1.0
    )
    source_topic = source_profile["normalized_topic"]
    donor_topic = donor_profile["normalized_topic"]
    checks = {
        "different_source": (
            donor.get("source_id") != unit.get("source_id")
        ),
        "essential_anchors_present": bool(anchors),
        "all_anchors_absent": bool(anchors)
        and all(not check["present"] for check in anchor_checks),
        "low_content_token_overlap": overlap <= MAX_CONTENT_TOKEN_OVERLAP,
        "distinct_normalized_topic": bool(source_topic)
        and bool(donor_topic)
        and source_topic != donor_topic,
    }
    eligible = all(checks.values())
    if not checks["different_source"]:
        reason = "same_source"
    elif not checks["essential_anchors_present"]:
        reason = "no_essential_anchors"
    elif not checks["all_anchors_absent"]:
        reason = "essential_anchor_present_in_donor"
    elif not checks["low_content_token_overlap"]:
        reason = "high_content_token_overlap"
    elif not checks["distinct_normalized_topic"]:
        reason = "same_or_missing_topic"
    else:
        reason = ""
    return {
        "eligible": eligible,
        "reason": reason,
        "essential_anchors": anchors,
        "anchor_checks": anchor_checks,
        "source_content_tokens": sorted(source_tokens),
        "donor_content_tokens": sorted(donor_tokens),
        "content_token_overlap": round(overlap, 6),
        "content_token_overlap_threshold": MAX_CONTENT_TOKEN_OVERLAP,
        "normalized_source_topic": source_topic,
        "normalized_donor_topic": donor_topic,
        "checks": checks,
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _load_json_with_hash(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), _sha256_bytes(raw)


def _metadata_by_id(items: list[dict]) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        evidence_id = item.get("evidence_id") or item.get("id")
        normalized_id = _normalize_evidence_id(evidence_id)
        if normalized_id:
            metadata.setdefault(normalized_id, item)
    return metadata


def _abstract_from_metadata(metadata: dict) -> str:
    return str(
        metadata.get("abstract")
        or metadata.get("abstract_snippet")
        or ""
    ).strip()


def _source_id(evidence_id: str, quote: str, abstract: str) -> str:
    abstract_hash = _sha256_text(abstract)
    payload = "|".join(
        (
            _normalize_evidence_id(evidence_id),
            _normalize_for_matching(quote),
            abstract_hash,
        )
    )
    return _sha256_text(payload)


def extract_source_units(repeat_root: Path) -> list[dict]:
    """Extract eligible immutable quote/source pairs from archived repeats."""
    repeat_root = Path(repeat_root)
    units: dict[str, dict] = {}
    seen_quote_pairs: set[tuple[str, str]] = set()

    for repeat_number in range(1, 7):
        repeat = repeat_root / f"rep_{repeat_number:02d}"
        verification_path = (
            repeat / "checkpoints" / "stages" / "verification.json"
        )
        pre_path = (
            repeat / "checkpoints" / "stages" / "pre_verification.json"
        )
        if not verification_path.is_file() or not pre_path.is_file():
            continue

        verification, verification_hash = _load_json_with_hash(
            verification_path
        )
        pre_verification, pre_hash = _load_json_with_hash(pre_path)
        raw_report = verification.get("raw_verification_report", {})
        metadata = _metadata_by_id(
            pre_verification.get("evidence_pool", [])
        )
        existence = {
            _normalize_evidence_id(key): value
            for key, value in (
                raw_report.get("v1_id_existence", {}) or {}
            ).items()
        }
        claims = (
            raw_report.get("claim_grounding", {}).get("claims", []) or []
        )

        for claim in claims:
            if not isinstance(claim, dict):
                continue
            for result in claim.get("evidence_results", []) or []:
                if not isinstance(result, dict):
                    continue
                if (
                    result.get("evidence_role") == "bridge_evidence"
                    or claim.get("evidence_role") == "bridge_evidence"
                ):
                    continue

                evidence_id = _normalize_evidence_id(
                    result.get("evidence_id")
                )
                if not re.fullmatch(r"PMID:\d+", evidence_id):
                    continue
                existence_record = existence.get(evidence_id)
                if (
                    not isinstance(existence_record, dict)
                    or existence_record.get("exists") is not True
                ):
                    continue

                source_metadata = metadata.get(evidence_id, {})
                if (
                    source_metadata.get("evidence_role")
                    == "bridge_evidence"
                ):
                    continue
                abstract = _abstract_from_metadata(source_metadata)
                quote = str(result.get("quote") or "").strip()
                normalized_quote = _normalize_for_matching(quote)
                if (
                    not abstract
                    or not normalized_quote
                    or normalized_quote
                    not in _normalize_for_matching(abstract)
                ):
                    continue

                quote_pair = (evidence_id, normalized_quote)
                if quote_pair in seen_quote_pairs:
                    continue
                seen_quote_pairs.add(quote_pair)

                source_id = _source_id(evidence_id, quote, abstract)
                units.setdefault(
                    source_id,
                    {
                        "source_id": source_id,
                        "repeat": repeat.name,
                        "hypothesis_id": str(
                            claim.get("hypothesis_id") or ""
                        ),
                        "claim_id": str(claim.get("claim_id") or ""),
                        "original_claim": str(claim.get("text") or ""),
                        "origin": copy.deepcopy(claim.get("origin", {})),
                        "evidence_id": evidence_id,
                        "exists": True,
                        "title": str(
                            source_metadata.get("title") or ""
                        ).strip(),
                        "abstract": abstract,
                        "statement": quote,
                        "gold": "supported",
                        "source_hashes": {
                            "verification_sha256": verification_hash,
                            "pre_verification_sha256": pre_hash,
                            "abstract_sha256": _sha256_text(abstract),
                            "quote_sha256": _sha256_text(quote),
                        },
                    },
                )

    return sorted(units.values(), key=lambda item: item["source_id"])


def _seed_for(seed: int, source_id: str, purpose: str) -> int:
    digest = _sha256_text(f"{seed}|{source_id}|{purpose}")
    return int(digest[:16], 16)


def _variant_id(source_id: str, variant_type: str, payload: str) -> str:
    return _sha256_text(f"{source_id}|{variant_type}|{payload}")


def _base_variant(unit: dict, variant_type: str) -> dict:
    variant = copy.deepcopy(unit)
    variant["variant_type"] = variant_type
    variant["exists"] = True
    variant["gold"] = (
        "supported" if variant_type == "clean_quote" else "unsupported"
    )
    variant["mutation_audit"] = {}
    return variant


def _clean_variant(unit: dict) -> dict:
    variant = _base_variant(unit, "clean_quote")
    variant["variant_id"] = _variant_id(
        unit["source_id"],
        "clean_quote",
        unit["statement"],
    )
    return variant


def _invalid_id_variant(unit: dict, frozen_ids: set[str]) -> dict:
    if RESERVED_BENCHMARK_PMID in frozen_ids:
        raise ValueError(
            "Reserved benchmark PMID collides with the frozen V1 universe"
        )
    variant = _base_variant(unit, "invalid_id")
    original_id = variant["evidence_id"]
    variant.update(
        {
            "evidence_id": RESERVED_BENCHMARK_PMID,
            "title": "",
            "abstract": "",
            "exists": False,
            "mutation_audit": {
                "operator": "invalid_id",
                "original_evidence_id": original_id,
                "replacement_evidence_id": RESERVED_BENCHMARK_PMID,
                "operational_gold": (
                    "source_unavailable_in_frozen_v1_universe"
                ),
                "validation_basis": (
                    "reserved_benchmark_id_absent_from_frozen_v1"
                ),
                "validation": {
                    "pmid_syntax": True,
                    "reserved_benchmark_namespace": True,
                    "absent_from_frozen_ids": (
                        RESERVED_BENCHMARK_PMID not in frozen_ids
                    ),
                    "source_removed": True,
                    "not_a_real_world_nonexistence_claim": True,
                },
            },
        }
    )
    variant["variant_id"] = _variant_id(
        unit["source_id"],
        "invalid_id",
        RESERVED_BENCHMARK_PMID,
    )
    return variant


def _citation_swap_variant(
    unit: dict,
    donor: dict,
    assessment: dict,
) -> dict:
    if not assessment.get("eligible"):
        raise ValueError("Citation swap received an unsafe donor")
    variant = _base_variant(unit, "citation_swap")
    variant.update(
        {
            "evidence_id": donor["evidence_id"],
            "title": donor.get("title", ""),
            "abstract": donor["abstract"],
            "exists": True,
            "mutation_audit": {
                "operator": "citation_swap",
                "original_evidence_id": unit["evidence_id"],
                "donor_source_id": donor["source_id"],
                "donor_evidence_id": donor["evidence_id"],
                "donor_hypothesis_id": donor.get("hypothesis_id", ""),
                "donor_source_hashes": copy.deepcopy(
                    donor.get("source_hashes", {})
                ),
                "essential_anchors": copy.deepcopy(
                    assessment["essential_anchors"]
                ),
                "anchor_checks": copy.deepcopy(
                    assessment["anchor_checks"]
                ),
                "source_content_tokens": copy.deepcopy(
                    assessment["source_content_tokens"]
                ),
                "donor_content_tokens": copy.deepcopy(
                    assessment["donor_content_tokens"]
                ),
                "content_token_overlap": assessment[
                    "content_token_overlap"
                ],
                "content_token_overlap_threshold": assessment[
                    "content_token_overlap_threshold"
                ],
                "normalized_source_topic": assessment[
                    "normalized_source_topic"
                ],
                "normalized_donor_topic": assessment[
                    "normalized_donor_topic"
                ],
                "validation": copy.deepcopy(assessment["checks"]),
            },
        }
    )
    variant["variant_id"] = _variant_id(
        unit["source_id"],
        "citation_swap",
        donor["source_id"],
    )
    return variant


def _causal_overclaim_variant(unit: dict) -> dict:
    exact_quote = str(unit["statement"])
    full_source = f"{unit.get('title', '')} {unit['abstract']}"
    indication = re.sub(
        r"\s+",
        " ",
        str(unit.get("hypothesis_id") or ""),
    ).strip()
    normalized_indication = _normalize_topic(indication)
    if not normalized_indication:
        raise MutationIneligible(
            "causal_overclaim",
            "missing_indication",
            {"source_id": unit["source_id"]},
        )

    anchor_checks = []
    for anchor in _OVERCLAIM_ANCHORS:
        matches = _phrase_matches(full_source, anchor["aliases"])
        anchor_checks.append(
            {
                "canonical": anchor["canonical"],
                "aliases": list(anchor["aliases"]),
                "present": bool(matches),
                "present_aliases": matches,
            }
        )
    if any(item["present"] for item in anchor_checks):
        raise MutationIneligible(
            "causal_overclaim",
            "compound_anchor_in_source",
            {
                "source_id": unit["source_id"],
                "anchor_checks": anchor_checks,
            },
        )

    disease_specific_aliases = (
        f"randomized phase III trial for {indication}",
        f"randomised phase III trial for {indication}",
        f"approved treatment for {indication}",
        f"approved therapy for {indication}",
        f"regulatory approval for {indication}",
    )
    disease_specific_matches = _phrase_matches(
        full_source,
        disease_specific_aliases,
    )
    disease_specific_checks = [
        {
            "canonical": "disease-specific clinical conclusion",
            "alias": alias,
            "present": alias in disease_specific_matches,
        }
        for alias in disease_specific_aliases
    ]
    if disease_specific_matches:
        raise MutationIneligible(
            "causal_overclaim",
            "disease_specific_conclusion_in_source",
            {
                "source_id": unit["source_id"],
                "matches": disease_specific_matches,
            },
        )

    overclaim_sentence = (
        "A randomized phase III clinical trial established that Kv1.3 "
        "inhibition has regulatory approval as an approved treatment for "
        f"{indication}."
    )
    if _normalize_for_matching(overclaim_sentence) in (
        _normalize_for_matching(full_source)
    ):
        raise MutationIneligible(
            "causal_overclaim",
            "overclaim_proposition_in_source",
            {"source_id": unit["source_id"]},
        )
    separator = " " if re.search(r"[.!?]\s*$", exact_quote) else ". "
    statement = exact_quote + separator + overclaim_sentence
    clean_exact_in_source = (
        _normalize_for_matching(exact_quote)
        in _normalize_for_matching(unit["abstract"])
    )
    if not clean_exact_in_source:
        raise ValueError(
            f"Clean quote is not in source: {unit['source_id']}"
        )

    variant = _base_variant(unit, "causal_overclaim")
    variant["statement"] = statement
    variant["mutation_audit"] = {
        "operator": "causal_overclaim",
        "clean_quote": exact_quote,
        "overclaim_sentence": overclaim_sentence,
        "normalized_indication": normalized_indication,
        "overclaim_anchors": copy.deepcopy(list(_OVERCLAIM_ANCHORS)),
        "anchor_checks": anchor_checks,
        "disease_specific_conclusion_checks": disease_specific_checks,
        "missing_conjuncts": [
            item["canonical"] for item in _OVERCLAIM_ANCHORS
        ],
        "validation": {
            "nonempty_normalized_indication": True,
            "clean_quote_preserved_exactly": statement.startswith(
                exact_quote + separator
            ),
            "clean_quote_exact_in_source": clean_exact_in_source,
            "overclaim_is_separate_sentence": separator in {" ", ". "},
            "overclaim_anchors_absent_from_source": True,
            "disease_specific_conclusions_absent_from_source": True,
            "no_randomized_phase_iii_anchor": True,
            "no_regulatory_approval_anchor": True,
            "added_proposition_absent_from_source": True,
            "clearly_missing_conjunct": True,
            "token_boundary_validation": True,
        },
    }
    variant["variant_id"] = _variant_id(
        unit["source_id"],
        "causal_overclaim",
        statement,
    )
    return variant


def _entity_swap_variant(unit: dict) -> dict:
    exact_quote = str(unit["statement"])
    frozen_source = f"{unit.get('title', '')} {unit['abstract']}"
    for rule in _ENTITY_RULES:
        matched_aliases = _phrase_matches(
            exact_quote,
            rule["aliases"],
        )
        if not matched_aliases:
            continue
        original = matched_aliases[0]
        replacement_alias_matches = _phrase_matches(
            frozen_source,
            rule["replacement_aliases"],
        )
        if replacement_alias_matches:
            raise MutationIneligible(
                "entity_swap",
                "replacement_alias_in_source",
                {
                    "recognized_entity": rule["canonical"],
                    "replacement": rule["replacement"],
                    "replacement_aliases": list(
                        rule["replacement_aliases"]
                    ),
                    "replacement_alias_matches": replacement_alias_matches,
                },
            )
        replaced = _replace_token_phrase(
            exact_quote,
            original,
            rule["replacement"],
        )
        if replaced is None:
            raise ValueError("Recognized entity could not be replaced")
        statement, _ = replaced
        variant = _base_variant(unit, "entity_swap")
        variant["statement"] = statement
        variant["mutation_audit"] = {
            "operator": "entity_swap",
            "original": original,
            "recognized_entity": rule["canonical"],
            "recognized_aliases": list(rule["aliases"]),
            "replacement": rule["replacement"],
            "replacement_aliases": list(rule["replacement_aliases"]),
            "replacement_alias_matches_in_source": (
                replacement_alias_matches
            ),
            "original_statement": exact_quote,
            "validation": {
                "token_boundary_match": True,
                "central_entity_in_exact_quote": True,
                "replacement_aliases_absent_from_source": True,
                "statement_changed": statement != unit["statement"],
            },
        }
        variant["variant_id"] = _variant_id(
            unit["source_id"],
            "entity_swap",
            statement,
        )
        return variant
    raise MutationIneligible(
        "entity_swap",
        "no_central_entity",
        {"source_id": unit["source_id"]},
    )


def _choose_cross_topic_donor(
    unit: dict,
    units: list[dict],
    seed: int,
    *,
    profiles: dict[str, dict] | None = None,
) -> tuple[dict, dict]:
    profiles = profiles or {
        str(candidate["source_id"]): _citation_profile(candidate)
        for candidate in units
    }
    source_profile = profiles[str(unit["source_id"])]
    candidates = sorted(
        (
            candidate
            for candidate in units
            if candidate.get("source_id") != unit.get("source_id")
        ),
        key=lambda item: item["source_id"],
    )
    rng = random.Random(
        _seed_for(seed, unit["source_id"], "citation_swap")
    )
    rng.shuffle(candidates)

    reason_counts: dict[str, int] = {}
    diagnostic_samples: list[dict] = []
    sample_counts: dict[str, int] = {}

    def record_rejection(candidate: dict, reason: str) -> None:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if (
            sample_counts.get(reason, 0)
            < MAX_DONOR_DIAGNOSTIC_SAMPLES_PER_REASON
        ):
            diagnostic_samples.append(
                {
                    "donor_source_id": candidate["source_id"],
                    "reason": reason,
                }
            )
            sample_counts[reason] = sample_counts.get(reason, 0) + 1

    for candidate in candidates:
        donor_profile = profiles[str(candidate["source_id"])]
        if not source_profile["anchors"]:
            record_rejection(candidate, "no_essential_anchors")
            continue
        if (
            not source_profile["normalized_topic"]
            or not donor_profile["normalized_topic"]
            or source_profile["normalized_topic"]
            == donor_profile["normalized_topic"]
        ):
            record_rejection(candidate, "same_or_missing_topic")
            continue

        anchor_present = False
        for anchor in source_profile["anchors"]:
            if anchor["kind"] == "numeric":
                anchor_present = bool(
                    _numeric_anchor_matches(
                        anchor,
                        donor_profile["donor_text"],
                    )
                )
            else:
                anchor_present = any(
                    alias in donor_profile["present_fixed_phrases"]
                    for alias in anchor["aliases"]
                )
            if anchor_present:
                break
        if anchor_present:
            record_rejection(
                candidate,
                "essential_anchor_present_in_donor",
            )
            continue

        union = (
            source_profile["statement_tokens"]
            | donor_profile["donor_tokens"]
        )
        overlap = (
            len(
                source_profile["statement_tokens"]
                & donor_profile["donor_tokens"]
            )
            / len(union)
            if union
            else 1.0
        )
        if overlap > MAX_CONTENT_TOKEN_OVERLAP:
            record_rejection(candidate, "high_content_token_overlap")
            continue

        assessment = _assess_citation_donor(
            unit,
            candidate,
            source_profile=source_profile,
            donor_profile=donor_profile,
        )
        if not assessment["eligible"]:
            record_rejection(
                candidate,
                assessment["reason"] or "failed_full_assessment",
            )
            continue
        return candidate, assessment

    raise MutationIneligible(
        "citation_swap",
        "no_safe_donor",
        {
            "source_id": unit["source_id"],
            "candidate_count": len(candidates),
            "reason_counts": dict(sorted(reason_counts.items())),
            "diagnostic_samples": diagnostic_samples,
            "diagnostic_sample_limit_per_reason": (
                MAX_DONOR_DIAGNOSTIC_SAMPLES_PER_REASON
            ),
        },
    )


def _validate_unit(unit: dict) -> None:
    required = {
        "source_id",
        "hypothesis_id",
        "evidence_id",
        "title",
        "abstract",
        "statement",
    }
    missing = sorted(required.difference(unit))
    if missing:
        raise ValueError(f"Source unit missing fields: {missing}")
    if (
        not _normalize_for_matching(unit["statement"])
        or _normalize_for_matching(unit["statement"])
        not in _normalize_for_matching(unit["abstract"])
    ):
        raise ValueError(
            f"Source statement is not in abstract: {unit['source_id']}"
        )


def _validate_variant_manifest(
    variants: list[dict],
    source_ids: set[str],
) -> None:
    allowed_types = {"clean_quote", *MUTATIONS}
    variant_ids: set[str] = set()
    by_source: dict[str, list[dict]] = {}
    for variant in variants:
        variant_id = variant["variant_id"]
        if variant_id in variant_ids:
            raise ValueError(f"Duplicate variant ID: {variant_id}")
        variant_ids.add(variant_id)
        by_source.setdefault(variant["source_id"], []).append(variant)

    if set(by_source) != source_ids:
        raise ValueError("Variant manifest source IDs do not match input")
    for source_id, source_variants in by_source.items():
        variant_types = [
            variant["variant_type"] for variant in source_variants
        ]
        if not set(variant_types).issubset(allowed_types):
            raise ValueError(
                f"Unknown variant type for source {source_id}"
            )
        if len(variant_types) != len(set(variant_types)):
            raise ValueError(
                f"Duplicate variant type for source {source_id}"
            )
        if not {"clean_quote", "invalid_id"}.issubset(variant_types):
            raise ValueError(
                f"Missing unconditional baseline for source {source_id}"
            )
        if sum(
            variant["gold"] == "supported"
            for variant in source_variants
        ) != 1:
            raise ValueError(
                f"Invalid gold labels for source {source_id}"
            )
        if any(
            not variant.get("mutation_audit")
            for variant in source_variants
            if variant["variant_type"] != "clean_quote"
        ):
            raise ValueError(
                f"Missing mutation audit for source {source_id}"
            )


def build_variants(
    units: list[dict],
    seed: int,
    *,
    return_report: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """Build validated variants and optionally return explicit attrition."""
    ordered_units = sorted(
        (copy.deepcopy(unit) for unit in units),
        key=lambda item: item.get("source_id", ""),
    )
    source_ids = [str(unit.get("source_id", "")) for unit in ordered_units]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Source IDs must be unique")
    for unit in ordered_units:
        _validate_unit(unit)

    frozen_ids = {
        _normalize_evidence_id(unit["evidence_id"])
        for unit in ordered_units
    }
    profiles = {
        str(unit["source_id"]): _citation_profile(unit)
        for unit in ordered_units
    }
    output: list[dict] = []
    attrition_records: list[dict] = []
    complete_source_count = 0
    for unit in ordered_units:
        failures: list[MutationIneligible] = []
        output.extend(
            (
                _clean_variant(unit),
                _invalid_id_variant(unit, frozen_ids),
            )
        )
        try:
            donor, donor_assessment = _choose_cross_topic_donor(
                unit,
                ordered_units,
                seed,
                profiles=profiles,
            )
            output.append(
                _citation_swap_variant(
                    unit,
                    donor,
                    donor_assessment,
                )
            )
        except MutationIneligible as exc:
            failures.append(exc)
        try:
            output.append(_causal_overclaim_variant(unit))
        except MutationIneligible as exc:
            failures.append(exc)
        try:
            output.append(_entity_swap_variant(unit))
        except MutationIneligible as exc:
            failures.append(exc)

        if failures:
            attrition_records.extend(
                {
                    "source_id": unit["source_id"],
                    "operator": failure.operator,
                    "reason": failure.reason,
                    "details": copy.deepcopy(failure.details),
                }
                for failure in failures
            )
        else:
            complete_source_count += 1

    all_source_ids = set(source_ids)
    _validate_variant_manifest(output, all_source_ids)
    reason_counts: dict[str, int] = {}
    for record in attrition_records:
        key = f"{record['operator']}:{record['reason']}"
        reason_counts[key] = reason_counts.get(key, 0) + 1
    emitted_counts = {
        variant_type: sum(
            item["variant_type"] == variant_type
            for item in output
        )
        for variant_type in ("clean_quote", *MUTATIONS)
    }
    report = {
        "input_source_count": len(ordered_units),
        "emitted_source_count": len(all_source_ids),
        "ineligible_source_count": 0,
        "complete_source_count": complete_source_count,
        "partial_source_count": (
            len(ordered_units) - complete_source_count
        ),
        "emitted_counts": emitted_counts,
        "reason_counts": dict(sorted(reason_counts.items())),
        "records": attrition_records,
    }
    if return_report:
        return output, report
    return output


def validate_variant_yields(
    variants: list[dict],
    *,
    minimum_per_type: int,
) -> dict:
    """Report pair-eligible yield and block only deficient corruptions."""
    if minimum_per_type < 0:
        raise ValueError("minimum_per_type must be non-negative")
    clean_sources = {
        str(item.get("source_id", ""))
        for item in variants
        if item.get("variant_type") == "clean_quote"
    }
    counts = {
        variant_type: len(
            {
                str(item.get("source_id", ""))
                for item in variants
                if (
                    item.get("variant_type") == variant_type
                    and str(item.get("source_id", "")) in clean_sources
                )
            }
        )
        for variant_type in MUTATIONS
    }
    shortfall = {
        variant_type: max(0, minimum_per_type - count)
        for variant_type, count in counts.items()
    }
    blocked_types = sorted(
        variant_type
        for variant_type, missing in shortfall.items()
        if missing
    )
    ready_types = sorted(
        variant_type
        for variant_type, missing in shortfall.items()
        if not missing
    )
    return {
        "minimum_per_type": minimum_per_type,
        "counts": counts,
        "shortfall": shortfall,
        "blocked_types": blocked_types,
        "ready_types": ready_types,
        "all_ready": not blocked_types,
        "api_calls_blocked_for_types": blocked_types,
    }


def sample_paired_variants(
    variants: list[dict],
    *,
    pairs_per_type: int,
    seed: int,
) -> list[dict]:
    """Sample deterministic corruption-clean pairs without pseudoreplication."""
    yield_report = validate_variant_yields(
        variants,
        minimum_per_type=pairs_per_type,
    )
    if not yield_report["all_ready"]:
        raise ValueError(
            "Insufficient paired yield for: "
            + ", ".join(yield_report["blocked_types"])
        )
    clean_by_source = {
        str(item["source_id"]): item
        for item in variants
        if item.get("variant_type") == "clean_quote"
    }
    output: list[dict] = []
    for variant_type in MUTATIONS:
        candidates = sorted(
            (
                item
                for item in variants
                if (
                    item.get("variant_type") == variant_type
                    and str(item.get("source_id", "")) in clean_by_source
                )
            ),
            key=lambda item: (
                str(item.get("source_id", "")),
                str(item.get("variant_id", "")),
            ),
        )
        random.Random(
            _seed_for(seed, variant_type, "paired_sample")
        ).shuffle(candidates)
        for corrupted in candidates[:pairs_per_type]:
            source_id = str(corrupted["source_id"])
            pair_id = _sha256_text(
                f"{seed}|{variant_type}|{source_id}"
            )
            clean = copy.deepcopy(clean_by_source[source_id])
            corruption = copy.deepcopy(corrupted)
            clean.update(
                {
                    "pair_id": pair_id,
                    "pair_role": "clean_control",
                    "paired_corruption_type": variant_type,
                }
            )
            corruption.update(
                {
                    "pair_id": pair_id,
                    "pair_role": "corruption",
                    "paired_corruption_type": variant_type,
                }
            )
            output.extend((clean, corruption))
    return output


def build_split_variants(
    units: list[dict],
    split: dict,
    *,
    seed: int,
) -> dict:
    """Generate variants independently after freezing disjoint source IDs."""
    pilot_ids = list(split.get("pilot_ids", []))
    confirmation_ids = list(split.get("confirmation_ids", []))
    if len(pilot_ids) != len(set(pilot_ids)):
        raise ValueError("Pilot source IDs must be unique")
    if len(confirmation_ids) != len(set(confirmation_ids)):
        raise ValueError("Confirmation source IDs must be unique")
    if not set(pilot_ids).isdisjoint(confirmation_ids):
        raise ValueError("Pilot and confirmation source IDs overlap")
    units_by_id = {str(unit.get("source_id", "")): unit for unit in units}
    requested_ids = set(pilot_ids) | set(confirmation_ids)
    missing = sorted(requested_ids.difference(units_by_id))
    if missing:
        raise ValueError(f"Split contains unknown source IDs: {missing}")

    output = {}
    for split_name, source_ids, minimum in (
        ("pilot", pilot_ids, PILOT_PAIRS_PER_CORRUPTION),
        (
            "confirmation",
            confirmation_ids,
            CONFIRMATION_PAIRS_PER_CORRUPTION,
        ),
    ):
        split_units = [units_by_id[source_id] for source_id in source_ids]
        split_variants, report = build_variants(
            split_units,
            seed,
            return_report=True,
        )
        output[split_name] = {
            "source_ids": source_ids,
            "variants": split_variants,
            "attrition_report": report,
            "yield_report": validate_variant_yields(
                split_variants,
                minimum_per_type=minimum,
            ),
        }
    return output


def _constructibility(
    units: list[dict],
    seed: int,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    ordered = sorted(units, key=lambda item: item["source_id"])
    profiles = {
        str(unit["source_id"]): _citation_profile(unit)
        for unit in ordered
    }
    eligible: dict[str, set[str]] = {
        str(unit["source_id"]): {"invalid_id"}
        for unit in ordered
    }
    citation_donors: dict[str, str] = {}
    for unit in ordered:
        source_id = str(unit["source_id"])
        try:
            donor, _ = _choose_cross_topic_donor(
                unit,
                ordered,
                seed,
                profiles=profiles,
            )
            eligible[source_id].add("citation_swap")
            citation_donors[source_id] = str(donor["source_id"])
        except MutationIneligible:
            pass
        try:
            _causal_overclaim_variant(unit)
            eligible[source_id].add("causal_overclaim")
        except MutationIneligible:
            pass
        try:
            _entity_swap_variant(unit)
            eligible[source_id].add("entity_swap")
        except MutationIneligible:
            pass
    return eligible, citation_donors


def _coverage_counts(
    selected: set[str],
    eligibility: dict[str, set[str]],
    citation_donors: dict[str, str],
) -> dict[str, int]:
    counts = {
        variant_type: sum(
            variant_type in eligibility[source_id]
            for source_id in selected
        )
        for variant_type in MUTATIONS
        if variant_type != "citation_swap"
    }
    counts["citation_swap"] = sum(
        "citation_swap" in eligibility[source_id]
        and citation_donors.get(source_id) in selected
        for source_id in selected
    )
    return counts


def _select_coverage_group(
    units_by_id: dict[str, dict],
    available_ids: set[str],
    *,
    size: int,
    required_per_type: int,
    seed: int,
    split_name: str,
    reserve_eligibility: dict[str, set[str]] | None = None,
    reserve_citation_donors: dict[str, str] | None = None,
    reserve_required_per_type: int = 0,
) -> tuple[list[str], dict]:
    available_units = [
        units_by_id[source_id]
        for source_id in sorted(available_ids)
    ]
    eligibility, citation_donors = _constructibility(
        available_units,
        seed,
    )
    ordered_ids = sorted(available_ids)
    random.Random(
        _seed_for(seed, split_name, "coverage_selection")
    ).shuffle(ordered_ids)
    selected: set[str] = set()
    strata: list[dict] = []

    def preserves_reserve(package: set[str]) -> bool:
        if not reserve_eligibility or not reserve_required_per_type:
            return True
        proposed = selected.union(package)
        remaining = available_ids.difference(proposed)
        reserve_counts = _coverage_counts(
            remaining,
            reserve_eligibility,
            reserve_citation_donors or {},
        )
        return all(
            reserve_counts[variant_type] >= reserve_required_per_type
            for variant_type in MUTATIONS
        )

    citation_candidates = [
        source_id
        for source_id in ordered_ids
        if "citation_swap" in eligibility[source_id]
    ]
    for source_id in citation_candidates:
        if (
            _coverage_counts(
                selected,
                eligibility,
                citation_donors,
            )["citation_swap"]
            >= required_per_type
        ):
            break
        donor_id = citation_donors[source_id]
        package = {source_id, donor_id}.difference(selected)
        if not package.issubset(available_ids):
            continue
        if len(selected) + len(package) > size:
            continue
        if not preserves_reserve(package):
            continue
        selected.update(package)
        strata.append(
            {
                "stratum": "citation_swap_core",
                "source_id": source_id,
                "donor_source_id": donor_id,
                "added_ids": sorted(package),
            }
        )

    semantic_types = ("causal_overclaim", "entity_swap")
    while len(selected) < size:
        counts = _coverage_counts(
            selected,
            eligibility,
            citation_donors,
        )
        unmet = {
            variant_type
            for variant_type in semantic_types
            if counts[variant_type] < required_per_type
        }
        if not unmet:
            break
        candidates = [
            source_id
            for source_id in ordered_ids
            if (
                source_id not in selected
                and eligibility[source_id].intersection(unmet)
                and preserves_reserve({source_id})
            )
        ]
        if not candidates:
            break
        candidates.sort(
            key=lambda source_id: (
                -len(eligibility[source_id].intersection(unmet)),
                ordered_ids.index(source_id),
            )
        )
        chosen = candidates[0]
        selected.add(chosen)
        strata.append(
            {
                "stratum": "semantic_core",
                "source_id": chosen,
                "constructible_types": sorted(
                    eligibility[chosen].intersection(unmet)
                ),
                "added_ids": [chosen],
            }
        )

    fill_ids = []
    for source_id in ordered_ids:
        if len(selected) >= size:
            break
        if source_id in selected:
            continue
        if preserves_reserve({source_id}):
            selected.add(source_id)
            fill_ids.append(source_id)
    if len(selected) < size:
        for source_id in ordered_ids:
            if len(selected) >= size:
                break
            if source_id not in selected:
                selected.add(source_id)
                fill_ids.append(source_id)
    if fill_ids:
        strata.append(
            {
                "stratum": "deterministic_fill",
                "added_ids": fill_ids,
            }
        )
    selected_ids = [
        source_id for source_id in ordered_ids if source_id in selected
    ]
    dependency_counts = _coverage_counts(
        selected,
        eligibility,
        citation_donors,
    )
    _, actual_report = build_variants(
        [units_by_id[source_id] for source_id in selected_ids],
        seed,
        return_report=True,
    )
    counts = {
        variant_type: actual_report["emitted_counts"].get(
            variant_type,
            0,
        )
        for variant_type in MUTATIONS
    }
    shortfall = {
        variant_type: max(0, required_per_type - counts[variant_type])
        for variant_type in MUTATIONS
    }
    return selected_ids, {
        "required_per_type": required_per_type,
        "constructible_counts": counts,
        "directed_dependency_counts": dependency_counts,
        "post_selection_yield_verified": True,
        "shortfall": shortfall,
        "strata": strata,
    }


def select_coverage_aware_split(
    units: list[dict],
    pilot_n: int,
    confirmation_n: int,
    seed: int,
) -> dict:
    """Select disjoint splits using constructibility, never model outcomes."""
    ordered_units = sorted(
        (copy.deepcopy(unit) for unit in units),
        key=lambda item: item.get("source_id", ""),
    )
    for unit in ordered_units:
        _validate_unit(unit)
    source_ids = [str(unit["source_id"]) for unit in ordered_units]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Source IDs must be unique")
    meaningful_minimum = min(pilot_n, MIN_MEANINGFUL_PILOT)
    if len(source_ids) < meaningful_minimum:
        raise ValueError(
            f"Fewer than {meaningful_minimum} sources cannot form a "
            "meaningful pilot"
        )
    actual_pilot_n = min(pilot_n, len(source_ids))
    actual_confirmation_n = min(
        confirmation_n,
        len(source_ids) - actual_pilot_n,
    )

    units_by_id = {
        str(unit["source_id"]): unit for unit in ordered_units
    }
    all_ids = set(source_ids)
    pool_eligibility, pool_citation_donors = _constructibility(
        ordered_units,
        seed,
    )
    pilot_ids, pilot_selection = _select_coverage_group(
        units_by_id,
        all_ids,
        size=actual_pilot_n,
        required_per_type=PILOT_PAIRS_PER_CORRUPTION,
        seed=seed,
        split_name="pilot",
        reserve_eligibility=pool_eligibility,
        reserve_citation_donors=pool_citation_donors,
        reserve_required_per_type=(
            CONFIRMATION_PAIRS_PER_CORRUPTION
            if actual_confirmation_n
            else 0
        ),
    )
    remaining_ids = all_ids.difference(pilot_ids)
    confirmation_ids, confirmation_selection = _select_coverage_group(
        units_by_id,
        remaining_ids,
        size=actual_confirmation_n,
        required_per_type=CONFIRMATION_PAIRS_PER_CORRUPTION,
        seed=seed,
        split_name="confirmation",
    )
    if not set(pilot_ids).isdisjoint(confirmation_ids):
        raise ValueError("Pilot and confirmation source IDs overlap")
    combined_shortfall = {
        variant_type: (
            pilot_selection["shortfall"][variant_type]
            + confirmation_selection["shortfall"][variant_type]
        )
        for variant_type in MUTATIONS
    }
    return {
        "pilot_ids": pilot_ids,
        "confirmation_ids": confirmation_ids,
        "selection_method": "constructibility",
        "selection_strata": {
            "pilot": pilot_selection,
            "confirmation": confirmation_selection,
        },
        "coverage_shortfall": combined_shortfall,
        "blocked_types": sorted(
            variant_type
            for variant_type, missing in combined_shortfall.items()
            if missing
        ),
        "shortfall": {
            "requested_total": pilot_n + confirmation_n,
            "available_total": len(source_ids),
            "total_shortfall": max(
                0,
                pilot_n + confirmation_n - len(source_ids),
            ),
            "pilot_shortfall": pilot_n - len(pilot_ids),
            "confirmation_shortfall": (
                confirmation_n - len(confirmation_ids)
            ),
            "used_all_available": (
                len(source_ids) < pilot_n + confirmation_n
            ),
        },
    }


def split_source_units(
    units: list[dict],
    pilot_n: int,
    confirmation_n: int,
    seed: int,
) -> dict:
    """Return deterministic source-disjoint pilot and confirmation IDs."""
    if pilot_n < 0 or confirmation_n < 0:
        raise ValueError("Split sizes must be non-negative")
    source_ids = [str(unit.get("source_id", "")) for unit in units]
    if any(not source_id for source_id in source_ids):
        raise ValueError("Every source unit must have a source_id")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Source IDs must be unique")
    requested_total = pilot_n + confirmation_n
    meaningful_minimum = min(pilot_n, MIN_MEANINGFUL_PILOT)
    if len(source_ids) < meaningful_minimum:
        raise ValueError(
            f"Fewer than {meaningful_minimum} sources cannot form a "
            "meaningful pilot"
        )

    shuffled = sorted(source_ids)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < requested_total:
        actual_pilot_n = min(pilot_n, len(shuffled))
        actual_confirmation_n = len(shuffled) - actual_pilot_n
    else:
        actual_pilot_n = pilot_n
        actual_confirmation_n = confirmation_n
    pilot_ids = shuffled[:actual_pilot_n]
    confirmation_ids = shuffled[
        actual_pilot_n : actual_pilot_n + actual_confirmation_n
    ]
    if not set(pilot_ids).isdisjoint(confirmation_ids):
        raise ValueError("Pilot and confirmation source IDs overlap")
    return {
        "pilot_ids": pilot_ids,
        "confirmation_ids": confirmation_ids,
        "shortfall": {
            "requested_total": requested_total,
            "available_total": len(source_ids),
            "total_shortfall": max(0, requested_total - len(source_ids)),
            "pilot_shortfall": pilot_n - len(pilot_ids),
            "confirmation_shortfall": (
                confirmation_n - len(confirmation_ids)
            ),
            "used_all_available": len(source_ids) < requested_total,
        },
    }


def normalize_binary_verdict(value: str) -> str:
    """Map every value except an exact supported verdict fail-closed."""
    return (
        "supported"
        if isinstance(value, str)
        and value.strip().casefold() == "supported"
        else "unsupported"
    )


def _prediction(
    item: dict,
    verdict: Any,
    *,
    schema_failure: bool,
    reason: str = "",
    quote: str = "",
) -> dict:
    result = {
        "variant_id": str(item["variant_id"]),
        "source_id": str(item["source_id"]),
        "prediction": normalize_binary_verdict(verdict),
        "schema_failure": bool(schema_failure),
    }
    if reason:
        result["reason"] = str(reason)
    if quote:
        result["quote"] = str(quote)
    return result


def run_id_only(variants: list[dict]) -> list[dict]:
    """Apply the deterministic identifier-existence ablation."""
    predictions = []
    for item in sorted(
        variants,
        key=lambda value: str(value.get("variant_id", "")),
    ):
        evidence_id = _normalize_evidence_id(item.get("evidence_id", ""))
        identifier_valid = bool(re.fullmatch(r"PMID:\d+", evidence_id))
        exists = (
            item.get("exists") is not False
            and evidence_id != RESERVED_BENCHMARK_PMID
            and identifier_valid
        )
        predictions.append(
            _prediction(
                item,
                "supported" if exists else "unsupported",
                schema_failure=False,
            )
        )
    return predictions


def _predictions_from_grounded(
    batch: list[dict],
    grounded: Any,
) -> list[dict]:
    by_id = {
        str(item.get("claim_id", "")): item
        for item in grounded
        if isinstance(item, dict)
    } if isinstance(grounded, list) else {}
    predictions = []
    for variant in batch:
        record = by_id.get(str(variant["variant_id"]))
        verdict = record.get("verdict") if isinstance(record, dict) else None
        normalized = (
            verdict.strip().casefold()
            if isinstance(verdict, str)
            else ""
        )
        predictions.append(
            _prediction(
                variant,
                verdict,
                schema_failure=normalized not in _GROUNDING_VERDICTS,
            )
        )
    return predictions


async def _judge_full_with_raw(
    verifier: object,
    batch: list[dict],
    target: str,
) -> tuple[list[dict], Any]:
    claims, metadata = _full_inputs(batch)
    grounded = await verifier.ground_existing_claims(
        claims,
        metadata,
        target,
    )
    return _predictions_from_grounded(batch, grounded), grounded


async def judge_full(
    verifier: object,
    batch: list[dict],
    target: str = "",
) -> list[dict]:
    """Judge frozen atomic claims through Claim Grounding only."""
    predictions, _ = await _judge_full_with_raw(verifier, batch, target)
    return predictions


def _holistic_payload(batch: list[dict]) -> list[dict]:
    return [
        {
            "variant_id": str(item["variant_id"]),
            "statement": str(item["statement"]),
            "evidence_id": str(item["evidence_id"]),
            "title": str(item.get("title", "")),
            "abstract": str(item.get("abstract", "")),
        }
        for item in batch
    ]


def _holistic_messages(
    batch: list[dict],
    *,
    repair: bool = False,
) -> list[dict]:
    prefix = (
        "Repair the omitted or malformed results below. "
        if repair
        else ""
    )
    prompt = (
        f"{prefix}Judge each item independently using only its supplied "
        "source. Return exactly one binary verdict for every variant_id, "
        "with a reason and optional exact quote.\n\n"
        f"Items:\n{json.dumps(_holistic_payload(batch), ensure_ascii=False)}"
        "\n\nReturn JSON:\n"
        '{"items": [{"variant_id": "...", "verdict": "supported or '
        'unsupported", "reason": "...", "quote": "..."}]}'
    )
    return [
        {"role": "system", "content": HOLISTIC_SYSTEM},
        {"role": "user", "content": prompt},
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_value(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        return _redact_text(value, secrets)
    if isinstance(value, dict):
        return {
            str(key): _safe_value(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, secrets) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_text(str(value), secrets)


def _safe_error(error: BaseException, secrets: tuple[str, ...]) -> dict:
    if isinstance(error, ProviderHTTPStatusError):
        return {
            "type": type(error).__name__,
            "status_code": error.status_code,
            "response_body": _redact_text(error.response_body, secrets),
            "retry_after": error.retry_after,
            "request_id": error.request_id,
            "safe_headers": _safe_value(error.safe_headers, secrets),
        }
    return {
        "type": type(error).__name__,
        "message": _redact_text(str(error), secrets)[:4096],
    }

_TRANSIENT_HTTP_STATUS = {429, 500, 502, 503, 504}
_PROMPT_VERSION = "evidence-integrity-task2-v2"
_FULL_RAW_VERDICTS = {
    "supported",
    "partial",
    "unsupported",
    "contradicted",
}
_RETRY_BACKOFF_BASE_SECONDS = 2.0
MAX_RETRY_BACKOFF_SECONDS = 300.0
TRANSIENT_TRANSPORT_RETRY_POLICY_VERSION = "benchmark-httpx-transient-v1"
TERMINAL_FAILURE_LATCH_POLICY_VERSION = "benchmark-terminal-latch-v1"
TRANSIENT_TRANSPORT_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.CloseError,
)
TRANSIENT_TRANSPORT_EXCEPTION_NAMES = tuple(
    exception_type.__name__
    for exception_type in TRANSIENT_TRANSPORT_EXCEPTIONS
)
_QUOTED_CREDENTIAL = re.compile(
    r'(?i)(["\']?(?:accesskey|api[_-]?key|authorization)["\']?\s*[:=]\s*)'
    r'(["\']?)(?:bearer\s+)?[^"\'\s,;}]+(["\']?)'
)
_BEARER_CREDENTIAL = re.compile(
    r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"
)


def _redact_text(value: str, secrets: tuple[str, ...]) -> str:
    redacted = str(value)
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    redacted = _QUOTED_CREDENTIAL.sub(
        lambda match: (
            f"{match.group(1)}{match.group(2)}"
            f"[REDACTED]{match.group(3)}"
        ),
        redacted,
    )
    return _BEARER_CREDENTIAL.sub("Bearer [REDACTED]", redacted)


def _full_inputs(batch: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    claims = []
    metadata: dict[str, dict] = {}
    observed: dict[str, dict] = {}
    for item in batch:
        evidence_id = str(item["evidence_id"])
        frozen = {
            "title": str(item.get("title", "")),
            "abstract": str(item.get("abstract", "")),
            "exists": item.get("exists"),
        }
        if evidence_id in observed and observed[evidence_id] != frozen:
            raise ValueError(
                f"Shared evidence ID has conflicting metadata: {evidence_id}"
            )
        observed[evidence_id] = frozen
        if frozen["abstract"]:
            metadata[evidence_id] = {
                "title": frozen["title"],
                "abstract": frozen["abstract"],
            }
        claims.append(
            {
                "claim_id": str(item["variant_id"]),
                "hypothesis_id": str(item["source_id"]),
                "text": str(item["statement"]),
                "expected_relation": "support",
                "evidence_ids": [evidence_id],
            }
        )
    return claims, metadata


def _parse_holistic(
    batch: list[dict],
    raw: Any,
) -> tuple[list[dict], list[dict]]:
    expected = {str(item["variant_id"]): item for item in batch}
    raw_items = raw.get("items", []) if isinstance(raw, dict) else []
    returned: dict[str, dict] = {}
    duplicate_ids: set[str] = set()
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            variant_id = str(item.get("variant_id", ""))
            if variant_id in returned:
                duplicate_ids.add(variant_id)
            elif variant_id in expected:
                returned[variant_id] = item
    predictions = []
    repair_batch = []
    for variant_id, variant in expected.items():
        item = returned.get(variant_id)
        verdict = item.get("verdict") if isinstance(item, dict) else None
        reason = item.get("reason") if isinstance(item, dict) else None
        valid = (
            variant_id not in duplicate_ids
            and isinstance(verdict, str)
            and verdict.strip().casefold() in {"supported", "unsupported"}
            and isinstance(reason, str)
            and bool(reason.strip())
        )
        predictions.append(
            _prediction(
                variant,
                verdict,
                schema_failure=not valid,
                reason=reason if isinstance(reason, str) else "",
                quote=item.get("quote", "") if isinstance(item, dict) else "",
            )
        )
        if not valid:
            repair_batch.append(variant)
    return predictions, repair_batch


def _strip_json_fences(content: str) -> str:
    text = str(content).strip()
    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    if text.casefold().startswith("json"):
        return text[4:].strip()
    return text


def _response_was_truncated(raw_response: Any) -> bool:
    if not isinstance(raw_response, dict):
        return False
    choices = raw_response.get("choices")
    if not isinstance(choices, list):
        return False
    return any(
        isinstance(choice, dict)
        and str(choice.get("finish_reason", "")).casefold() == "length"
        for choice in choices
    )


def _dependency_hashes() -> dict[str, str]:
    benchmark_path = Path(__file__)
    claim_verifier_path = Path(inspect.getsourcefile(ClaimVerifier) or "")
    llm_path = Path(inspect.getsourcefile(ProviderHTTPStatusError) or "")
    if not claim_verifier_path.is_file() or not llm_path.is_file():
        raise RuntimeError("Cannot fingerprint production dependencies")
    return {
        "benchmark_module_sha256": _sha256_bytes(
            benchmark_path.read_bytes()
        ),
        "claim_verifier_sha256": _sha256_bytes(
            claim_verifier_path.read_bytes()
        ),
        "llm_sha256": _sha256_bytes(llm_path.read_bytes()),
    }


def _retry_after_seconds(
    value: str | None,
    *,
    now: datetime | None = None,
) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return max(0.0, (parsed - current).total_seconds())


def _claim_ids_from_messages(messages: list[dict]) -> list[str]:
    if not messages:
        return []
    prompt = str(messages[-1].get("content", ""))
    if "Claims:\n" not in prompt or "\n\nAllowed verdicts" not in prompt:
        return []
    try:
        payload = json.loads(
            prompt.split("Claims:\n", 1)[1].split(
                "\n\nAllowed verdicts",
                1,
            )[0]
        )
    except (json.JSONDecodeError, TypeError):
        return []
    return [
        str(item.get("claim_id", ""))
        for item in payload
        if isinstance(item, dict) and item.get("claim_id")
    ]


def _validate_full_response_ids(
    result: Any,
    expected_ids: list[str],
) -> tuple[Any, dict]:
    if not expected_ids or not isinstance(result, dict):
        return result, {}
    items = result.get("items", [])
    if not isinstance(items, list):
        return {"items": []}, {
            "missing_ids": expected_ids,
            "duplicate_ids": [],
            "malformed_items": True,
        }
    counts: dict[str, int] = {}
    invalid_items: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            invalid_items.append({"reason": "item_not_object"})
            continue
        claim_id = item.get("claim_id")
        verdict = item.get("verdict")
        reason = item.get("reason")
        quote = item.get("quote")
        if isinstance(claim_id, str) and claim_id in expected_ids:
            counts[claim_id] = counts.get(claim_id, 0) + 1
        field_errors = []
        if not isinstance(claim_id, str) or claim_id not in expected_ids:
            field_errors.append("invalid_claim_id")
        if (
            not isinstance(verdict, str)
            or verdict.strip().casefold() not in _FULL_RAW_VERDICTS
        ):
            field_errors.append("invalid_verdict")
        if not isinstance(reason, str) or not reason.strip():
            field_errors.append("invalid_reason")
        if not isinstance(quote, str):
            field_errors.append("invalid_quote")
        if field_errors:
            invalid_items.append(
                {
                    "claim_id": claim_id
                    if isinstance(claim_id, str)
                    else None,
                    "errors": field_errors,
                }
            )
            continue
    duplicates = sorted(
        claim_id for claim_id, count in counts.items() if count != 1
    )
    valid_items = [
        item
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("claim_id"), str)
        and item.get("claim_id") in counts
        and item.get("claim_id") not in duplicates
        and isinstance(item.get("verdict"), str)
        and item.get("verdict").strip().casefold() in _FULL_RAW_VERDICTS
        and isinstance(item.get("reason"), str)
        and bool(item.get("reason").strip())
        and isinstance(item.get("quote"), str)
    ]
    present = {str(item["claim_id"]) for item in valid_items}
    missing = sorted(set(expected_ids).difference(present))
    issues = {
        "missing_ids": missing,
        "duplicate_ids": duplicates,
        "invalid_items": invalid_items,
    }
    if missing or duplicates or invalid_items:
        sanitized = copy.deepcopy(result)
        sanitized["items"] = valid_items
        return sanitized, issues
    return result, {}


class RecordingStructuredClient:
    """Benchmark-only structured client with explicit request accounting."""

    def __init__(
        self,
        underlying: object,
        *,
        record_request: object,
        max_request_retries: int,
        retry_sleep: object,
        secrets: tuple[str, ...],
        retry_backoff_cap: float,
    ) -> None:
        if not hasattr(underlying, "chat_raw"):
            raise TypeError("Benchmark LLM must implement chat_raw")
        self.underlying = underlying
        self.record_request = record_request
        self.max_request_retries = max(0, int(max_request_retries))
        self.retry_sleep = retry_sleep
        self.secrets = secrets
        self.retry_backoff_cap = min(
            MAX_RETRY_BACKOFF_SECONDS,
            max(0.0, float(retry_backoff_cap)),
        )
        self.api_key = getattr(underlying, "api_key", "")
        self.model = getattr(underlying, "model", "")
        self.final_transport_failures = 0
        self._terminal_error: BaseException | None = None

    @property
    def terminal_error(self) -> BaseException | None:
        return self._terminal_error

    def _latch_terminal(self, error: BaseException) -> None:
        if self._terminal_error is None:
            self._terminal_error = error

    def _raise_if_terminal(self) -> None:
        if self._terminal_error is not None:
            raise BenchmarkTerminalFailure(
                "Benchmark request blocked by terminal failure latch"
            ) from self._terminal_error

    async def structured(
        self,
        messages: list[dict],
        schema: dict | None = None,
        max_tokens: int = 4096,
        max_retries: int = 0,
        task: str | None = None,
        temperature: float = 0,
        **_: Any,
    ) -> tuple[dict, dict]:
        self._raise_if_terminal()
        del max_retries
        request_messages = _safe_value(
            copy.deepcopy(messages),
            self.secrets,
        )
        del schema
        expected_ids = _claim_ids_from_messages(messages)
        for orchestration_attempt in range(self.max_request_retries + 1):
            started_at = _utc_now()
            self._raise_if_terminal()
            try:
                response = await self.underlying.chat_raw(
                    request_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    max_retries=0,
                    task=task,
                )
                response_truncated = _response_was_truncated(
                    response.raw_response
                )
                requested_model = _normalized_provider_model(self.model)
                provider_model = _normalized_provider_model(
                    response.provider_model
                )
                if (
                    not requested_model
                    or not provider_model
                    or provider_model != requested_model
                ):
                    mismatch = ProviderModelMismatch(
                        "provider_model missing or does not match requested model"
                    )
                    self.record_request(
                        messages=request_messages,
                        raw_response={
                            "content": response.content,
                            "provider_response": response.raw_response,
                        },
                        usage=response.usage,
                        provider_model=response.provider_model,
                        started_at=started_at,
                        error=mismatch,
                        request_succeeded=False,
                        schema_issues={
                            "provider_model_mismatch": True,
                            **(
                                {"response_truncated": True}
                                if response_truncated
                                else {}
                            ),
                        },
                        response_truncated=response_truncated,
                    )
                    self._latch_terminal(mismatch)
                    self.final_transport_failures += 1
                    raise mismatch
                try:
                    parsed = json.loads(_strip_json_fences(response.content))
                    parsed, schema_issues = _validate_full_response_ids(
                        parsed,
                        expected_ids,
                    )
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    malformed = MalformedResponseError(str(error))
                    self.record_request(
                        messages=request_messages,
                        raw_response={
                            "content": response.content,
                            "provider_response": response.raw_response,
                        },
                        usage=response.usage,
                        provider_model=response.provider_model,
                        started_at=started_at,
                        error=malformed,
                        request_succeeded=True,
                        schema_issues={
                            "malformed_json": True,
                            **(
                                {"response_truncated": True}
                                if response_truncated
                                else {}
                            ),
                        },
                        response_truncated=response_truncated,
                    )
                    return {"error": "json_parse_failed"}, response.usage
                self.record_request(
                    messages=request_messages,
                    raw_response={
                        "content": response.content,
                        "provider_response": response.raw_response,
                    },
                    usage=response.usage,
                    provider_model=response.provider_model,
                    started_at=started_at,
                    error=None,
                    request_succeeded=True,
                    normalized_response=parsed,
                    schema_issues={
                        **schema_issues,
                        **(
                            {"response_truncated": True}
                            if response_truncated
                            else {}
                        ),
                    },
                    response_truncated=response_truncated,
                )
                if response_truncated:
                    return {
                        "error": "response_truncated",
                        "normalized_response": parsed,
                    }, response.usage
                return parsed, response.usage
            except ProviderModelMismatch:
                raise
            except ProviderHTTPStatusError as error:
                transient = error.status_code in _TRANSIENT_HTTP_STATUS
                will_retry = (
                    transient
                    and orchestration_attempt < self.max_request_retries
                )
                wait = None
                if will_retry:
                    exponential = _RETRY_BACKOFF_BASE_SECONDS * (
                        2 ** orchestration_attempt
                    )
                    provider_wait = _retry_after_seconds(error.retry_after)
                    wait = min(
                        self.retry_backoff_cap,
                        max(exponential, provider_wait or 0.0),
                    )
                self.record_request(
                    messages=request_messages,
                    raw_response=None,
                    usage={},
                    provider_model=None,
                    started_at=started_at,
                    error=error,
                    request_succeeded=False,
                    schema_issues={},
                    selected_retry_wait_seconds=wait,
                )
                if not will_retry:
                    self._latch_terminal(error)
                    self.final_transport_failures += 1
                    raise
                await self.retry_sleep(wait)
            except TRANSIENT_TRANSPORT_EXCEPTIONS as error:
                will_retry = (
                    orchestration_attempt < self.max_request_retries
                )
                wait = None
                if will_retry:
                    wait = min(
                        self.retry_backoff_cap,
                        _RETRY_BACKOFF_BASE_SECONDS
                        * (2 ** orchestration_attempt),
                    )
                self.record_request(
                    messages=request_messages,
                    raw_response=None,
                    usage={},
                    provider_model=None,
                    started_at=started_at,
                    error=error,
                    request_succeeded=False,
                    schema_issues={},
                    selected_retry_wait_seconds=wait,
                )
                if not will_retry:
                    self._latch_terminal(error)
                    self.final_transport_failures += 1
                    raise
                await self.retry_sleep(wait)
            except Exception as error:
                self.record_request(
                    messages=request_messages,
                    raw_response=None,
                    usage={},
                    provider_model=None,
                    started_at=started_at,
                    error=error,
                    request_succeeded=False,
                    schema_issues={},
                    selected_retry_wait_seconds=None,
                )
                self._latch_terminal(error)
                self.final_transport_failures += 1
                raise
        raise RuntimeError("unreachable")


class EvidenceIntegrityRunner:
    """Auditable isolated condition runner with strict content resume."""

    def __init__(
        self,
        *,
        variants: dict[str, list[dict]],
        output_dir: Path,
        fingerprint: str = "",
        verifier: object | None = None,
        llm: object | None = None,
        target: str = "",
        batch_size: int = 20,
        holistic_batch_size: int = DEFAULT_HOLISTIC_BATCH_SIZE,
        max_request_retries: int = 1,
        retry_sleep: object = asyncio.sleep,
        retry_backoff_cap: float = 300.0,
        test_mode: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 1 <= holistic_batch_size <= MAX_HOLISTIC_BATCH_SIZE:
            raise ValueError(
                "holistic_batch_size must be between 1 and 5"
            )
        self.variants = copy.deepcopy(variants)
        self.output_dir = Path(output_dir)
        self.external_fingerprint = str(fingerprint)
        self.verifier = verifier
        self.llm = llm
        self.target = str(target)
        self.batch_size = int(batch_size)
        self.holistic_batch_size = int(holistic_batch_size)
        self.max_request_retries = max(0, int(max_request_retries))
        self.retry_sleep = retry_sleep
        self.retry_backoff_cap = min(
            MAX_RETRY_BACKOFF_SECONDS,
            max(0.0, float(retry_backoff_cap)),
        )
        self.test_mode = bool(test_mode)
        self._owner = f"{os.getpid()}-{uuid.uuid4().hex}"
        self._active_fingerprint = ""
        self._active_components: dict = {}
        self._secrets = self._collect_secrets()

    def _collect_secrets(self) -> tuple[str, ...]:
        candidates = [
            os.environ.get("BH_API_KEY", ""),
            getattr(self.llm, "api_key", ""),
            getattr(self.verifier, "api_key", ""),
            getattr(getattr(self.verifier, "llm", None), "api_key", ""),
        ]
        return tuple(
            dict.fromkeys(
                str(value)
                for value in candidates
                if isinstance(value, str) and value
            )
        )

    @staticmethod
    def _identity(value: object | None) -> dict:
        if value is None:
            return {}
        return {
            "class": (
                f"{value.__class__.__module__}."
                f"{value.__class__.__qualname__}"
            ),
            "model": str(getattr(value, "model", "")),
            "base_url": str(getattr(value, "base_url", "")),
            "timeout": getattr(value, "timeout", None),
        }

    def _fingerprint_for(
        self,
        condition: str,
        split: str,
        repetitions: int,
    ) -> tuple[str, dict]:
        selected = sorted(
            self.variants[split],
            key=lambda item: str(item.get("variant_id", "")),
        )
        canonical_variants = json.dumps(
            selected,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        verifier_llm = getattr(self.verifier, "llm", None)
        verifier_config = {
            "class": self._identity(self.verifier).get("class", ""),
            "task": str(getattr(self.verifier, "task", "")),
            "call_timeout_seconds": getattr(
                self.verifier,
                "call_timeout_seconds",
                None,
            ),
            "max_concurrency": getattr(
                self.verifier,
                "max_concurrency",
                None,
            ),
        }
        components = {
            "external_manifest_fingerprint": self.external_fingerprint,
            "variants_sha256": _sha256_text(canonical_variants),
            "target": self.target,
            "condition": condition,
            "split": split,
            "model_provider": {
                "holistic": self._identity(self.llm),
                "verifier": self._identity(verifier_llm),
            },
            "verifier_config": verifier_config,
            "wrapper_config": {
                "max_request_retries": self.max_request_retries,
                "retry_backoff_base_seconds": (
                    _RETRY_BACKOFF_BASE_SECONDS
                ),
                "retry_backoff_cap_seconds": self.retry_backoff_cap,
                "transient_transport_retry_policy_version": (
                    TRANSIENT_TRANSPORT_RETRY_POLICY_VERSION
                ),
                "transient_transport_exception_classes": list(
                    TRANSIENT_TRANSPORT_EXCEPTION_NAMES
                ),
                "terminal_failure_latch_policy_version": (
                    TERMINAL_FAILURE_LATCH_POLICY_VERSION
                ),
            },
            "provider_model_normalization": {
                "rule_version": PROVIDER_MODEL_NORMALIZATION_RULE_VERSION,
                "case_sensitive": True,
                "routing_prefixes": ["bh:", "BohrClaw/"],
                "normalized_requested_models": {
                    "holistic": _normalized_provider_model(
                        getattr(self.llm, "model", "")
                    ),
                    "verifier": _normalized_provider_model(
                        getattr(verifier_llm, "model", "")
                    ),
                },
            },
            "prompt_version": _PROMPT_VERSION,
            "holistic_system_sha256": _sha256_text(HOLISTIC_SYSTEM),
            "holistic_prompt_sha256": _sha256_text(
                json.dumps(
                    _holistic_messages([], repair=False),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            ),
            "full_prompt_sha256": _sha256_text(
                inspect.getsource(ClaimVerifier._ground_claims)
            ),
            "repetitions": repetitions,
            "batch_size": self.batch_size,
            "holistic_batch_size": self.holistic_batch_size,
            "code_hashes": _dependency_hashes(),
            "test_mode": self.test_mode,
        }
        canonical = json.dumps(
            components,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return _sha256_text(canonical), components

    def _safe_json_bytes(self, value: dict) -> bytes:
        safe = _safe_value(value, self._secrets)
        text = json.dumps(
            safe,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        for secret in self._secrets:
            if secret and secret in text:
                raise ValueError("Secret scan failed before artifact write")
        return text.encode("utf-8")

    def _write_json(self, path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{self._owner}.{uuid.uuid4().hex}.tmp"
        )
        try:
            payload = self._safe_json_bytes(value)
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _run_root(self, condition: str, split: str) -> Path:
        return self.output_dir / "attempts" / condition / split

    def _acquire_lock(self, condition: str, split: str) -> Path:
        lock = self._run_root(condition, split) / ".run.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        payload = self._safe_json_bytes(
            {
                "owner": self._owner,
                "pid": os.getpid(),
                "created_at": _utc_now(),
            }
        )
        try:
            descriptor = os.open(
                lock,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as error:
            raise ConcurrentRunError(
                f"Condition/split lock already exists: {lock}"
            ) from error
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return lock

    def _release_lock(self, lock: Path) -> None:
        try:
            record = json.loads(lock.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ConcurrentRunError("Run lock became unreadable") from error
        if record.get("owner") != self._owner:
            raise ConcurrentRunError("Run lock ownership changed")
        lock.unlink()

    def _batch_dir(
        self,
        condition: str,
        split: str,
        repetition: int,
        batch_number: int,
    ) -> Path:
        return (
            self._run_root(condition, split)
            / f"rep_{repetition:02d}"
            / f"batch_{batch_number:04d}"
        )

    def _read_records(self, batch_dir: Path, pattern: str) -> list[dict]:
        records = []
        for path in sorted(batch_dir.glob(pattern)):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise FingerprintMismatch(
                    f"Corrupt benchmark artifact: {path}"
                ) from error
            if (
                not isinstance(record, dict)
                or record.get("fingerprint") != self._active_fingerprint
                or record.get("fingerprint_components")
                != self._active_components
            ):
                raise FingerprintMismatch(
                    f"Benchmark fingerprint mismatch: {path}"
                )
            records.append(record)
        return records

    def _next_number(self, batch_dir: Path, prefix: str) -> int:
        paths = sorted(batch_dir.glob(f"{prefix}_*.json"))
        if not paths:
            return 1
        try:
            return max(int(path.stem.rsplit("_", 1)[1]) for path in paths) + 1
        except (ValueError, IndexError) as error:
            raise FingerprintMismatch(
                f"Malformed append-only artifact name in {batch_dir}"
            ) from error

    def _write_request(
        self,
        batch_dir: Path,
        *,
        messages: list[dict],
        raw_response: Any,
        usage: dict,
        provider_model: str | None,
        started_at: str,
        error: BaseException | None,
        request_succeeded: bool,
        schema_issues: dict,
        normalized_response: Any = None,
        response_truncated: bool = False,
        selected_retry_wait_seconds: float | None = None,
    ) -> None:
        number = self._next_number(batch_dir, "request")
        record = {
            "kind": "actual_http_request",
            "request_number": number,
            "safe_prompt": messages,
            "raw_response": raw_response,
            "normalized_response": normalized_response,
            "token_usage": usage,
            "provider_model": provider_model,
            "normalized_provider_model": _normalized_provider_model(
                provider_model
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
            "started_at": started_at,
            "finished_at": _utc_now(),
            "error": _safe_error(error, self._secrets) if error else None,
            "request_succeeded": bool(request_succeeded),
            "response_truncated": bool(response_truncated),
            "schema_issues": schema_issues,
            "selected_retry_wait_seconds": (
                selected_retry_wait_seconds
            ),
            "fingerprint": self._active_fingerprint,
            "fingerprint_components": self._active_components,
            "test_mode": self.test_mode,
        }
        self._write_json(
            batch_dir / f"request_{number:04d}.json",
            record,
        )

    def _recording_client(
        self,
        underlying: object,
        batch_dir: Path,
    ) -> RecordingStructuredClient:
        return RecordingStructuredClient(
            underlying,
            record_request=lambda **kwargs: self._write_request(
                batch_dir,
                **kwargs,
            ),
            max_request_retries=self.max_request_retries,
            retry_sleep=self.retry_sleep,
            secrets=self._secrets,
            retry_backoff_cap=self.retry_backoff_cap,
        )

    def _write_result(
        self,
        batch_dir: Path,
        *,
        expected_ids: list[str],
        predictions: list[dict],
        raw_response: Any,
        complete: bool,
        error: BaseException | None = None,
    ) -> None:
        number = self._next_number(batch_dir, "batch_result")
        self._write_json(
            batch_dir / f"batch_result_{number:02d}.json",
            {
                "kind": "batch_result",
                "expected_variant_ids": expected_ids,
                "normalized_predictions": predictions,
                "raw_response": raw_response,
                "token_usage": {},
                "safe_prompt": None,
                "started_at": _utc_now(),
                "finished_at": _utc_now(),
                "error": (
                    _safe_error(error, self._secrets) if error else None
                ),
                "complete": bool(complete),
                "fingerprint": self._active_fingerprint,
                "fingerprint_components": self._active_components,
                "test_mode": self.test_mode,
            },
        )

    def _batch_complete(
        self,
        batch_dir: Path,
        expected_ids: list[str],
    ) -> bool:
        self._read_records(batch_dir, "request_*.json")
        records = self._read_records(batch_dir, "batch_result_*.json")
        if not records:
            return False
        for record in records:
            if record.get("expected_variant_ids") != expected_ids:
                raise FingerprintMismatch(
                    "Batch expected variant IDs changed"
                )
        final = records[-1]
        predictions = final.get("normalized_predictions")
        if not isinstance(predictions, list):
            raise FingerprintMismatch("Invalid batch result predictions")
        ids = [str(item.get("variant_id", "")) for item in predictions]
        if len(ids) != len(set(ids)):
            raise FingerprintMismatch("Duplicate batch prediction IDs")
        return bool(final.get("complete")) and sorted(ids) == expected_ids

    @staticmethod
    def _strict_full_predictions(
        batch: list[dict],
        grounded: Any,
        recorder: RecordingStructuredClient | None,
    ) -> tuple[list[dict], bool]:
        expected = {str(item["variant_id"]): item for item in batch}
        returned: dict[str, dict] = {}
        duplicates: set[str] = set()
        if isinstance(grounded, list):
            for item in grounded:
                if not isinstance(item, dict):
                    continue
                claim_id = str(item.get("claim_id", ""))
                if claim_id in returned:
                    duplicates.add(claim_id)
                elif claim_id in expected:
                    returned[claim_id] = item
        predictions = []
        complete = len(returned) == len(expected) and not duplicates
        for claim_id, variant in expected.items():
            record = returned.get(claim_id)
            verdict = record.get("verdict") if isinstance(record, dict) else None
            normalized = (
                verdict.strip().casefold()
                if isinstance(verdict, str)
                else ""
            )
            reasons = [
                str(item.get("reason", ""))
                for item in record.get("evidence_results", [])
                if isinstance(item, dict)
            ] if isinstance(record, dict) else []
            technical_unverifiable = (
                normalized == "unverifiable"
                and "grounding_result_missing" in reasons
            )
            schema_failure = (
                claim_id in duplicates
                or normalized not in _GROUNDING_VERDICTS
                or technical_unverifiable
            )
            complete = complete and not schema_failure
            predictions.append(
                _prediction(
                    variant,
                    verdict,
                    schema_failure=schema_failure,
                )
            )
        return predictions, complete

    async def _run_full(
        self,
        batch: list[dict],
        batch_dir: Path,
        expected_ids: list[str],
    ) -> None:
        if self.verifier is None:
            raise ValueError("full condition requires verifier")
        _full_inputs(batch)
        recorder = None
        verifier = self.verifier
        underlying = getattr(self.verifier, "llm", None)
        if isinstance(self.verifier, ClaimVerifier):
            recorder = self._recording_client(underlying, batch_dir)
            verifier = ClaimVerifier(
                recorder,
                task=self.verifier.task,
                call_timeout_seconds=self.verifier.call_timeout_seconds,
                max_concurrency=self.verifier.max_concurrency,
            )
        try:
            _, grounded = await _judge_full_with_raw(
                verifier,
                batch,
                self.target,
            )
        except Exception as error:
            terminal = (
                recorder.terminal_error
                if recorder is not None
                else None
            )
            cause = terminal or error
            failed = [
                _prediction(item, None, schema_failure=True)
                for item in batch
            ]
            self._write_result(
                batch_dir,
                expected_ids=expected_ids,
                predictions=failed,
                raw_response=None,
                complete=False,
                error=cause,
            )
            raise IncompleteCondition("Full batch failed") from cause
        if (
            recorder is not None
            and (
                recorder.terminal_error is not None
                or recorder.final_transport_failures
            )
        ):
            terminal = recorder.terminal_error or BenchmarkTerminalFailure(
                "Benchmark Full request ended in a terminal failure"
            )
            failed = [
                _prediction(item, None, schema_failure=True)
                for item in batch
            ]
            self._write_result(
                batch_dir,
                expected_ids=expected_ids,
                predictions=failed,
                raw_response=None,
                complete=False,
                error=terminal,
            )
            raise IncompleteCondition(
                "Full batch ended in a terminal request failure"
            ) from terminal
        predictions, complete = self._strict_full_predictions(
            batch,
            grounded,
            recorder,
        )
        self._write_result(
            batch_dir,
            expected_ids=expected_ids,
            predictions=predictions,
            raw_response=grounded,
            complete=complete,
        )
        if not complete:
            raise IncompleteCondition(
                "Full batch has unresolved technical/schema failures"
            )

    async def _holistic_request(
        self,
        batch: list[dict],
        batch_dir: Path,
        *,
        repair: bool,
    ) -> tuple[list[dict], list[dict], Any]:
        if self.llm is None:
            raise ValueError("holistic condition requires llm")
        recorder = self._recording_client(self.llm, batch_dir)
        messages = _holistic_messages(batch, repair=repair)
        raw, _ = await recorder.structured(
            messages,
            schema={
                "type": "object",
                "required": ["items"],
                "properties": {"items": {"type": "array"}},
            },
            max_tokens=2048,
            task="evidence_integrity_holistic",
            temperature=0,
        )
        predictions, unresolved = _parse_holistic(batch, raw)
        return predictions, unresolved, raw

    async def _run_holistic(
        self,
        batch: list[dict],
        batch_dir: Path,
        expected_ids: list[str],
    ) -> None:
        try:
            predictions, unresolved, raw = await self._holistic_request(
                batch,
                batch_dir,
                repair=False,
            )
        except Exception as error:
            failed = [
                _prediction(item, None, schema_failure=True)
                for item in batch
            ]
            self._write_result(
                batch_dir,
                expected_ids=expected_ids,
                predictions=failed,
                raw_response=None,
                complete=False,
                error=error,
            )
            raise IncompleteCondition("Holistic request failed") from error
        if not unresolved:
            self._write_result(
                batch_dir,
                expected_ids=expected_ids,
                predictions=predictions,
                raw_response=raw,
                complete=True,
            )
            return
        self._write_result(
            batch_dir,
            expected_ids=expected_ids,
            predictions=predictions,
            raw_response=raw,
            complete=False,
        )
        try:
            repaired, still_unresolved, repair_raw = (
                await self._holistic_request(
                    unresolved,
                    batch_dir,
                    repair=True,
                )
            )
        except Exception as error:
            self._write_result(
                batch_dir,
                expected_ids=expected_ids,
                predictions=predictions,
                raw_response=None,
                complete=False,
                error=error,
            )
            raise IncompleteCondition("Holistic repair failed") from error
        by_id = {item["variant_id"]: item for item in predictions}
        by_id.update({item["variant_id"]: item for item in repaired})
        final = [by_id[str(item["variant_id"])] for item in batch]
        complete = not still_unresolved and all(
            not item["schema_failure"] for item in final
        )
        self._write_result(
            batch_dir,
            expected_ids=expected_ids,
            predictions=final,
            raw_response=repair_raw,
            complete=complete,
        )
        if not complete:
            raise IncompleteCondition("Holistic repair remains unresolved")

    async def _run_id_only(
        self,
        batch: list[dict],
        batch_dir: Path,
        expected_ids: list[str],
    ) -> None:
        self._write_result(
            batch_dir,
            expected_ids=expected_ids,
            predictions=run_id_only(batch),
            raw_response={"deterministic": True},
            complete=True,
        )

    async def run_condition(self, condition: str, split: str) -> None:
        if condition not in {"full", "holistic", "id_only"}:
            raise ValueError(f"Unknown condition: {condition}")
        if split not in self.variants:
            raise ValueError(f"Unknown split: {split}")
        if (
            condition == "full"
            and not isinstance(self.verifier, ClaimVerifier)
            and not self.test_mode
        ):
            raise TypeError(
                "Paid Full condition requires a real ClaimVerifier; "
                "test doubles require test_mode=True"
            )
        ordered = sorted(
            copy.deepcopy(self.variants[split]),
            key=lambda item: str(item.get("variant_id", "")),
        )
        if not ordered:
            raise ValueError("Benchmark split is empty")
        variant_ids = [str(item.get("variant_id", "")) for item in ordered]
        if any(not variant_id for variant_id in variant_ids):
            raise ValueError("Every variant must have a variant_id")
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("Variant IDs must be unique")
        repetitions = 3 if condition in JUDGED_CONDITIONS else 1
        (
            self._active_fingerprint,
            self._active_components,
        ) = self._fingerprint_for(condition, split, repetitions)
        lock = self._acquire_lock(condition, split)
        primary_error: BaseException | None = None
        try:
            condition_batch_size = (
                self.holistic_batch_size
                if condition == "holistic"
                else self.batch_size
            )
            batches = [
                ordered[start : start + condition_batch_size]
                for start in range(0, len(ordered), condition_batch_size)
            ]
            for repetition in range(1, repetitions + 1):
                for batch_number, batch in enumerate(batches, start=1):
                    expected_ids = sorted(
                        str(item["variant_id"]) for item in batch
                    )
                    batch_dir = self._batch_dir(
                        condition,
                        split,
                        repetition,
                        batch_number,
                    )
                    if self._batch_complete(batch_dir, expected_ids):
                        continue
                    if condition == "full":
                        await self._run_full(
                            batch,
                            batch_dir,
                            expected_ids,
                        )
                    elif condition == "holistic":
                        await self._run_holistic(
                            batch,
                            batch_dir,
                            expected_ids,
                        )
                    else:
                        await self._run_id_only(
                            batch,
                            batch_dir,
                            expected_ids,
                        )
            for repetition in range(1, repetitions + 1):
                for batch_number, batch in enumerate(batches, start=1):
                    expected_ids = sorted(
                        str(item["variant_id"]) for item in batch
                    )
                    if not self._batch_complete(
                        self._batch_dir(
                            condition,
                            split,
                            repetition,
                            batch_number,
                        ),
                        expected_ids,
                    ):
                        raise IncompleteCondition(
                            "Condition lacks all required repetition artifacts"
                        )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                self._release_lock(lock)
            except BaseException as release_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    f"Run lock release also failed: {release_error!r}"
                )
                raise primary_error from release_error
