"""Evidence Verifier Agent - claim-level verification of evidence accuracy.

Verifies that:
  V1: Evidence IDs (PMID/EPMC/DOI) exist in external databases
  Claim Grounding: atomic claims are bound to exact source spans
  V2/V3: compatibility views derived from Claim Grounding
  V4: Evidence is relevant to the hypothesis indication (no tissue mismatch etc.)
  V5: Evidence counts in report match actual pool

Design principles:
- Model-produced descriptions are claims, never trusted source quotations
- Use full abstracts from EvidenceCacheClient (not truncated)
- Validate every returned quote against source text in deterministic code
- Degrade gracefully to retrieved abstract snippets or unverifiable status
- Results feed back to SemanticScorer for score adjustment
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging

from dp_indicator.clients.evidence_cache import EvidenceCacheClient

logger = logging.getLogger(__name__)


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


def _issue_ids(issues: list[dict], severity: str) -> set[str]:
    return {
        str(issue.get("evidence_id", ""))
        for issue in issues
        if issue.get("severity") == severity and issue.get("evidence_id")
    }


def apply_bridge_evidence(
    hypotheses: list[dict],
    evidence_pool: list[dict],
    retrieval: dict,
) -> None:
    """Attach retained bridge sets to their exact originating causal steps."""
    hypotheses_by_id = {
        str(
            hypothesis.get("hypothesis_id")
            or hypothesis.get("indication")
            or ""
        ): hypothesis
        for hypothesis in hypotheses
    }
    retrieved_by_id = {
        str(item.get("evidence_id", "")): item
        for item in (retrieval or {}).get("evidence", []) or []
        if isinstance(item, dict) and item.get("evidence_id")
    }
    pool_ids = {
        str(item.get("evidence_id") or item.get("id") or "")
        for item in evidence_pool
        if isinstance(item, dict)
    }

    for group in (retrieval or {}).get("groups", []) or []:
        if not isinstance(group, dict):
            continue
        selected = group.get("selected") or {}
        decision = selected.get("decision")
        if decision not in {"retain_supported", "retain_partial"}:
            continue
        origin = group.get("origin") or {}
        if origin.get("kind") != "causal_step":
            continue
        hypothesis = hypotheses_by_id.get(
            str(group.get("hypothesis_id", ""))
        )
        if hypothesis is None:
            continue
        try:
            axis_index = int(origin["axis_index"])
            step_index = int(origin["step_index"])
            step = hypothesis["causal_chain"]["mechanism_axes"][
                axis_index
            ]["steps"][step_index]
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if not isinstance(step, dict):
            continue

        selected_ids = list(
            dict.fromkeys(
                str(evidence_id)
                for evidence_id in selected.get("evidence_ids", [])
                if evidence_id
            )
        )
        step_ids = step.setdefault("evidence_ids", [])
        cited_ids = hypothesis.setdefault("cited_evidence_ids", [])
        step_cited_ids = step.setdefault("cited_evidence_ids", [])
        sources = step.setdefault("sources", [])
        source_ids = {
            str(source.get("evidence_id", ""))
            for source in sources
            if isinstance(source, dict)
        }
        for evidence_id in selected_ids:
            if evidence_id not in step_ids:
                step_ids.append(evidence_id)
            if evidence_id not in cited_ids:
                cited_ids.append(evidence_id)
            if evidence_id not in step_cited_ids:
                step_cited_ids.append(evidence_id)
            evidence = retrieved_by_id.get(evidence_id, {})
            if evidence_id not in source_ids:
                metadata = copy.deepcopy(evidence.get("source_metadata") or {})
                source = {
                    "evidence_id": evidence_id,
                    "evidence_role": "bridge_evidence",
                    "retrieval_reason": "uncited_causal_gap",
                    "title": evidence.get("title", ""),
                    "source_metadata": metadata,
                }
                source.update(metadata)
                sources.append(source)
                source_ids.add(evidence_id)
            if evidence_id not in pool_ids:
                bridge_record = copy.deepcopy(evidence)
                bridge_record["evidence_id"] = evidence_id
                bridge_record["evidence_role"] = "bridge_evidence"
                bridge_record["retrieval_reason"] = "uncited_causal_gap"
                evidence_pool.append(bridge_record)
                pool_ids.add(evidence_id)

        verified_spans = step.setdefault("verified_spans", [])
        span_keys = {
            (
                str(span.get("claim_id", "")),
                str(span.get("evidence_id", "")),
                str(span.get("quote", "")),
            )
            for span in verified_spans
            if isinstance(span, dict)
        }
        for span in selected.get("verified_spans", []) or []:
            if (
                not isinstance(span, dict)
                or str(span.get("evidence_id", "")) not in selected_ids
            ):
                continue
            normalized = copy.deepcopy(span)
            normalized["evidence_role"] = "bridge_evidence"
            key = (
                str(normalized.get("claim_id", "")),
                str(normalized.get("evidence_id", "")),
                str(normalized.get("quote", "")),
            )
            if key not in span_keys:
                verified_spans.append(normalized)
                span_keys.add(key)
        step["status"] = (
            "supported" if decision == "retain_supported" else "inferred"
        )


def claim_grounding_to_legacy_issues(
    claims: list[dict],
) -> tuple[dict[str, list], dict[str, list]]:
    """Project claim-level results into the V2/V3 issue views."""
    v2: dict[str, list] = {}
    v3: dict[str, list] = {}
    severity_by_verdict = {
        "partial": "medium",
        "mixed": "medium",
        "unsupported": "high",
        "contradicted": "high",
        "unverifiable": "warning",
    }
    for claim in claims:
        hid = str(claim.get("hypothesis_id", ""))
        origin = claim.get("origin", {})
        target = v2 if origin.get("kind") == "causal_step" else v3
        for result in claim.get("evidence_results", []):
            if result.get("evidence_role") == "bridge_evidence":
                continue
            verdict = str(result.get("verdict", "unverifiable")).lower()
            severity = severity_by_verdict.get(verdict)
            if not severity:
                continue
            issue = {
                "claim_id": claim.get("claim_id", ""),
                "evidence_id": result.get("evidence_id", ""),
                "issue": (
                    f"Claim {verdict}: "
                    f"{result.get('reason', '')}"
                ).strip(),
                "severity": severity,
            }
            if target is v2:
                issue["step"] = origin.get("layer", "")
            target.setdefault(hid, []).append(issue)
    return v2, v3


def compute_claim_score_adjustments(
    claims: list[dict],
    v4_results: dict,
    hypotheses: list[dict],
) -> dict[str, float]:
    """Compute proportional factual, availability, and relevance penalties."""
    error_weights = {
        "supported": 0.0,
        "partial": 0.5,
        "mixed": 0.5,
        "unsupported": 1.0,
        "contradicted": 1.0,
    }
    claims_by_hypothesis: dict[str, list[dict]] = {}
    for claim in claims:
        claims_by_hypothesis.setdefault(
            str(claim.get("hypothesis_id", "")),
            [],
        ).append(claim)

    adjustments: dict[str, float] = {}
    for hypothesis in hypotheses:
        hid = str(
            hypothesis.get("hypothesis_id")
            or hypothesis.get("indication")
            or ""
        )
        unique_claims = []
        seen_claims: set[str] = set()
        for claim in claims_by_hypothesis.get(hid, []):
            claim_id = str(claim.get("claim_id", ""))
            if claim_id and claim_id in seen_claims:
                continue
            if claim_id:
                seen_claims.add(claim_id)
            unique_claims.append(claim)
        verifiable = [
            claim
            for claim in unique_claims
            if str(claim.get("verdict", "")).lower() in error_weights
        ]
        unverifiable_count = sum(
            1
            for claim in unique_claims
            if str(claim.get("verdict", "")).lower() == "unverifiable"
        )
        penalty = 0.0
        if verifiable:
            weighted_error_ratio = sum(
                error_weights[
                    str(claim.get("verdict", "")).lower()
                ]
                for claim in verifiable
            ) / len(verifiable)
            penalty -= 0.13 * weighted_error_ratio
        total_count = len(verifiable) + unverifiable_count
        if total_count:
            unverifiable_ratio = unverifiable_count / total_count
            if unverifiable_ratio > 0.40:
                penalty -= 0.02 * min(
                    (unverifiable_ratio - 0.40) / 0.60,
                    1.0,
                )
        for issue in v4_results.get(hid, []):
            severity = issue.get("severity", "")
            if severity == "high":
                penalty -= 0.04
            elif severity == "low":
                penalty -= 0.01
        penalty = max(penalty, -0.15)
        if penalty < 0:
            adjustments[hid] = round(penalty, 6)
    return adjustments


def compute_gap_acceptance(
    uncited_before: list[dict],
    claims: list[dict],
    retrieval: dict,
) -> dict:
    """Measure accepted bridge coverage without counting mere LLM output."""
    before_ids = {
        str(claim.get("claim_id", ""))
        for claim in uncited_before
        if claim.get("claim_id")
    }
    retained_ids = {
        str(evidence_id)
        for group in (retrieval or {}).get("groups", []) or []
        if isinstance(group, dict)
        and (group.get("selected") or {}).get("decision")
        in {"retain_supported", "retain_partial"}
        for evidence_id in (group.get("selected") or {}).get(
            "evidence_ids", []
        )
        if evidence_id
    }
    resolved_ids = {
        str(claim.get("claim_id", ""))
        for claim in claims
        if str(claim.get("claim_id", "")) in before_ids
        and str(claim.get("verdict", "")).lower()
        in {"supported", "partial"}
        and any(
            result.get("evidence_role") == "bridge_evidence"
            and result.get("quote_verified") is True
            and str(result.get("verdict", "")).lower()
            in {"supported", "partial"}
            and str(result.get("evidence_id", "")) in retained_ids
            for result in claim.get("evidence_results", [])
            if isinstance(result, dict)
        )
    }
    denominator = len(before_ids)
    gain = len(resolved_ids) / denominator if denominator else 0.0
    return {
        "uncited_atomic_before": denominator,
        "supported_or_partial_after": len(resolved_ids),
        "coverage_gain": gain,
        "acceptance_threshold": 0.30,
        "acceptance_passed": gain >= 0.30 if denominator else False,
    }


def _claim_effects_for_hypothesis(
    claims: list[dict],
    hypothesis_id: str,
    bridge_aggregates: list[dict] | None = None,
) -> dict:
    effects = {
        "step_remove": {},
        "step_downgrade": set(),
        "step_spans": {},
        "mapping_remove": set(),
        "mapping_annotations": {},
    }
    from dp_indicator.agents.claim_verifier import (
        aggregate_parent_claims,
    )

    strong_steps = set()
    weak_steps = set()
    for aggregate in aggregate_parent_claims(claims):
        if str(aggregate.get("hypothesis_id", "")) != hypothesis_id:
            continue
        origin = aggregate.get("origin", {})
        evidence_id = str(aggregate.get("evidence_id", ""))
        decision = aggregate.get("decision")
        if origin.get("kind") == "causal_step":
            key = (
                int(origin.get("axis_index", -1)),
                int(origin.get("step_index", -1)),
            )
            if decision == "remove" and evidence_id:
                effects["step_remove"].setdefault(key, set()).add(
                    evidence_id
                )
                weak_steps.add(key)
            elif decision == "retain_supported":
                if aggregate.get("unverifiable_ratio", 1.0) <= 0.40:
                    strong_steps.add(key)
                else:
                    weak_steps.add(key)
            else:
                weak_steps.add(key)
            if decision in {"retain_supported", "retain_partial"}:
                effects["step_spans"].setdefault(key, []).extend(
                    aggregate.get("verified_spans", [])
                )
        elif origin.get("kind") == "evidence_mapping":
            key = (
                str(origin.get("bucket", "")),
                int(origin.get("item_index", -1)),
            )
            if decision == "remove":
                effects["mapping_remove"].add(key)
            else:
                effects["mapping_annotations"][key] = {
                    "claim_id": aggregate.get("parent_claim_id", ""),
                    "verification_status": decision,
                    "verification_coverage": aggregate.get("coverage"),
                    "verified_spans": aggregate.get("verified_spans", []),
                }
    for aggregate in bridge_aggregates or []:
        if (
            str(aggregate.get("hypothesis_id", "")) != hypothesis_id
            or aggregate.get("aggregate_kind") != "bridge_set"
        ):
            continue
        origin = aggregate.get("origin", {})
        if origin.get("kind") != "causal_step":
            continue
        key = (
            int(origin.get("axis_index", -1)),
            int(origin.get("step_index", -1)),
        )
        decision = aggregate.get("decision")
        if decision == "retain_supported":
            strong_steps.add(key)
        elif decision == "retain_partial":
            weak_steps.add(key)
        else:
            continue
        effects["step_spans"].setdefault(key, []).extend(
            aggregate.get("verified_spans", [])
        )
    effects["step_downgrade"] = weak_steps - strong_steps
    return effects


def sanitize_hypotheses(
    hypotheses: list[dict],
    verification_result: dict,
) -> tuple[list[dict], list[dict]]:
    """Conservatively sanitize verified hypotheses without mutating raw output.

    Hard failures (non-existent IDs and high-severity V2/V3 issues) are removed.
    Medium V2/V3 and V4 relevance issues remain visible but lose support or gain
    explicit annotations. Every deterministic change is returned in an audit log.
    """
    sanitized = copy.deepcopy(hypotheses)
    report = verification_result.get("verification_report", {}) or {}
    v1 = report.get("v1_id_existence", {}) or {}
    globally_invalid = {
        str(evidence_id)
        for evidence_id, result in v1.items()
        if result.get("exists") is False
    }
    audit: list[dict] = []

    for hypothesis in sanitized:
        hid = str(
            hypothesis.get("hypothesis_id")
            or hypothesis.get("indication")
            or ""
        )
        v2_issues = (report.get("v2_citation_accuracy", {}) or {}).get(hid, [])
        v3_issues = (report.get("v3_description_accuracy", {}) or {}).get(hid, [])
        v4_issues = (report.get("v4_indication_relevance", {}) or {}).get(hid, [])
        claim_effects = _claim_effects_for_hypothesis(
            (
                report.get("claim_grounding", {}).get("claims", [])
                if isinstance(report.get("claim_grounding", {}), dict)
                else []
            ),
            hid,
            (
                report.get("claim_grounding", {}).get(
                    "bridge_aggregates",
                    [],
                )
                if isinstance(report.get("claim_grounding", {}), dict)
                else []
            ),
        )
        has_claim_grounding = isinstance(
            report.get("claim_grounding"),
            dict,
        )

        hard_remove = set(globally_invalid)
        if not has_claim_grounding:
            hard_remove.update(_issue_ids(v2_issues, "high"))
            hard_remove.update(_issue_ids(v3_issues, "high"))

        partial_by_step = (
            set()
            if has_claim_grounding
            else {
                (
                    str(issue.get("step", "")),
                    str(issue.get("evidence_id", "")),
                )
                for issue in v2_issues
                if issue.get("severity") == "medium"
                and issue.get("evidence_id")
            }
        )
        overinterpreted = (
            set()
            if has_claim_grounding
            else _issue_ids(v3_issues, "medium")
        )
        relevance = {}
        for issue in v4_issues:
            evidence_id = str(issue.get("evidence_id", ""))
            text = str(issue.get("issue", "")).lower()
            if evidence_id:
                relevance[evidence_id] = (
                    "mismatch" if "mismatch" in text else "indirect"
                )

        chain = hypothesis.get("causal_chain", {})
        indexed_steps = []
        if isinstance(chain, dict) and "mechanism_axes" in chain:
            indexed_steps = [
                ((axis_index, step_index), step)
                for axis_index, axis in enumerate(
                    chain.get("mechanism_axes", [])
                )
                for step_index, step in enumerate(axis.get("steps", []))
                if isinstance(step, dict)
            ]
        else:
            indexed_steps = [
                ((-1, index), step)
                for index, step in enumerate(_hypothesis_steps(hypothesis))
            ]

        for step_key, step in indexed_steps:
            step_name = str(step.get("layer", step.get("level", "")))
            original_ids = [
                str(evidence_id)
                for evidence_id in step.get("evidence_ids", [])
                if evidence_id
            ]
            claim_remove = claim_effects["step_remove"].get(
                step_key,
                set(),
            )
            retained_ids = [
                evidence_id
                for evidence_id in original_ids
                if evidence_id not in hard_remove
                and evidence_id not in claim_remove
            ]
            for evidence_id in original_ids:
                if evidence_id not in retained_ids:
                    audit.append(
                        {
                            "hypothesis_id": hid,
                            "action": "remove_citation",
                            "location": step_name,
                            "evidence_id": evidence_id,
                            "reason": (
                                "unsupported_claim"
                                if evidence_id in claim_remove
                                else "hard_verification_failure"
                            ),
                        }
                    )
            step["evidence_ids"] = retained_ids
            verified_spans = [
                span
                for span in claim_effects["step_spans"].get(step_key, [])
                if span.get("evidence_id") in retained_ids
            ]
            if verified_spans:
                step["verified_spans"] = verified_spans
            has_partial = any(
                (step_name, evidence_id) in partial_by_step
                for evidence_id in retained_ids
            )
            if step.get("status") == "supported" and (
                not retained_ids
                or has_partial
                or step_key in claim_effects["step_downgrade"]
            ):
                step["status"] = "inferred"
                audit.append(
                    {
                        "hypothesis_id": hid,
                        "action": "downgrade_step",
                        "location": step_name,
                        "reason": (
                            "partial_citation"
                            if has_partial
                            else (
                                "claim_not_fully_supported"
                                if step_key
                                in claim_effects["step_downgrade"]
                                else "no_verified_citations"
                            )
                        ),
                    }
                )

        mapping = hypothesis.get("evidence_mapping", {})
        if isinstance(mapping, dict):
            for bucket in (
                "positive_evidence",
                "indirect_evidence",
                "contradicting_evidence",
            ):
                items = mapping.get(bucket, [])
                if not isinstance(items, list):
                    continue
                retained_items = []
                seen_ids: set[str] = set()
                for item_index, item in enumerate(items):
                    if not isinstance(item, dict):
                        retained_items.append(item)
                        continue
                    evidence_id = str(
                        item.get("id") or item.get("evidence_id") or ""
                    )
                    mapping_key = (bucket, item_index)
                    if mapping_key in claim_effects["mapping_remove"]:
                        audit.append(
                            {
                                "hypothesis_id": hid,
                                "action": "remove_mapping",
                                "location": bucket,
                                "evidence_id": evidence_id,
                                "reason": "unsupported_claim",
                            }
                        )
                        continue
                    if evidence_id and evidence_id in hard_remove:
                        audit.append(
                            {
                                "hypothesis_id": hid,
                                "action": "remove_mapping",
                                "location": bucket,
                                "evidence_id": evidence_id,
                                "reason": "hard_verification_failure",
                            }
                        )
                        continue
                    if evidence_id and evidence_id in seen_ids:
                        audit.append(
                            {
                                "hypothesis_id": hid,
                                "action": "remove_duplicate",
                                "location": bucket,
                                "evidence_id": evidence_id,
                                "reason": "duplicate_evidence_id",
                            }
                        )
                        continue
                    if evidence_id:
                        seen_ids.add(evidence_id)
                    if evidence_id in overinterpreted:
                        item["verification_status"] = "overinterpreted"
                        audit.append(
                            {
                                "hypothesis_id": hid,
                                "action": "annotate_mapping",
                                "location": bucket,
                                "evidence_id": evidence_id,
                                "reason": "overinterpreted",
                            }
                        )
                    if evidence_id in relevance:
                        item["verification_relevance"] = relevance[evidence_id]
                        audit.append(
                            {
                                "hypothesis_id": hid,
                                "action": "annotate_mapping",
                                "location": bucket,
                                "evidence_id": evidence_id,
                                "reason": relevance[evidence_id],
                            }
                        )
                    annotation = claim_effects[
                        "mapping_annotations"
                    ].get(mapping_key)
                    if annotation:
                        item.update(annotation)
                    retained_items.append(item)
                mapping[bucket] = retained_items

    if isinstance(report.get("claim_grounding"), dict):
        from dp_indicator.agents.claim_verifier import (
            recheck_sanitized_hypotheses,
        )

        sanitized, recheck_audit = recheck_sanitized_hypotheses(sanitized)
        audit.extend(recheck_audit)

    return sanitized, audit


def _remaining_evidence_ids(hypotheses: list[dict]) -> set[str]:
    remaining: set[str] = set()
    for hypothesis in hypotheses:
        for step in _hypothesis_steps(hypothesis):
            remaining.update(
                str(evidence_id)
                for evidence_id in step.get("evidence_ids", [])
                if evidence_id
            )
        mapping = hypothesis.get("evidence_mapping", {})
        if isinstance(mapping, dict):
            for bucket in (
                "positive_evidence",
                "indirect_evidence",
                "contradicting_evidence",
            ):
                for item in mapping.get(bucket, []):
                    if isinstance(item, dict):
                        evidence_id = item.get("id") or item.get("evidence_id")
                        if evidence_id:
                            remaining.add(str(evidence_id))
        remaining.update(
            str(evidence_id)
            for evidence_id in hypothesis.get("evidence_ids", [])
            if evidence_id
        )
    return remaining


def filter_verification_report(
    raw_report: dict,
    sanitized_hypotheses: list[dict],
) -> dict:
    """Filter verifier findings to IDs still present in the final output."""
    remaining = _remaining_evidence_ids(sanitized_hypotheses)
    filtered = copy.deepcopy(raw_report)
    filtered["v1_id_existence"] = {
        evidence_id: result
        for evidence_id, result in (
            raw_report.get("v1_id_existence", {}) or {}
        ).items()
        if evidence_id in remaining
    }
    claim_grounding = raw_report.get("claim_grounding")
    if isinstance(claim_grounding, dict):
        from dp_indicator.agents.claim_verifier import (
            aggregate_claim_verdict,
        )

        filtered_claims = []
        for claim in claim_grounding.get("claims", []):
            retained_results = [
                copy.deepcopy(result)
                for result in claim.get("evidence_results", [])
                if str(result.get("evidence_id", "")) in remaining
            ]
            if not retained_results:
                continue
            retained_claim = copy.deepcopy(claim)
            retained_claim["evidence_results"] = retained_results
            direct_verdict = str(claim.get("verdict", "")).lower()
            if direct_verdict not in {"unsupported", "contradicted"}:
                direct_results = [
                    result
                    for result in retained_results
                    if result.get("evidence_role") != "bridge_evidence"
                ]
                retained_claim["verdict"] = (
                    aggregate_claim_verdict(direct_results)
                    if direct_results
                    else direct_verdict
                )
            filtered_claims.append(retained_claim)
        by_hypothesis: dict[str, list[dict]] = {}
        for claim in filtered_claims:
            by_hypothesis.setdefault(
                str(claim.get("hypothesis_id", "")),
                [],
            ).append(claim)
        filtered["claim_grounding"] = {
            "claims": filtered_claims,
            "by_hypothesis": by_hypothesis,
            "bridge_aggregates": [],
        }
        for aggregate in claim_grounding.get("bridge_aggregates", []) or []:
            retained_ids = [
                str(evidence_id)
                for evidence_id in aggregate.get("evidence_ids", [])
                if str(evidence_id) in remaining
            ]
            if not retained_ids:
                continue
            retained_aggregate = copy.deepcopy(aggregate)
            retained_aggregate["evidence_ids"] = retained_ids
            retained_aggregate["verified_spans"] = [
                copy.deepcopy(span)
                for span in aggregate.get("verified_spans", [])
                if str(span.get("evidence_id", "")) in retained_ids
            ]
            filtered["claim_grounding"]["bridge_aggregates"].append(
                retained_aggregate
            )
    for section_name in (
        "v2_citation_accuracy",
        "v3_description_accuracy",
        "v4_indication_relevance",
    ):
        filtered_section = {}
        for hid, issues in (raw_report.get(section_name, {}) or {}).items():
            retained_issues = [
                copy.deepcopy(issue)
                for issue in issues
                if not issue.get("evidence_id")
                or str(issue.get("evidence_id")) in remaining
            ]
            filtered_section[hid] = retained_issues
        filtered[section_name] = filtered_section
    return filtered


class EvidenceVerifier:
    """Verify evidence accuracy after hypothesis generation."""

    def __init__(self, llm: object, audit: object,
                 model: str = "glm-5.1", task: str = "reasoner",
                 cache_client: EvidenceCacheClient = None,
                 gap_retriever: object | None = None,
                 enable_gap_retrieval: bool = True):
        self.llm = llm
        self.audit = audit
        self.model = model
        self.task = task
        self.cache_client = cache_client or EvidenceCacheClient()
        self.gap_retriever = gap_retriever
        self.enable_gap_retrieval = enable_gap_retrieval

    def _build_gap_retriever(self):
        from dp_indicator.agents.claim_verifier import ClaimVerifier
        from dp_indicator.agents.gap_evidence_retriever import (
            GapEvidenceRetriever,
        )
        from dp_indicator.clients.databases import PubMedClient

        return GapEvidenceRetriever(
            llm=self.llm,
            search_client=PubMedClient(),
            claim_verifier=ClaimVerifier(
                llm=self.llm,
                task=self.task,
            ),
            bridge_relevance_checker=self._verify_bridge_relevance,
        )

    async def verify(self, hypotheses: list[dict],
                     evidence_pool: list[dict],
                     target: str = "") -> dict:
        """Run all verification checks.

        Returns:
            {
                "verified_hypotheses": [...],  # hypotheses with verification annotations
                "verification_report": {
                    "v1_id_existence": {...},
                    "claim_grounding": {...},
                    "v2_citation_accuracy": {...},
                    "v3_description_accuracy": {...},
                    "v4_indication_relevance": {...},
                    "v5_count_consistency": {...},
                },
                "score_adjustments": {hyp_id: float_delta},
                "summary": str,
            }
        """
        print("  [heartbeat] EvidenceVerifier: starting verification...", flush=True)

        # Collect all evidence IDs cited in hypotheses
        all_cited_ids = set()
        for hyp in hypotheses:
            chain = hyp.get("causal_chain", {})
            if isinstance(chain, dict):
                if "mechanism_axes" in chain:
                    for axis in chain.get("mechanism_axes", []):
                        for step in axis.get("steps", []):
                            all_cited_ids.update(step.get("evidence_ids", []))
                else:
                    for link in chain.values():
                        if isinstance(link, dict):
                            all_cited_ids.update(link.get("evidence_ids", []))
            eh = hyp.get("evidence_mapping", {})
            for pos in eh.get("positive_evidence", []):
                if isinstance(pos, dict) and pos.get("id"):
                    all_cited_ids.add(pos["id"])
            for ind in eh.get("indirect_evidence", []):
                if isinstance(ind, dict) and ind.get("id"):
                    all_cited_ids.add(ind["id"])
            for neg in eh.get("contradicting_evidence", []):
                if isinstance(neg, dict) and neg.get("id"):
                    all_cited_ids.add(neg["id"])

        print(f"  [heartbeat] EvidenceVerifier: {len(all_cited_ids)} unique evidence IDs to verify across {len(hypotheses)} hypotheses", flush=True)

        # V1: ID existence check (pure API, no LLM)
        print("  [heartbeat] EvidenceVerifier V1: checking ID existence...", flush=True)
        v1_results = await self._v1_check_id_existence(list(all_cited_ids))

        # Fetch full metadata for all cited evidence
        cited_metadata = {}
        if all_cited_ids:
            cited_metadata = await self.cache_client.fetch_batch(list(all_cited_ids))
        cited_metadata = copy.deepcopy(cited_metadata)
        for evidence in evidence_pool:
            evidence_id = str(
                evidence.get("evidence_id") or evidence.get("id") or ""
            )
            if not evidence_id or evidence_id not in all_cited_ids:
                continue
            metadata = cited_metadata.setdefault(evidence_id, {})
            if not metadata.get("title") and evidence.get("title"):
                metadata["title"] = evidence.get("title")
            if (
                not metadata.get("abstract")
                and evidence.get("abstract_snippet")
            ):
                metadata["abstract"] = evidence.get("abstract_snippet")

        # Derive overlapping V2/V3 issue views from the claim ledger.
        print(
            "  [heartbeat] EvidenceVerifier: grounding atomic claims...",
            flush=True,
        )
        from dp_indicator.agents.claim_verifier import (
            ClaimVerifier,
            aggregate_bridge_claims,
            merge_bridge_grounding,
        )

        claim_grounding = await ClaimVerifier(
            llm=self.llm,
            task=self.task,
        ).verify(hypotheses, cited_metadata, target)
        uncited_atomic_before = [
            copy.deepcopy(claim)
            for claim in claim_grounding.get("claims", [])
            if claim.get("verdict") == "unverifiable"
            and not claim.get("evidence_results")
        ]
        gap_retrieval = {
            "groups": [],
            "evidence": [],
            "summary": {},
        }
        if self.enable_gap_retrieval:
            try:
                retriever = self.gap_retriever or self._build_gap_retriever()
                retrieved = await retriever.retrieve(
                    claim_grounding.get("claims", []),
                    hypotheses,
                    target,
                )
                if not isinstance(retrieved, dict):
                    raise TypeError("gap retriever returned a non-dict result")
                gap_retrieval = retrieved
                merge_bridge_grounding(
                    claim_grounding.get("claims", []),
                    gap_retrieval,
                )
                apply_bridge_evidence(
                    hypotheses,
                    evidence_pool,
                    gap_retrieval,
                )
            except Exception as exc:
                logger.warning("Gap evidence retrieval failed: %s", exc)
                gap_retrieval = {
                    "groups": [],
                    "evidence": [],
                    "summary": {
                        "errors": [
                            f"{type(exc).__name__}: {exc}",
                        ],
                    },
                }
        claim_grounding["bridge_aggregates"] = aggregate_bridge_claims(
            claim_grounding.get("claims", []),
            gap_retrieval,
        )
        gap_retrieval.setdefault("summary", {}).update(
            compute_gap_acceptance(
                uncited_atomic_before,
                claim_grounding.get("claims", []),
                gap_retrieval,
            )
        )
        v2_results, v3_results = claim_grounding_to_legacy_issues(
            claim_grounding.get("claims", [])
        )

        # V4: Indication relevance / tissue mismatch (LLM)
        print("  [heartbeat] EvidenceVerifier V4: checking indication relevance...", flush=True)
        v4_results = await self._v4_check_indication_relevance(hypotheses, cited_metadata, target)

        # V5: Count consistency (pure Python)
        print("  [heartbeat] EvidenceVerifier V5: checking count consistency...", flush=True)
        v5_results = self._v5_check_count_consistency(hypotheses, evidence_pool)

        # Compute score adjustments
        score_adjustments = compute_claim_score_adjustments(
            claim_grounding.get("claims", []),
            v4_results,
            hypotheses,
        )

        # Annotate hypotheses with verification results
        for hyp in hypotheses:
            hid = hyp.get("hypothesis_id", hyp.get("indication", ""))
            hyp["verification"] = {
                "claim_grounding": claim_grounding.get(
                    "by_hypothesis",
                    {},
                ).get(hid, []),
                "v2_citation_issues": v2_results.get(hid, []),
                "v3_description_issues": v3_results.get(hid, []),
                "v4_relevance_issues": v4_results.get(hid, []),
                "score_adjustment": score_adjustments.get(hid, 0.0),
            }

        # Generate summary
        claim_issue_count = sum(
            1
            for claim in claim_grounding.get("claims", [])
            if claim.get("verdict") != "supported"
        )
        total_issues = claim_issue_count + sum(
            len(items) for items in v4_results.values()
        )
        summary = self._generate_summary(
            v1_results,
            claim_grounding,
            v2_results,
            v3_results,
            v4_results,
            v5_results,
            total_issues,
        )
        print(f"  [heartbeat] EvidenceVerifier: {total_issues} issues found across {len(hypotheses)} hypotheses", flush=True)
        print(f"  [heartbeat] EvidenceVerifier summary:\n{summary}", flush=True)

        self.audit.record("EvidenceVerifier", "verify", "complete", {
            "n_verified": len(all_cited_ids),
            "n_v1_failures": sum(1 for v in v1_results.values() if not v.get("exists")),
            "n_v2_issues": sum(len(v) for v in v2_results.values()),
            "n_v3_issues": sum(len(v) for v in v3_results.values()),
            "n_v4_issues": sum(len(v) for v in v4_results.values()),
            "n_claims": len(claim_grounding.get("claims", [])),
            "total_issues": total_issues,
        })

        await self.cache_client.close()

        return {
            "verified_hypotheses": hypotheses,
            "verification_report": {
                "v1_id_existence": v1_results,
                "claim_grounding": claim_grounding,
                "gap_retrieval": gap_retrieval,
                "v2_citation_accuracy": v2_results,
                "v3_description_accuracy": v3_results,
                "v4_indication_relevance": v4_results,
                "v5_count_consistency": v5_results,
            },
            "score_adjustments": score_adjustments,
            "summary": summary,
        }

    # ── V1: ID Existence (pure API) ──

    async def _v1_check_id_existence(self, evidence_ids: list[str]) -> dict[str, dict]:
        """Check if each evidence ID exists in external databases."""
        results = {}
        if not evidence_ids:
            return results

        try:
            batch_data = await self.cache_client.fetch_batch(evidence_ids)
            for eid in evidence_ids:
                if eid in batch_data and batch_data[eid].get("title"):
                    results[eid] = {"exists": True, "title": batch_data[eid]["title"]}
                else:
                    results[eid] = {"exists": False, "title": ""}
                    print(f"  [warning] V1: Evidence ID not found: {eid}", flush=True)
        except Exception as e:
            logger.warning(f"V1 check failed: {e}")
            for eid in evidence_ids:
                results[eid] = {"exists": "unknown", "title": "", "error": str(e)[:100]}

        n_missing = sum(1 for v in results.values() if not v.get("exists"))
        if n_missing:
            print(f"  [heartbeat] V1: {n_missing}/{len(evidence_ids)} IDs not found in external DBs", flush=True)
        else:
            print(f"  [heartbeat] V1: All {len(evidence_ids)} IDs verified", flush=True)

        return results

    # ── V2: Citation Accuracy (LLM) ──

    async def _v2_check_citation_accuracy(self, hypotheses: list[dict],
                                          cited_metadata: dict[str, dict],
                                          target: str) -> dict[str, list]:
        """Verify that source_text citations in causal chains match actual article content."""
        results: dict[str, list] = {}
        sem = asyncio.Semaphore(3)  # glm-5.1 concurrency limit

        tasks = []
        for hyp in hypotheses[:5]:
            hid = hyp.get("hypothesis_id", hyp.get("indication", ""))
            chain = hyp.get("causal_chain", {})
            if not isinstance(chain, dict):
                continue

            # Extract (step, evidence_id, source_text) triples
            citations = []
            if "mechanism_axes" in chain:
                for axis in chain.get("mechanism_axes", []):
                    for step in axis.get("steps", []):
                        step_name = step.get("layer", step.get("level", ""))
                        source_text = step.get("source_text", "")
                        for eid in step.get("evidence_ids", []):
                            if source_text and eid:
                                citations.append({
                                    "step": step_name,
                                    "evidence_id": eid,
                                    "cited_text": source_text[:300],
                                })
            else:
                for level, link in chain.items():
                    source_text = link.get("source_text", "")
                    for eid in link.get("evidence_ids", []):
                        if source_text and eid:
                            citations.append({
                                "step": level,
                                "evidence_id": eid,
                                "cited_text": source_text[:300],
                            })

            if citations:
                tasks.append(self._v2_verify_citations(
                    hid, citations, cited_metadata, target, sem, results
                ))

        await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def _v2_verify_citations(self, hid: str, citations: list[dict],
                                   cited_metadata: dict, target: str,
                                   sem: asyncio.Semaphore,
                                   results: dict[str, list]):
        """Verify citations for one hypothesis."""
        issues = []
        async with sem:
            for cit in citations:
                eid = cit["evidence_id"]
                cited_text = cit["cited_text"]
                metadata = cited_metadata.get(eid, {})

                if not metadata:
                    # Can't verify - API didn't return data
                    issues.append({
                        "evidence_id": eid,
                        "step": cit["step"],
                        "issue": "Could not retrieve metadata for verification",
                        "severity": "warning",
                    })
                    continue

                actual_abstract = metadata.get("abstract", "")
                actual_title = metadata.get("title", "")

                if not actual_abstract:
                    # No abstract available - skip
                    continue

                # Quick string match first (no LLM needed)
                # Check if any 50-char substring of cited_text appears in abstract
                if len(cited_text) >= 50:
                    snippet = cited_text[:50].lower()
                    if snippet in actual_abstract.lower():
                        continue  # Good match, no issue

                # LLM verification for non-matching citations
                prompt = f"""## Task
Verify if the "cited text" actually appears in or is accurately paraphrased from the article abstract.

## Article
- ID: {eid}
- Title: {actual_title}
- Abstract: {actual_abstract[:1500]}

## Cited Text (from hypothesis causal chain)
"{cited_text}"

## Verification
Does the cited text accurately represent content from this article?
- "accurate": The text is directly from or accurately paraphrases the abstract
- "misattributed": The text appears to be from a different article
- "fabricated": The text is not supported by the abstract at all
- "partial": Parts are accurate but contains additions not in the abstract

Return JSON: {{"verdict": "accurate|misattributed|fabricated|partial", "reason": "str"}}"""

                try:
                    result, _ = await self.llm.structured([
                        {"role": "system", "content": "You are a citation verification specialist. Compare cited text against the actual article abstract and determine if the citation is accurate."},
                        {"role": "user", "content": prompt},
                    ], max_tokens=256, task=self.task, temperature=0)
                    if isinstance(result, dict) and not result.get("error"):
                        verdict = result.get("verdict", "unknown")
                        reason = result.get("reason", "")
                        if verdict in ("misattributed", "fabricated", "partial"):
                            issues.append({
                                "evidence_id": eid,
                                "step": cit["step"],
                                "issue": f"Citation {verdict}: {reason[:200]}",
                                "severity": "high" if verdict in ("misattributed", "fabricated") else "medium",
                            })
                            print(f"  [warning] V2 [{hid}] {eid} step {cit['step']}: {verdict}", flush=True)
                except asyncio.TimeoutError:
                    # Should not happen (no asyncio.wait_for), but keep as safety net
                    issues.append({
                        "evidence_id": eid,
                        "step": cit["step"],
                        "issue": "LLM request timed out, skipped",
                        "severity": "warning",
                    })
                    print(f"  [warning] V2 [{hid}] {eid} step {cit['step']}: LLM timeout, skipping this item", flush=True)
                except Exception as e:
                    logger.debug(f"V2 LLM verification failed for {eid}: {e}")

        results[hid] = issues

    # ── V3: Description Accuracy (LLM) ──

    async def _v3_check_description_accuracy(self, hypotheses: list[dict],
                                             cited_metadata: dict[str, dict],
                                             target: str) -> dict[str, list]:
        """Verify that evidence descriptions/rationales match actual article content."""
        results: dict[str, list] = {}
        sem = asyncio.Semaphore(3)

        tasks = []
        for hyp in hypotheses[:5]:
            hid = hyp.get("hypothesis_id", hyp.get("indication", ""))
            eh = hyp.get("evidence_mapping", {})
            pos_evidence = eh.get("positive_evidence", [])

            if pos_evidence:
                tasks.append(self._v3_verify_descriptions(
                    hid, pos_evidence, cited_metadata, target, sem, results
                ))

        await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def _v3_verify_descriptions(self, hid: str, pos_evidence: list[dict],
                                      cited_metadata: dict, target: str,
                                      sem: asyncio.Semaphore,
                                      results: dict[str, list]):
        """Verify rationale descriptions for one hypothesis."""
        issues = []
        async with sem:
            for item in pos_evidence[:15]:  # cap to control cost
                eid = item.get("id", "")
                rationale = item.get("rationale", "")
                if not eid or not rationale:
                    continue

                metadata = cited_metadata.get(eid, {})
                if not metadata:
                    continue

                actual_abstract = metadata.get("abstract", "")
                actual_title = metadata.get("title", "")

                if not actual_abstract:
                    continue

                prompt = f"""## Task
Verify if the evidence rationale accurately describes the article's content.

## Article
- ID: {eid}
- Title: {actual_title}
- Abstract: {actual_abstract[:1200]}

## Evidence Rationale (from system's evidence mapping)
"{rationale[:400]}"

## Verification
Does the rationale accurately reflect the article's content?
- "accurate": Rationale is well-supported by the abstract
- "overinterpreted": Rationale makes claims beyond what the abstract supports
- "misattributed": Rationale describes content not in this article
- "accurate_but_vague": Rationale is technically correct but too vague

Return JSON: {{"verdict": "accurate|overinterpreted|misattributed|accurate_but_vague", "reason": "str"}}"""

                try:
                    result, _ = await self.llm.structured([
                        {"role": "system", "content": "You are an evidence accuracy reviewer. Check if the system's description of this article is accurate and not over-interpreted."},
                        {"role": "user", "content": prompt},
                    ], max_tokens=256, task=self.task, temperature=0)
                    if isinstance(result, dict) and not result.get("error"):
                        verdict = result.get("verdict", "unknown")
                        reason = result.get("reason", "")
                        if verdict in ("overinterpreted", "misattributed"):
                            issues.append({
                                "evidence_id": eid,
                                "issue": f"Rationale {verdict}: {reason[:200]}",
                                "severity": "medium" if verdict == "overinterpreted" else "high",
                            })
                            print(f"  [warning] V3 [{hid}] {eid}: {verdict}", flush=True)
                except asyncio.TimeoutError:
                    print(f"  [warning] V3 [{hid}] {eid}: LLM timeout, skipping", flush=True)
                except Exception as e:
                    logger.debug(f"V3 LLM verification failed for {eid}: {e}")

        results[hid] = issues

    async def _verify_bridge_relevance(
        self,
        group: dict,
        candidates: list[dict],
        target: str,
    ) -> dict[str, dict]:
        """Independently check bridge papers before they can support a step."""
        evidence_items = [
            {
                "id": str(candidate.get("evidence_id", "")),
                "title": str(candidate.get("title", ""))[:150],
                "abstract": str(candidate.get("abstract_snippet", ""))[:800],
            }
            for candidate in candidates
            if candidate.get("evidence_id")
        ]
        if not evidence_items:
            return {}
        prompt = f"""## Task
Check whether each bridge paper is relevant to the disease and tissue context
of this causal claim. A mismatch cannot be used as positive bridge evidence.

Target: {target}
Indication: {group.get("indication", "")}
Claims: {json.dumps(group.get("claims", []), ensure_ascii=False)}
Evidence: {json.dumps(evidence_items, ensure_ascii=False)}

Return JSON:
{{"items": [{{"id": "...", "relevance": "direct|indirect|mismatch",
"reason": "..."}}]}}"""
        result, _ = await asyncio.wait_for(
            self.llm.structured(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a conservative biomedical relevance "
                            "reviewer. Return JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                task=self.task,
                temperature=0,
            ),
            timeout=120,
        )
        if not isinstance(result, dict):
            raise TypeError("bridge relevance verifier returned a non-dict result")
        return {
            str(item.get("id", "")): {
                "relevance": str(item.get("relevance", "mismatch")),
                "reason": str(item.get("reason", "")),
            }
            for item in result.get("items", [])
            if isinstance(item, dict) and item.get("id")
        }

    # ── V4: Indication Relevance / Tissue Mismatch (LLM) ──

    async def _v4_check_indication_relevance(self, hypotheses: list[dict],
                                             cited_metadata: dict[str, dict],
                                             target: str) -> dict[str, list]:
        """Check for tissue/organ mismatch between evidence and hypothesis indication."""
        results: dict[str, list] = {}
        sem = asyncio.Semaphore(3)

        tasks = []
        for hyp in hypotheses[:5]:
            hid = hyp.get("hypothesis_id", hyp.get("indication", ""))
            indication = hyp.get("indication", "")
            eh = hyp.get("evidence_mapping", {})
            pos_evidence = eh.get("positive_evidence", [])

            if pos_evidence:
                tasks.append(self._v4_verify_relevance(
                    hid, indication, pos_evidence, cited_metadata, target, sem, results
                ))

        await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def _v4_verify_relevance(self, hid: str, indication: str,
                                   pos_evidence: list[dict],
                                   cited_metadata: dict, target: str,
                                   sem: asyncio.Semaphore,
                                   results: dict[str, list]):
        """Verify indication relevance for one hypothesis."""
        issues = []
        async with sem:
            # Batch verify: send all positive evidence for this hypothesis in one LLM call
            evidence_items = []
            for item in pos_evidence[:15]:
                eid = item.get("id", "")
                metadata = cited_metadata.get(eid, {})
                if metadata:
                    evidence_items.append({
                        "id": eid,
                        "title": metadata.get("title", "")[:150],
                        "abstract": metadata.get("abstract", "")[:500],
                    })

            if not evidence_items:
                results[hid] = []
                return

            prompt = f"""## Task
Check if the evidence studies are directly relevant to the hypothesis indication, or if there's a tissue/organ mismatch.

## Hypothesis
- Target: {target}
- Indication: {indication}

## Evidence Items
{json.dumps(evidence_items, ensure_ascii=False, indent=2)}

## Verification
For each item, check:
1. Was the study conducted in the same tissue/organ system as the indication?
2. Is the evidence directly relevant to {indication}?
3. If the study was in a different system (e.g., CNS microglia for a kidney disease), flag it.

Return JSON:
{{
  "items": [
    {{"id": "...", "relevance": "direct|indirect|mismatch", "reason": "..."}}
  ]
}}"""

            try:
                result, _ = await self.llm.structured([
                    {"role": "system", "content": "You are a biomedical evidence reviewer specializing in tissue-specific relevance assessment."},
                    {"role": "user", "content": prompt},
                ], max_tokens=512, task=self.task, temperature=0)
                if isinstance(result, dict) and not result.get("error"):
                    items = result.get("items", [])
                    for item in items:
                        relevance = item.get("relevance", "direct")
                        reason = item.get("reason", "")
                        if relevance == "mismatch":
                            issues.append({
                                "evidence_id": item.get("id", ""),
                                "issue": f"Tissue/organ mismatch: {reason[:200]}",
                                "severity": "high",
                            })
                            print(f"  [warning] V4 [{hid}] {item.get('id', '')}: tissue mismatch", flush=True)
                        elif relevance == "indirect":
                            issues.append({
                                "evidence_id": item.get("id", ""),
                                "issue": f"Indirect relevance: {reason[:200]}",
                                "severity": "low",
                            })
            except asyncio.TimeoutError:
                print(f"  [warning] V4 [{hid}]: LLM timeout, skipping", flush=True)
            except Exception as e:
                logger.debug(f"V4 LLM verification failed for {hid}: {e}")

        results[hid] = issues

    # ── V5: Count Consistency (pure Python) ──

    def _v5_check_count_consistency(self, hypotheses: list[dict],
                                    evidence_pool: list[dict]) -> dict:
        """Check if evidence counts in hypotheses match the actual pool."""
        results = {}

        for hyp in hypotheses[:5]:
            hid = hyp.get("hypothesis_id", hyp.get("indication", ""))
            eh = hyp.get("evidence_mapping", {})
            pos_count = len(eh.get("positive_evidence", []))
            ind_count = len(eh.get("indirect_evidence", []))
            neg_count = len(eh.get("contradicting_evidence", []))
            total_mapped = pos_count + ind_count + neg_count

            issues = []
            if total_mapped == 0 and len(evidence_pool) > 10:
                issues.append({
                    "issue": "No evidence mapped despite substantial evidence pool",
                    "severity": "high",
                })

            # Check for duplicate evidence IDs
            all_ids = []
            for pos in eh.get("positive_evidence", []):
                if isinstance(pos, dict):
                    all_ids.append(pos.get("id", ""))
            duplicates = [eid for eid in all_ids if all_ids.count(eid) > 1]
            if duplicates:
                issues.append({
                    "issue": f"Duplicate evidence IDs in positive_evidence: {set(duplicates)}",
                    "severity": "medium",
                })

            if issues:
                results[hid] = issues

        return results

    # ── Score Adjustments ──

    def _compute_score_adjustments(self, v2_results: dict, v3_results: dict,
                                   v4_results: dict, hypotheses: list[dict]) -> dict[str, float]:
        """Compute score adjustments based on verification results.

        Penalty logic:
        - Each V2 "misattributed"/"fabricated": -0.05
        - Each V2 "partial": -0.02
        - Each V3 "overinterpreted": -0.03
        - Each V3 "misattributed": -0.05
        - Each V4 "mismatch": -0.04
        - Each V4 "indirect": -0.01
        - Maximum total penalty: -0.15 (capped)
        """
        adjustments = {}
        for hyp in hypotheses:
            hid = hyp.get("hypothesis_id", hyp.get("indication", ""))
            penalty = 0.0

            for issue in v2_results.get(hid, []):
                sev = issue.get("severity", "")
                if sev == "high":
                    penalty -= 0.05
                elif sev == "medium":
                    penalty -= 0.02

            for issue in v3_results.get(hid, []):
                sev = issue.get("severity", "")
                if sev == "high":
                    penalty -= 0.05
                elif sev == "medium":
                    penalty -= 0.03

            for issue in v4_results.get(hid, []):
                sev = issue.get("severity", "")
                if sev == "high":
                    penalty -= 0.04
                elif sev == "low":
                    penalty -= 0.01

            # Cap penalty
            penalty = max(penalty, -0.15)
            if penalty < 0:
                adjustments[hid] = penalty

        return adjustments

    # ── Summary Generation ──

    @staticmethod
    def _generate_summary(
        v1: dict,
        claim_grounding: dict,
        v2: dict,
        v3: dict,
        v4: dict,
        v5: dict,
        total_issues: int,
    ) -> str:
        """Generate a human-readable verification summary."""
        lines = []
        lines.append("=== Evidence Verification Summary ===")

        # V1
        v1_total = len(v1)
        v1_found = sum(1 for v in v1.values() if v.get("exists"))
        lines.append(f"V1 ID Existence: {v1_found}/{v1_total} verified")

        claims = claim_grounding.get("claims", [])
        supported = sum(
            1 for claim in claims if claim.get("verdict") == "supported"
        )
        verdict_counts = {}
        for claim in claims:
            verdict = str(claim.get("verdict", "unverifiable"))
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        details = ", ".join(
            f"{verdict}={count}"
            for verdict, count in sorted(verdict_counts.items())
        )
        lines.append(
            f"Claim Grounding: {supported}/{len(claims)} supported"
            + (f" ({details})" if details else "")
        )

        # V2/V3 are compatibility views generated from Claim Grounding.
        v2_total = sum(len(v) for v in v2.values())
        lines.append(f"V2 Compatibility View: {v2_total} issues found")

        v3_total = sum(len(v) for v in v3.values())
        lines.append(f"V3 Compatibility View: {v3_total} issues found")

        # V4
        v4_total = sum(len(v) for v in v4.values())
        lines.append(f"V4 Indication Relevance: {v4_total} issues found")

        # V5
        v5_total = sum(len(v) for v in v5.values())
        lines.append(f"V5 Count Consistency: {v5_total} issues found")

        lines.append(f"Total issues: {total_issues}")
        lines.append("=" * 40)

        return "\n".join(lines)
