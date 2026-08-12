from __future__ import annotations

import asyncio
import copy
import json
import re

from dp_indicator.agents.claim_verifier import ClaimVerifier

MAX_GAP_GROUPS = 19
MAX_QUERIES_PER_GROUP = 2
MAX_RESULTS_PER_QUERY = 5
MAX_BRIDGE_PAPERS = 3


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def collect_uncited_parent_groups(
    claims: list[dict],
    max_groups: int = MAX_GAP_GROUPS,
) -> list[dict]:
    groups: dict[tuple[str, str], dict] = {}
    for claim in claims:
        if claim.get("verdict") != "unverifiable":
            continue
        if claim.get("evidence_results"):
            continue
        if claim.get("origin", {}).get("kind") != "causal_step":
            continue
        key = (
            str(claim.get("hypothesis_id", "")),
            str(claim.get("parent_claim_id", "")),
        )
        group = groups.setdefault(
            key,
            {
                "hypothesis_id": key[0],
                "parent_claim_id": key[1],
                "origin": copy.deepcopy(claim.get("origin", {})),
                "claims": [],
            },
        )
        group["claims"].append(copy.deepcopy(claim))
    limit = min(max(max_groups, 0), MAX_GAP_GROUPS)
    return list(groups.values())[:limit]


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(title).lower())


def _extract_pmid(item: dict) -> str:
    evidence_id = str(item.get("evidence_id", ""))
    if evidence_id.startswith("PMID:"):
        return evidence_id.split(":", 1)[1]
    metadata = item.get("source_metadata") or {}
    pmid = metadata.get("pmid")
    return str(pmid) if pmid else ""


def _extract_doi(item: dict) -> str:
    evidence_id = str(item.get("evidence_id", ""))
    if evidence_id.startswith("DOI:"):
        return evidence_id.split(":", 1)[1].lower()
    metadata = item.get("source_metadata") or {}
    doi = metadata.get("doi")
    return str(doi).lower() if doi else ""


def _candidate_keys(item: dict) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    pmid = _extract_pmid(item)
    if pmid:
        keys.append(("pmid", pmid))
    doi = _extract_doi(item)
    if doi:
        keys.append(("doi", doi))
    title = _normalize_title(item.get("title", ""))
    if title:
        keys.append(("title", title))
    return keys


def deduplicate_candidates(items: list[dict]) -> list[dict]:
    if not items:
        return []

    union_find = _UnionFind(len(items))
    key_to_index: dict[tuple[str, str], int] = {}

    for index, item in enumerate(items):
        linked_indices: list[int] = []
        for key in _candidate_keys(item):
            if key in key_to_index:
                linked_indices.append(key_to_index[key])
            else:
                key_to_index[key] = index
        for linked_index in linked_indices:
            union_find.union(index, linked_index)

    component_min: dict[int, int] = {}
    for index in range(len(items)):
        root = union_find.find(index)
        current_min = component_min.get(root)
        if current_min is None or index < current_min:
            component_min[root] = index

    return [items[index] for index in sorted(component_min.values())]


def _verdict_weight(verdict: str) -> float:
    return {
        "supported": 1.0,
        "partial": 0.5,
        "unsupported": 0.0,
        "contradicted": 0.0,
    }.get(verdict, 0.0)


def _result_weight(result: dict) -> float:
    if result.get("quote_verified") is not True:
        return 0.0
    return _verdict_weight(str(result.get("verdict", "")))


def _candidate_has_contradiction(candidate: dict) -> bool:
    return any(
        result.get("verdict") == "contradicted"
        for result in candidate.get("results", [])
    )


def _union_coverage(claims: list[dict], candidates: list[dict]) -> float:
    claim_weights = {
        str(claim.get("claim_id", "")): float(claim.get("weight", 1.0))
        for claim in claims
    }
    total_weight = sum(claim_weights.values())
    if total_weight <= 0:
        return 0.0

    best_by_claim = {claim_id: 0.0 for claim_id in claim_weights}
    for candidate in candidates:
        for result in candidate.get("results", []):
            claim_id = str(result.get("claim_id", ""))
            if claim_id not in best_by_claim:
                continue
            weight = _result_weight(result)
            best_by_claim[claim_id] = max(best_by_claim[claim_id], weight)

    covered = sum(
        best_by_claim[claim_id] * claim_weights[claim_id]
        for claim_id in claim_weights
    )
    return covered / total_weight


def select_bridge_set(
    claims: list[dict],
    candidates: list[dict],
    max_papers: int = MAX_BRIDGE_PAPERS,
) -> dict:
    paper_limit = min(max(max_papers, 0), MAX_BRIDGE_PAPERS)
    eligible = [
        candidate
        for candidate in candidates
        if not _candidate_has_contradiction(candidate)
    ]
    selected: list[dict] = []

    while len(selected) < paper_limit:
        current_coverage = _union_coverage(claims, selected)
        best_candidate: dict | None = None
        best_gain = 0.0

        for candidate in eligible:
            if candidate in selected:
                continue
            new_coverage = _union_coverage(claims, selected + [candidate])
            gain = new_coverage - current_coverage
            if gain > best_gain:
                best_gain = gain
                best_candidate = candidate

        if best_candidate is None or best_gain <= 0:
            break
        selected.append(best_candidate)

    coverage = _union_coverage(claims, selected)
    if coverage >= 0.60:
        decision = "retain_supported"
    elif coverage >= 0.30:
        decision = "retain_partial"
    else:
        decision = "remove"

    verified_spans: list[dict] = []
    seen_spans: set[tuple[str, str]] = set()
    for candidate in selected:
        evidence_id = str(candidate.get("evidence_id", ""))
        for result in candidate.get("results", []):
            if result.get("quote_verified") is not True:
                continue
            if result.get("verdict") not in {"supported", "partial"}:
                continue
            quote = str(result.get("quote", ""))
            if not quote:
                continue
            span_key = (evidence_id, quote)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            verified_spans.append(
                {
                    "claim_id": result.get("claim_id", ""),
                    "evidence_id": evidence_id,
                    "verdict": result.get("verdict", ""),
                    "quote": quote,
                    "quote_verified": True,
                    "evidence_role": "bridge_evidence",
                }
            )

    return {
        "evidence_ids": [str(candidate.get("evidence_id", "")) for candidate in selected],
        "coverage": coverage,
        "decision": decision,
        "verified_spans": verified_spans,
    }


def fallback_query(indication: str, claims: list[dict]) -> str:
    terms = " ".join(str(claim.get("text", "")) for claim in claims)
    words = re.findall(r"[A-Za-z0-9α-ωΑ-Ω+-]+", terms)
    unique = list(dict.fromkeys(word for word in words if len(word) > 3))
    return f'"{indication}" ' + " ".join(unique[:8])


class GapEvidenceRetriever:
    def __init__(
        self,
        llm: object,
        search_client: object,
        claim_verifier: ClaimVerifier,
        bridge_relevance_checker: object | None = None,
        max_groups: int = MAX_GAP_GROUPS,
        max_queries_per_group: int = MAX_QUERIES_PER_GROUP,
        max_results_per_query: int = MAX_RESULTS_PER_QUERY,
        max_selected_papers: int = MAX_BRIDGE_PAPERS,
        planning_timeout_seconds: float = 120,
    ):
        self.llm = llm
        self.search_client = search_client
        self.claim_verifier = claim_verifier
        self.bridge_relevance_checker = bridge_relevance_checker
        self._planning_errors: list[dict] = []
        self.max_groups = min(max(max_groups, 0), MAX_GAP_GROUPS)
        self.max_queries_per_group = min(
            max(max_queries_per_group, 0),
            MAX_QUERIES_PER_GROUP,
        )
        self.max_results_per_query = min(
            max(max_results_per_query, 0),
            MAX_RESULTS_PER_QUERY,
        )
        self.max_selected_papers = min(
            max(max_selected_papers, 0),
            MAX_BRIDGE_PAPERS,
        )
        self.planning_timeout_seconds = planning_timeout_seconds

    async def _plan_queries(
        self,
        groups: list[dict],
        indications: dict[str, str],
        target: str,
    ) -> dict[tuple[str, str], list[str]]:
        if not groups or self.max_queries_per_group == 0:
            return {}
        payload = [
            {
                "hypothesis_id": group["hypothesis_id"],
                "parent_claim_id": group["parent_claim_id"],
                "indication": indications.get(group["hypothesis_id"], ""),
                "claims": [
                    str(claim.get("text", ""))
                    for claim in group["claims"]
                ],
            }
            for group in groups
        ]
        prompt = f"""## Task
Plan bounded PubMed queries for each uncited causal gap.
Return no more than {self.max_queries_per_group} concise queries per group.
Target: {target}
Groups:
{json.dumps(payload, ensure_ascii=False)}

Return JSON:
{{"groups": [{{"hypothesis_id": "...", "parent_claim_id": "...", "queries": ["..."]}}]}}"""
        self._planning_errors = []
        try:
            response = await asyncio.wait_for(
                self.llm.structured(
                    [
                        {
                            "role": "system",
                            "content": (
                                "You plan conservative biomedical PubMed "
                                "queries. Return JSON only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2048,
                    task="reasoner",
                    temperature=0,
                ),
                timeout=self.planning_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            self._planning_errors.append(
                {
                    "stage": "planner_timeout",
                    "error_type": type(exc).__name__,
                    "message": "query planner timed out; deterministic fallback used",
                }
            )
            return {}
        except Exception as exc:
            self._planning_errors.append(
                {
                    "stage": "planner_failure",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            return {}
        result = response[0] if isinstance(response, tuple) else response
        if not isinstance(result, dict):
            self._planning_errors.append(
                {
                    "stage": "planner_parse",
                    "error_type": "InvalidResponse",
                    "message": "planner returned a non-object response",
                }
            )
            return {}

        planned: dict[tuple[str, str], list[str]] = {}
        try:
            raw_groups = result.get("groups", [])
            if not isinstance(raw_groups, list):
                self._planning_errors.append(
                    {
                        "stage": "planner_parse",
                        "error_type": "InvalidGroups",
                        "message": "planner groups was not a list",
                    }
                )
                return {}
            for item in raw_groups:
                if not isinstance(item, dict):
                    continue
                raw_queries = item.get("queries", [])
                if not isinstance(raw_queries, list):
                    continue
                queries = [
                    query.strip()
                    for query in raw_queries
                    if isinstance(query, str) and query.strip()
                ][: self.max_queries_per_group]
                if not queries:
                    continue
                hypothesis_id = str(item.get("hypothesis_id", ""))
                parent_claim_id = str(item.get("parent_claim_id", ""))
                planned[(hypothesis_id, parent_claim_id)] = queries
                if not hypothesis_id:
                    planned[("", parent_claim_id)] = queries
        except Exception as exc:
            self._planning_errors.append(
                {
                    "stage": "planner_parse",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            return {}
        return planned

    async def _check_bridge_relevance(
        self,
        group: dict,
        candidates: list[dict],
        target: str,
    ) -> dict[str, dict]:
        """Return injected relevance judgments keyed by evidence ID."""
        if self.bridge_relevance_checker is None:
            return {
                str(candidate.get("evidence_id", "")): {
                    "relevance": "unknown",
                    "reason": "relevance_checker_unavailable",
                }
                for candidate in candidates
            }
        result = self.bridge_relevance_checker(group, candidates, target)
        if hasattr(result, "__await__"):
            result = await result
        if not isinstance(result, dict):
            raise TypeError("bridge relevance checker returned a non-dict result")
        return {
            str(evidence_id): value
            for evidence_id, value in result.items()
            if isinstance(value, dict)
        }

    async def retrieve(
        self,
        claims: list[dict],
        hypotheses: list[dict],
        target: str,
    ) -> dict:
        groups = collect_uncited_parent_groups(claims, self.max_groups)
        indications = {
            str(
                hypothesis.get("hypothesis_id")
                or hypothesis.get("indication")
                or ""
            ): str(hypothesis.get("indication", ""))
            for hypothesis in hypotheses
        }
        plans = await self._plan_queries(groups, indications, target)
        output_groups: list[dict] = []
        all_evidence: list[dict] = []
        errors: list[dict] = list(self._planning_errors)
        resolved_parent_steps = 0
        covered_claim_ids: set[str] = set()

        for group in groups:
            hypothesis_id = str(group["hypothesis_id"])
            parent_claim_id = str(group["parent_claim_id"])
            group["indication"] = indications.get(hypothesis_id, "")
            queries = plans.get(
                (hypothesis_id, parent_claim_id),
                plans.get(("", parent_claim_id), []),
            )
            if not queries and self.max_queries_per_group:
                queries = [
                    fallback_query(
                        indications.get(hypothesis_id, ""),
                        group["claims"],
                    )
                ]
            queries = queries[: self.max_queries_per_group]

            retrieved: list[dict] = []
            for query in queries:
                try:
                    items = await self.search_client.search(
                        query,
                        max_results=self.max_results_per_query,
                    )
                except Exception as exc:
                    errors.append(
                        {
                            "stage": "retrieval",
                            "hypothesis_id": hypothesis_id,
                            "parent_claim_id": parent_claim_id,
                            "query": query,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    continue
                for item in list(items or [])[: self.max_results_per_query]:
                    if not isinstance(item, dict):
                        continue
                    if not str(item.get("abstract_snippet", "")).strip():
                        continue
                    candidate = copy.deepcopy(item)
                    candidate["evidence_role"] = "bridge_evidence"
                    candidate["retrieval_reason"] = "uncited_causal_gap"
                    candidate["bridge_parent_claim_id"] = parent_claim_id
                    retrieved.append(candidate)

            # PubMed order is stable input order. Deduplicate before applying
            # the hard three-paper grounding budget.
            candidates = deduplicate_candidates(retrieved)[
                : self.max_selected_papers
            ]
            grounded_claims: list[dict] = []
            relevance: dict[str, dict] = {}
            rejected_candidates: list[dict] = []
            try:
                relevance = await self._check_bridge_relevance(
                    group,
                    candidates,
                    target,
                )
                accepted_candidates = []
                for candidate in candidates:
                    evidence_id = str(candidate.get("evidence_id", ""))
                    judgment = relevance.get(evidence_id, {})
                    relevance_value = (
                        judgment.get("relevance")
                        if isinstance(judgment, dict)
                        else None
                    )
                    if relevance_value in {"direct", "indirect"}:
                        accepted_candidates.append(candidate)
                        continue
                    normalized_relevance = (
                        str(relevance_value)
                        if relevance_value not in (None, "")
                        else "unknown"
                    )
                    reason = (
                        str(judgment.get("reason", ""))
                        if isinstance(judgment, dict)
                        else ""
                    ) or "relevance_not_approved"
                    rejected_candidates.append(
                        {
                            "evidence_id": evidence_id,
                            "title": candidate.get("title", ""),
                            "source_metadata": copy.deepcopy(
                                candidate.get("source_metadata") or {}
                            ),
                            "relevance": {
                                "relevance": normalized_relevance,
                                "reason": reason,
                            },
                        }
                    )
                    errors.append(
                        {
                            "stage": "bridge_relevance",
                            "hypothesis_id": hypothesis_id,
                            "parent_claim_id": parent_claim_id,
                            "evidence_id": evidence_id,
                            "error_type": "RelevanceRejected",
                            "message": reason,
                        }
                    )
                candidates = accepted_candidates
            except Exception as exc:
                errors.append(
                    {
                        "stage": "bridge_relevance",
                        "hypothesis_id": hypothesis_id,
                        "parent_claim_id": parent_claim_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                for candidate in candidates:
                    evidence_id = str(candidate.get("evidence_id", ""))
                    relevance[evidence_id] = {
                        "relevance": "unknown",
                        "reason": "relevance_checker_failed",
                    }
                    rejected_candidates.append(
                        {
                            "evidence_id": evidence_id,
                            "title": candidate.get("title", ""),
                            "source_metadata": copy.deepcopy(
                                candidate.get("source_metadata") or {}
                            ),
                            "relevance": {
                                "relevance": "unknown",
                                "reason": "relevance_checker_failed",
                            },
                        }
                    )
                candidates = []
            if candidates:
                try:
                    claims_to_ground = copy.deepcopy(group["claims"])
                    evidence_ids = [
                        str(candidate.get("evidence_id", ""))
                        for candidate in candidates
                        if candidate.get("evidence_id")
                    ]
                    for claim in claims_to_ground:
                        claim["evidence_ids"] = evidence_ids
                    metadata = {
                        str(candidate["evidence_id"]): {
                            **copy.deepcopy(candidate.get("source_metadata") or {}),
                            "title": candidate.get("title", ""),
                            "abstract": candidate.get("abstract_snippet", ""),
                        }
                        for candidate in candidates
                        if candidate.get("evidence_id")
                    }
                    grounded_claims = (
                        await self.claim_verifier.ground_existing_claims(
                            claims_to_ground,
                            metadata,
                            target,
                        )
                    )
                    for candidate in candidates:
                        evidence_id = str(candidate.get("evidence_id", ""))
                        candidate["results"] = [
                            {
                                "claim_id": claim.get("claim_id", ""),
                                **copy.deepcopy(result),
                            }
                            for claim in grounded_claims
                            for result in claim.get("evidence_results", [])
                            if str(result.get("evidence_id", "")) == evidence_id
                        ]
                except Exception as exc:
                    errors.append(
                        {
                            "stage": "grounding",
                            "hypothesis_id": hypothesis_id,
                            "parent_claim_id": parent_claim_id,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    candidates = []
                    grounded_claims = []

            selected = select_bridge_set(
                group["claims"],
                candidates,
                self.max_selected_papers,
            )
            selected["evidence_role"] = "bridge_evidence"
            if selected["decision"] in {
                "retain_supported",
                "retain_partial",
            }:
                resolved_parent_steps += 1
                covered_claim_ids.update(
                    str(span.get("claim_id", ""))
                    for span in selected["verified_spans"]
                    if span.get("claim_id")
                    and span.get("quote_verified") is True
                    and span.get("verdict") in {"supported", "partial"}
                )
            output_groups.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "parent_claim_id": parent_claim_id,
                    "origin": copy.deepcopy(group["origin"]),
                    "queries": queries,
                    "candidate_evidence": candidates,
                    "grounded_candidates": candidates,
                    "grounded_claims": grounded_claims,
                    "selected": selected,
                    "bridge_relevance": relevance,
                    "rejected_candidates": rejected_candidates,
                }
            )
            all_evidence.extend(candidates)

        total_atomic_claims = sum(len(group["claims"]) for group in groups)
        return {
            "groups": output_groups,
            "evidence": deduplicate_candidates(all_evidence),
            "summary": {
                "gap_parent_steps": len(groups),
                "searched_parent_steps": sum(
                    1 for group in output_groups if group["queries"]
                ),
                "resolved_parent_steps": resolved_parent_steps,
                "unresolved_atomic_claims": (
                    total_atomic_claims - len(covered_claim_ids)
                ),
                "errors": errors,
            },
        }
