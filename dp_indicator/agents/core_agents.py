from __future__ import annotations
import asyncio
import json
import re
import time
from typing import Optional
import copy

from dp_indicator.clients.fulltext_fetcher import FullTextFetcher


class AuditRecorder:
    def __init__(self):
        self.events = []

    def record(self, agent: str, phase: str, event_type: str, payload: dict = None):
        self.events.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "agent": agent,
            "phase": phase,
            "event_type": event_type,
            "payload": payload or {},
        })


def known_disease_deprioritized_grade(ev: dict) -> float:
    """Rank key that deprioritizes, but does not exclude, evidence tagged
    is_known_disease_evidence — i.e. evidence retrieved specifically because it
    connects the target to a disease already known/established for that target
    (see RetrieverAgent._get_associated_diseases). Such evidence tends to be
    well-cited and score high on grade_score alone, which would otherwise crowd
    mechanism-level evidence out of the small top-N slices reasoning agents see.
    A small penalty lets it still appear when nothing else is available, without
    letting it dominate purely because "already-studied" evidence is easier to find
    and grade well.
    """
    grade = ev.get("grade_score", 2)
    if ev.get("is_known_disease_evidence") and not ev.get("is_bridge_evidence"):
        grade -= 0.5
    return grade


def select_evidence_with_bridge_quota(sorted_evidence: list[dict], max_total: int,
                                       min_bridge_slots: int = 2) -> list[dict]:
    """Truncate evidence to `max_total` while reserving up to
    `min_bridge_slots` for the best-graded items tagged `is_bridge_evidence`.

    Rationale: a pure global sort by grade_score alone starves target-specific
    mechanism-bridge evidence (e.g. plasma cell / macrophage polarization studies),
    which tends to score lower than well-cited generic reviews, out of any small
    top-N cut even though it's exactly the evidence the bridge search was meant to
    surface. This keeps the same total budget but guarantees representation.

    `sorted_evidence` must already be sorted by descending priority (e.g. grade_score).
    """
    selected = list(sorted_evidence[:max_total])
    if min_bridge_slots <= 0 or not selected:
        return selected
    selected_ids = {id(e) for e in selected}
    n_bridge_in_selected = sum(1 for e in selected if e.get("is_bridge_evidence"))
    needed = min_bridge_slots - n_bridge_in_selected
    if needed <= 0:
        return selected
    bridge_candidates = [
        e for e in sorted_evidence
        if e.get("is_bridge_evidence") and id(e) not in selected_ids
    ][:needed]
    if not bridge_candidates:
        return selected
    # Bump the lowest-priority non-bridge slots (tail of the already-sorted
    # selection) to make room, preserving as much of the original ranking as possible.
    replaceable_idx = [i for i, e in enumerate(selected) if not e.get("is_bridge_evidence")]
    for idx, extra in zip(reversed(replaceable_idx), bridge_candidates):
        selected[idx] = extra
    return selected


class RetrieverAgent:
    def __init__(self, clients: dict, llm: object, audit: AuditRecorder,
                 model: str = "glm-5.1", task: str = "retriever"):
        self.clients = clients
        self.llm = llm
        self.audit = audit
        self.model = model
        self.task = task

    async def run(self, target: str, synonyms: list = None,
                  direction: str = None,
                  focus_areas: list = None) -> list[dict]:
        all_evidence = []
        retrieval_log = []
        mechanistic_terms = self._get_mechanistic_expansion_terms(target)
        if mechanistic_terms:
            mech_evidence = await self._batch_search(
                mechanistic_terms, "target+mechanism", retrieval_log
            )
            all_evidence.extend(mech_evidence)
        base_terms = [target] + (synonyms or [])
        base_evidence = await self._batch_search(
            base_terms, "target", retrieval_log
        )
        all_evidence.extend(base_evidence)
        assoc_diseases = await self._get_associated_diseases(target)
        if assoc_diseases:
            expand_terms = [f"{target} {d}" for d in assoc_diseases[:10]]
            assoc_evidence = await self._batch_search(
                expand_terms, "target+disease", retrieval_log
            )
            all_evidence.extend(assoc_evidence)
            print(f"    → 扩展检索新增: {len(assoc_evidence)} 条")
        if base_evidence and self.llm:
            try:
                expansion_terms = await self._llm_extract_expansion_terms(
                    target, base_evidence
                )
            except Exception as e:
                print(f"  ⚠️ LLM expansion keyword extraction failed, skipping: {e}", flush=True)
                expansion_terms = []
            if expansion_terms:
                print(f"  [Layer 3] LLM 扩展关键词: {expansion_terms}")
                expand_queries = [f"{target} {t}" for t in expansion_terms]
                llm_evidence = await self._batch_search(
                    expand_queries, "target+llm_expand", retrieval_log
                )
                all_evidence.extend(llm_evidence)
        if focus_areas:
            focus_terms = [f"{target} {f}" for f in focus_areas]
            focus_evidence = await self._batch_search(
                focus_terms, "target+focus", retrieval_log
            )
            all_evidence.extend(focus_evidence)
        bridge_terms = self._get_bridge_search_terms(target)
        if bridge_terms:
            bridge_evidence = await self._batch_search(
                bridge_terms, "mechanism_bridge", retrieval_log
            )
            all_evidence.extend(bridge_evidence)
            if bridge_evidence:
                print(f"  [Bridge] 机制桥接检索新增: {len(bridge_evidence)} 条")
        unique = self._dedup_evidence(all_evidence)
        self.audit.record("Retriever", "retrieve", "aggregate",
                          {"total": len(unique), "log": retrieval_log,
                           "deduped": len(all_evidence) - len(unique)})
        self._last_retrieval_log = retrieval_log
        return unique

    # Boolean retrieval-provenance tags that must survive cross-database dedup merges
    # (OR'd together rather than silently dropped when duplicates are found).
    _PROVENANCE_TAGS = ("is_bridge_evidence", "is_known_disease_evidence")

    @classmethod
    def _merge_provenance_tags(cls, dst: dict, src: dict) -> None:
        for tag in cls._PROVENANCE_TAGS:
            if src.get(tag):
                dst[tag] = True

    @classmethod
    def _dedup_evidence(cls, evidence_list: list[dict]) -> list[dict]:
        """跨数据库证据去重:PMID/EPMC 映射 + 标题相似度"""
        seen_ids = set()
        pmid_map = {}
        title_map = {}
        id_to_unique_idx = {}
        unique = []
        for ev in evidence_list:
            eid = ev.get("evidence_id", "")
            if eid and eid in seen_ids:
                # Same paper re-found via another search (e.g. also matched a
                # bridge_search_terms query): OR the provenance tags onto the kept
                # copy instead of silently dropping them.
                if eid in id_to_unique_idx:
                    cls._merge_provenance_tags(unique[id_to_unique_idx[eid]], ev)
                continue
            raw_id = ev.get("raw_id", "")
            source = ev.get("source_db", "")
            pmid_key = None
            if source == "pubmed" and raw_id.isdigit():
                pmid_key = raw_id
            elif source == "europe_pmc" and raw_id.isdigit():
                pmid_key = raw_id
            if pmid_key and pmid_key in pmid_map:
                existing = pmid_map[pmid_key]
                cls._merge_provenance_tags(existing, ev)
                if len(ev.get("abstract_snippet", "")) > len(existing.get("abstract_snippet", "")):
                    cls._merge_provenance_tags(ev, existing)
                    for i, u in enumerate(unique):
                        if u.get("evidence_id") == existing.get("evidence_id"):
                            unique[i] = ev
                            id_to_unique_idx[ev.get("evidence_id", "")] = i
                            break
                    pmid_map[pmid_key] = ev
                continue
            if pmid_key:
                pmid_map[pmid_key] = ev
            title = ev.get("title", "").strip()[:120].lower().strip()
            if title and title in title_map:
                continue
            if title:
                title_map[title] = eid
            seen_ids.add(eid)
            id_to_unique_idx[eid] = len(unique)
            unique.append(ev)
        return unique

    def _get_mechanistic_expansion_terms(self, target: str) -> list[str]:
        from dp_indicator.core.target_knowledge import get_target_profile
        profile = get_target_profile(target)
        if not profile or profile.get("official_name") == "Unknown target":
            return []
        terms = []
        for cell_type, info in profile.get("cell_type_expression", {}).items():
            if info.get("level") in ("high", "upregulated"):
                cell_name = cell_type.replace("_", " ")
                terms.append(f"{target} {cell_name}")
        functional_chain = profile.get("functional_chain", [])
        for step in functional_chain:
            for kw in ["calcium", "NFAT", "cytokine", "activation",
                       "membrane potential", "CRAC", "signaling",
                       "proliferation", "differentiation"]:
                if kw.lower() in step.lower():
                    terms.append(f"{target} {kw}")
        for category in profile.get("disease_categories", {}).keys():
            terms.append(f"{target} {category.replace('_', ' ')}")
        unique = []
        seen = set()
        for t in terms:
            t_lower = t.lower()
            if t_lower not in seen:
                seen.add(t_lower)
                unique.append(t)
        return unique[:15]

    @staticmethod
    def _get_bridge_search_terms(target: str) -> list[str]:
        from dp_indicator.core.target_knowledge import get_bridge_search_terms
        return get_bridge_search_terms(target)

    async def _batch_search(self, terms: list[str],
                            search_type: str, retrieval_log: list) -> list[dict]:
        tasks = []
        for name, client in self.clients.items():
            if not hasattr(client, 'search'):
                continue
            for term in terms:
                tasks.append(self._safe_search_with_log(
                    name, client, term, search_type, retrieval_log
                ))
        semaphore = asyncio.Semaphore(5)
        async def _bounded(t):
            async with semaphore:
                return await t
        results = await asyncio.gather(
            *[_bounded(t) for t in tasks], return_exceptions=True
        )
        all_evidence = []
        for r in results:
            if isinstance(r, list):
                all_evidence.extend(r)
        return all_evidence

    async def _safe_search_with_log(self, name: str, client, query: str,
                                     search_type: str,
                                     retrieval_log: list) -> list[dict]:
        try:
            results = await client.search(query)
            n_results = len(results)
            if search_type == "mechanism_bridge":
                # Tag bridge evidence so downstream truncation can reserve slots.
                # instead of letting it be crowded out by generic high-grade reviews.
                for r in results:
                    r["is_bridge_evidence"] = True
            elif search_type == "target+disease":
                # Tag evidence from target plus known-disease queries so the reasoner
                # can deprioritize, but not exclude, it. This evidence
                # is about connections already established for the target, so it shouldn't
                # crowd out mechanism-level evidence when reasoning about NOVEL indications.
                for r in results:
                    r["is_known_disease_evidence"] = True
            retrieval_log.append({
                "db": name, "term": query, "type": search_type,
                "status": "ok", "n_results": n_results
            })
            if n_results > 0:
                self.audit.record(name, "retrieve", "client_ok",
                                  {"term": query, "n_results": n_results,
                                   "search_type": search_type})
            return results
        except Exception as e:
            self.audit.record(name, "retrieve", "client_error",
                              {"error": str(e), "term": query, "type": search_type})
            retrieval_log.append({
                "db": name, "term": query, "type": search_type,
                "status": "error", "error": str(e)[:100]
            })
            return []

    async def _get_associated_diseases(self, target: str) -> list[str]:
        diseases = []
        ot_client = self.clients.get("opentargets")
        if ot_client:
            try:
                results = await ot_client.search(target)
                for r in results:
                    title = r.get("title", "")
                    if "→" in title:
                        disease_name = title.split("→")[-1].strip()
                        if disease_name:
                            diseases.append(disease_name)
            except Exception as e:
                self.audit.record("opentargets", "assoc_disease", "error",
                                  {"error": str(e)[:100]})
        kegg_client = self.clients.get("kegg")
        if kegg_client:
            try:
                results = await kegg_client.search(target)
                for r in results:
                    title = r.get("title", "")
                    if title and len(title) > 3:
                        diseases.append(title)
            except Exception as e:
                self.audit.record("kegg", "assoc_disease", "error",
                                  {"error": str(e)[:100]})
        unique = []
        seen = set()
        for d in diseases:
            d_lower = d.lower()
            if d_lower not in seen and len(d) > 2:
                seen.add(d_lower)
                unique.append(d)

        # Gene-level database lookups can surface associations from any
        # biological role of this gene, not just the one this profile's mechanism model
        # (cell_type_expression + functional_chain) actually covers. Drop candidates that
        # match a declared out-of-scope mechanism (e.g. a channel gene's neuronal-excitability
        # role vs. its immune-cell-activation role) — this is a mechanism-scope filter, not a
        # disease-specific one; it applies identically regardless of which disease is named.
        out_of_scope_kw = self._get_out_of_scope_keywords(target)
        if out_of_scope_kw:
            in_scope = [d for d in unique if not any(kw.lower() in d.lower() for kw in out_of_scope_kw)]
            n_dropped = len(unique) - len(in_scope)
            if n_dropped:
                print(f"    → 已知关联疾病过滤: {n_dropped} 条超出本profile机制范围(如神经兴奋性/离子通道病相关)，已排除", flush=True)
            unique = in_scope

        return unique[:15]

    def _get_out_of_scope_keywords(self, target: str) -> list[str]:
        from dp_indicator.core.target_knowledge import get_out_of_scope_keywords
        return get_out_of_scope_keywords(target)

    async def _llm_extract_expansion_terms(self, target: str,
                                           evidence: list[dict]) -> list[str]:
        if not evidence or not self.llm:
            return []
        evidence_summary = []
        for ev in evidence[:15]:
            evidence_summary.append({
                "id": ev.get("evidence_id", ""),
                "title": ev.get("title", "")[:100],
                "abstract": ev.get("abstract_snippet", "")[:150],
            })
        prompt = f"""Target: {target}
Evidence ({len(evidence_summary)} items):
{json.dumps(evidence_summary, ensure_ascii=False, indent=2)}
Based on the evidence above, extract 3-5 key disease areas, cell types, or biological pathways that this target is involved in.
These should be terms that could be used to search for additional evidence connecting this target to potential indications.
Return JSON array of strings. Example: ["autoimmune disease", "T cell activation", "macrophage", "inflammatory bowel disease", "multiple sclerosis"]"""
        max_retries = 2
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                result, _ = await asyncio.wait_for(
                    self.llm.structured(
                        [{"role": "user", "content": prompt}], max_tokens=256, task=self.task),
                    timeout=45,
                )
                if isinstance(result, list):
                    return [r for r in result if isinstance(r, str) and len(r) > 2]
                return []
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"    ⚠️ LLM expansion error (attempt {attempt+1}): {e}, retrying...", flush=True)
                    continue
                raise RuntimeError(
                    f"LLM expansion keyword extraction failed after {max_retries + 1} attempts: {last_error}"
                ) from last_error


class EvidenceFilter:
    """Tier 1 pre-filter: multi-dimensional relevance scoring before LLM grading.

    Scoring dimensions:
    - Target mention density (TF-style)
    - Mechanism keyword coverage (from profile & retrieval expansion)
    - Disease/indication keyword coverage
    - Source diversity bonus
    - Abstract length/quality proxy
    """

    # Generic, target-agnostic vocabulary only; no cell-type or
    # mechanism-specific terms live here. This is the floor that applies when
    # a target has no profile at all. Target-specific mechanism keywords are derived
    # dynamically from the target's YAML profile in `_get_mechanism_keywords`
    # below, so pre-filtering generalizes to any target and genuinely goes
    # away when that target's profile is removed, instead of silently falling
    # back to Kv1.3-shaped keywords hardcoded here.
    GENERIC_MECHANISM_KEYWORDS = {
        'inhibition', 'activation', 'signaling', 'signalling', 'pathway',
        'receptor', 'expression', 'regulation', 'binding', 'transcription',
        'proliferation', 'differentiation', 'apoptosis', 'inflammation',
        'immunity', 'immune', 'therapeutic', 'suppression', 'blockade',
        'disease', 'treatment', 'clinical',
    }

    @staticmethod
    def _get_mechanism_keywords(target: str) -> set[str]:
        """Mechanism keyword set for Tier-1 relevance scoring: generic base
        vocabulary + terms mined from this target's YAML profile (functional
        chain steps, cell types, mechanistic-bridge axes/cell types). Returns
        only the generic base set if the target has no profile."""
        from dp_indicator.core.target_knowledge import get_target_profile
        keywords = set(EvidenceFilter.GENERIC_MECHANISM_KEYWORDS)
        profile = get_target_profile(target)
        if not profile or profile.get("official_name") == "Unknown target":
            return keywords
        for step in profile.get("functional_chain", []):
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9+]{2,}", step.lower()):
                keywords.add(token)
        for cell_type in profile.get("cell_type_expression", {}).keys():
            keywords.add(cell_type.replace("_", " ").lower())
        for bridge in profile.get("mechanistic_bridges", []):
            axis = bridge.get("axis", "")
            if axis:
                keywords.add(axis.replace("_", " ").lower())
            for ct in bridge.get("cell_types", bridge.get("cell_types_involved", [])):
                keywords.add(ct.replace("_", " ").lower())
        return keywords

    @staticmethod
    def pre_filter(evidence_list: list[dict], target: str,
                   min_title_len: int = 10, min_relevance_score: float = 0.05) -> list[dict]:
        """Multi-dimensional pre-filter to produce graded relevance scores.

        Returns top-scoring candidates for Tier 2 (EvidenceGrader) grading.
        """
        filtered = []
        target_lower = target.lower()
        target_tokens = set(target_lower.replace('.', '').split())
        # Also match without the dot variant (kv13 vs kv1.3)
        target_variants = {target_lower, target_lower.replace('.', ''),
                           target_lower.replace('1.', '1'), target_lower.replace('.3', '3')}
        mechanism_keywords = EvidenceFilter._get_mechanism_keywords(target)

        for ev in evidence_list:
            title = ev.get("title", "")
            abstract = ev.get("abstract_snippet", "")

            # Tier 0: basic quality gate
            if len(title) < min_title_len:
                continue

            text = f"{title} {abstract}".lower()
            text_len = max(len(text), 1)

            # --- Dimension 1: Target mention density (TF-style) ---
            target_count = sum(text.count(v) for v in target_variants)
            target_density = min(target_count / 10.0, 1.0)  # cap at 10 mentions

            # --- Dimension 2: Mechanism keyword coverage ---
            mech_hits = sum(1 for kw in mechanism_keywords if kw in text)
            mech_score = min(mech_hits / 5.0, 1.0)  # cap at 5 keywords

            # --- Dimension 3: Abstract quality proxy ---
            abstract_len = len(abstract)
            quality_score = min(abstract_len / 500.0, 1.0)  # cap at 500 chars

            # --- Dimension 4: Source diversity bonus ---
            source_db = ev.get("source_db", "")
            source_bonus = 0.1 if source_db in {"pubmed", "europe_pmc"} else 0.0

            # --- Composite score ---
            relevance = (
                0.35 * target_density +
                0.35 * mech_score +
                0.20 * quality_score +
                0.10 * source_bonus
            )

            # --- Fallback: if target not mentioned at all, heavily penalize ---
            if target_count == 0 and not any(v in text for v in target_variants):
                relevance *= 0.3

            if relevance >= min_relevance_score:
                ev["_prefilter_score"] = round(relevance, 4)
                filtered.append(ev)

        # Sort by prefilter score, keep top 150 for Tier 2
        filtered.sort(key=lambda e: e.get("_prefilter_score", 0), reverse=True)
        return filtered[:150]


class EvidenceGrader:
    """Grade evidence quality separately from adversarial hypothesis review."""
    def __init__(self, llm: object, audit: AuditRecorder,
                 model: str = "glm-5.1", task: str = "grader"):
        self.llm = llm
        self.audit = audit
        self.model = model
        self.task = task

    async def run(self, evidence_pool: list[dict],
                  exclude: list[str] | None = None) -> list[dict]:
        graded = []
        # Batches of ten limit request overhead while bounding prompt size.
        batch_size = 10
        batches = [evidence_pool[i:i + batch_size] for i in range(0, len(evidence_pool), batch_size)]
        semaphore = asyncio.Semaphore(2)  # Avoid excess concurrent grading timeouts.
        async def _bounded_grade(i, batch):
            async with semaphore:
                print(f"  [heartbeat] Grading batch {i+1}/{len(batches)} ({len(batch)} items)...", flush=True)
                try:
                    # Allow long structured grading responses to complete.
                    return await asyncio.wait_for(self._grade_batch(i, batch, exclude), timeout=180)
                except asyncio.TimeoutError:
                    print(f"  ⚠️ Batch {i+1} timeout, using fallback grades", flush=True)
                    self.audit.record("Grader", "grade", "batch_timeout", {"batch_idx": i, "batch_size": len(batch)})
                    fallback = []
                    for ev in batch:
                        g = dict(ev)
                        g.setdefault("grade_score", 2)
                        g.setdefault("grade_rating", "⊕⊕○○")
                        g.setdefault("inclusion", True)
                        g.setdefault("relevance_to_target", "medium")
                        g.setdefault("evidence_type", ev.get("evidence_type", "literature"))
                        g["grade_source"] = "timeout_fallback"
                        fallback.append(g)
                    return fallback
        tasks = [_bounded_grade(i, batch) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                self.audit.record("Grader", "grade", "batch_error", {"error": str(batch_result)})
                continue
            for item in batch_result:
                graded.append(item)
        print(f"  [heartbeat] Grading complete: {len(graded)} items", flush=True)
        return graded

    async def _grade_batch(self, batch_idx: int, evidence_batch: list[dict],
                           exclude: list[str] | None = None) -> list[dict]:
        # Deep-copy before modification to avoid mutating the evidence pool.
        evidence_batch = [copy.deepcopy(ev) for ev in evidence_batch]

        sys_prompt = (
            "You are an evidence critic. Grade multiple biomedical evidence items using GRADE. "
            "Also classify each item's evidence_type and flag any interpretation errors. "
            "Return JSON array."
        )
        items_text = []
        for i, ev in enumerate(evidence_batch):
            items_text.append(f"""[Item {i}]
ID: {ev.get("evidence_id", "")}
Title: {ev.get("title", "")}
Abstract: {ev.get("abstract_snippet", "")[:300]}
Source: {ev.get("source_db", "")}
Preliminary type: {ev.get("evidence_type", "")}""")
        exclude_instruction = ""
        if exclude:
            exclude_instruction = f"\nEXCLUSION CONSTRAINT: The user has requested to exclude evidence related to: {', '.join(exclude)}. If an evidence item's title/abstract is primarily about one of these excluded areas, set inclusion=false for that item."
        user_prompt = f"""Grade the following {len(evidence_batch)} evidence items using GRADE framework.{exclude_instruction}
{chr(10).join(items_text)}
Return a JSON array with one object per item, in the same order. Each object must have:
- grade_score: int (1-4)
- grade_rating: str (e.g. ⊕⊕⊕⊕)
- inclusion: bool (true/false)
- relevance_to_target: str ("high"/"medium"/"low"/"unknown")
- evidence_type: str - classify into one of:
  "RCT_human" (randomized controlled trial in humans),
  "clinical_trial" (non-RCT human study),
  "cohort" (prospective/retrospective cohort),
  "case_control" (case-control study),
  "gwas" (genome-wide association),
  "expert_curation" (curated database annotation),
  "animal" (animal model study),
  "in_vitro" (cell line or biochemical assay),
  "database_association" (statistical association from database),
  "literature" (other published research),
  "preprint" (unpublished preprint),
  "review" (review article)
- interpretation_error: bool - true if the evidence's claims in title/abstract are overstated, misleading, or the study design does not match the claims (e.g., an in-vitro study claiming clinical efficacy). false otherwise.
- interpretation_note: str - brief note if interpretation_error is true, empty otherwise.
Example: [{{"grade_score": 3, "grade_rating": "⊕⊕⊕○", "inclusion": true, "relevance_to_target": "high", "evidence_type": "in_vitro", "interpretation_error": false, "interpretation_note": ""}}, ...]"""
        start = time.time()
        result, usage = await self.llm.structured([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ], max_tokens=4096, task=self.task)
        elapsed = time.time() - start
        self.audit.record("Critic", "grade", "batch_call", {
            "batch_idx": batch_idx, "batch_size": len(evidence_batch),
            "latency_ms": elapsed * 1000, "tokens": usage,
        })
        if isinstance(result, dict) and result.get("error") == "json_parse_failed":
            self.audit.record("Critic", "grade", "json_parse_failed",
                              {"raw": result.get("raw", "")[:200]})
            graded = []
            for ev in evidence_batch:
                g = dict(ev)
                g.setdefault("grade_score", 2)
                g.setdefault("grade_rating", "⊕⊕○○")
                g.setdefault("inclusion", True)
                g.setdefault("relevance_to_target", "medium")
                g.setdefault("evidence_type", ev.get("evidence_type", "literature"))
                g["grade_source"] = "default_fallback"
                graded.append(g)
            return graded
        graded = []
        results_list = result if isinstance(result, list) else result.get("results", [])
        if len(results_list) != len(evidence_batch):
            self.audit.record("Critic", "grade", "alignment_mismatch", {
                "batch_idx": batch_idx,
                "expected": len(evidence_batch),
                "received": len(results_list),
            })
        for i, ev in enumerate(evidence_batch):
            grade = {
                "grade_score": 2,
                "grade_rating": "⊕⊕○○",
                "inclusion": True,
                "relevance_to_target": "medium",
                "evidence_type": ev.get("evidence_type", "literature"),
                "interpretation_error": False,
                "interpretation_note": "",
                "grade_source": "default_fallback",
            }
            if i < len(results_list) and isinstance(results_list[i], dict):
                g = results_list[i]
                grade["grade_score"] = max(1, min(4, int(g.get("grade_score", 2))))
                grade["grade_rating"] = g.get("grade_rating", "⊕⊕○○")
                grade["inclusion"] = bool(g.get("inclusion", True))
                grade["relevance_to_target"] = g.get("relevance_to_target", "medium")
                grade["interpretation_error"] = bool(g.get("interpretation_error", False))
                grade["interpretation_note"] = g.get("interpretation_note", "")
                grade["grade_source"] = "llm_graded"
                from dp_indicator.schema import EvidenceType
                valid_types = {e.value for e in EvidenceType}
                et = g.get("evidence_type", ev.get("evidence_type", "literature"))
                grade["evidence_type"] = et if et in valid_types else "literature"
                rt = g.get("relevance_to_target", "medium")
                if rt not in ("high", "medium", "low", "unknown"):
                    rt = "medium"
                grade["relevance_to_target"] = rt
            # Merge grade fields into evidence (ev is already a deep copy)
            ev.update(grade)
            graded.append(ev)
        return graded


class KnowledgeSynthesizer:
    """Read the entire included evidence pool, not just a top-N slice, and
    distills it into a compact, citation-linked knowledge base of mechanistic facts.

    Rationale: ReasonerAgent's context can only hold ~5-10 raw evidence excerpts before
    prompt size/attention quality degrades, so any pure grade_score top-N cut is a lottery
    over which evidence the Reasoner ever "sees" — a paper that happens to be graded 2 in
    one run and 4 in another (same content, non-deterministic LLM grading) can silently
    swing which indications get proposed. This agent removes that bottleneck by mapping
    over 100% of the pool: every item gets read and reduced to a handful of short,
    disease-neutral mechanistic statements (what cell type / pathway / molecule is
    affected, and how), each still tagged with its source evidence_id. The resulting
    knowledge base (usually a few hundred short facts) is small enough to fit in full into
    the Reasoner's prompt, so the Reasoner reasons over the whole literature corpus instead
    of a randomly-survived slice of it.

    The extraction prompt deliberately never asks about, or names, any specific disease —
    it only asks "what biology does this evidence establish" — so the knowledge base is
    generic infrastructure, not something built to point toward any particular indication.
    """
    def __init__(self, llm: object, audit: AuditRecorder,
                 model: str = "deepseek-v3.2", task: str = "synthesizer",
                 batch_size: int = 8, max_facts_per_item: int = 3):
        self.llm = llm
        self.audit = audit
        self.model = model
        self.task = task
        self.batch_size = batch_size
        self.max_facts_per_item = max_facts_per_item

    async def run(self, evidence_pool: list[dict], target: str) -> list[dict]:
        items = [ev for ev in evidence_pool if ev.get("inclusion", True)]
        if not items:
            return []
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        semaphore = asyncio.Semaphore(3)

        async def _bounded_synthesize(i, batch):
            async with semaphore:
                print(f"  [heartbeat] KnowledgeSynthesizer: batch {i+1}/{len(batches)} "
                      f"({len(batch)} items)...", flush=True)
                try:
                    return await asyncio.wait_for(
                        self._synthesize_batch(i, batch, target), timeout=180
                    )
                except asyncio.TimeoutError:
                    print(f"  ⚠️ KnowledgeSynthesizer batch {i+1} timeout, skipping", flush=True)
                    self.audit.record("KnowledgeSynthesizer", "synthesize", "batch_timeout",
                                      {"batch_idx": i, "batch_size": len(batch)})
                    return []
                except Exception as e:
                    print(f"  ⚠️ KnowledgeSynthesizer batch {i+1} error: {type(e).__name__}: {e}, skipping", flush=True)
                    self.audit.record("KnowledgeSynthesizer", "synthesize", "batch_error",
                                      {"batch_idx": i, "error": str(e)})
                    return []

        tasks = [_bounded_synthesize(i, batch) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*tasks)
        valid_ids = {ev.get("evidence_id", "") for ev in evidence_pool}
        knowledge_base = []
        for facts in batch_results:
            for f in facts:
                if f.get("evidence_id") in valid_ids and f.get("fact"):
                    knowledge_base.append(f)
        print(f"  [heartbeat] KnowledgeSynthesizer complete: {len(knowledge_base)} facts "
              f"distilled from {len(items)} evidence items", flush=True)
        self.audit.record("KnowledgeSynthesizer", "synthesize", "complete",
                          {"n_facts": len(knowledge_base), "n_evidence": len(items)})
        return knowledge_base

    async def _synthesize_batch(self, batch_idx: int, batch: list[dict], target: str) -> list[dict]:
        sys_prompt = (
            "You are a biomedical knowledge extraction system. For each evidence item, extract "
            "the underlying MECHANISTIC biology it establishes — not disease conclusions. "
            "Return ONLY a valid JSON array."
        )
        items_text = []
        for i, ev in enumerate(batch):
            items_text.append(f"""[Item {i}]
ID: {ev.get("evidence_id", "")}
Title: {ev.get("title", "")}
Abstract: {ev.get("abstract_snippet", "")[:800]}""")
        user_prompt = f"""Target: {target}

For each of the {len(batch)} evidence items below, extract up to {self.max_facts_per_item} \
mechanistic facts (each <= 45 words, but use only as many words as the content actually needs — \
short is fine). A mechanistic fact describes WHAT biological entity/process is affected and HOW — \
e.g. which cell type, molecule, receptor, signaling pathway, or physiological process is involved.

For each fact, also capture (briefly, as separate fields so detail isn't lost to compression):
- system: the study system/model this was observed in, e.g. "human T cells", "mouse EAE model", \
"Jurkat cell line", "human PBMC". Leave empty if not stated.
- direction: one of "increase", "decrease", "no_change", "complex" — the direction of the effect \
described, from the target's perspective (e.g. inhibiting the target increases/decreases X).

Rules:
- Do NOT mention or infer any disease name UNLESS the evidence's title/abstract explicitly \
studied that disease as its outcome measure. Describing the biology (cell types, pathways, \
molecules) is the goal, not diagnosing which disease it might relate to.
- If an item is purely structural/pharmacological (e.g. binding affinity, channel kinetics) \
with no broader biological consequence described, you may return 0 facts for it.
- Stay faithful to what the abstract actually states; do not embellish or extrapolate beyond it.
- Prefer extracting distinct, non-redundant facts over restating the same point twice.

{chr(10).join(items_text)}

Return a JSON array. Each element:
{{"evidence_id": "<exact ID from above>", "fact": "<mechanistic statement>", "layer": "molecular|cellular|tissue|systemic|clinical", "system": "<study system or empty>", "direction": "increase|decrease|no_change|complex"}}
Omit items with nothing extractable. Do not invent evidence_ids not listed above."""
        result, usage = await self.llm.structured([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ], max_tokens=2048, task=self.task)
        self.audit.record("KnowledgeSynthesizer", "synthesize", "batch_call", {
            "batch_idx": batch_idx, "batch_size": len(batch), "tokens": usage,
        })
        if isinstance(result, dict) and result.get("error") == "json_parse_failed":
            self.audit.record("KnowledgeSynthesizer", "synthesize", "json_parse_failed",
                              {"raw": result.get("raw", "")[:200]})
            return []
        facts_list = result if isinstance(result, list) else result.get("facts", result.get("results", []))
        if not isinstance(facts_list, list):
            return []
        cleaned = []
        for f in facts_list:
            if isinstance(f, dict) and f.get("evidence_id") and f.get("fact"):
                cleaned.append({
                    "evidence_id": str(f["evidence_id"]),
                    "fact": str(f["fact"])[:400],
                    "layer": f.get("layer", "") if f.get("layer") in
                             ("molecular", "cellular", "tissue", "systemic", "clinical") else "",
                    "system": str(f.get("system", ""))[:60],
                    "direction": f.get("direction", "") if f.get("direction") in
                                 ("increase", "decrease", "no_change", "complex") else "",
                })
        return cleaned


GLOBAL_REASONING_SYSTEM_PROMPT = (
    "You are a senior immunologist and drug discovery scientist. Return ONLY valid JSON. "
    "Do not include markdown, prose, hidden reasoning, or commentary outside JSON."
)
# Full-corpus reasoning can produce roughly 6k-token structured responses, so this
# budget leaves headroom for elaborate multi-axis causal chains.
REASONER_MAX_TOKENS = 12000


class ReasonerAgent:
    def __init__(self, llm: object, audit: AuditRecorder,
                 model: str = "glm-5.1", task: str = "reasoner"):
        self.llm = llm
        self.audit = audit
        self.model = model
        self.task = task

    def _load_prompt(self) -> str:
        path = __import__('pathlib').Path(__file__).parent.parent / "prompts" / "reasoner.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    @staticmethod
    def _get_reasoning_guidance(target: str) -> list[str]:
        from dp_indicator.core.target_knowledge import get_reasoning_guidance
        return get_reasoning_guidance(target)

    @staticmethod
    def _get_intersection_guidance(target: str) -> str:
        from dp_indicator.core.target_knowledge import get_intersection_guidance
        return get_intersection_guidance(target)

    @staticmethod
    def _format_knowledge_base(knowledge_base: list[dict]) -> str:
        """Group facts by layer for readability; each line keeps its evidence_id
        so the model can cite it directly in causal_chain steps."""
        from collections import defaultdict
        by_layer = defaultdict(list)
        for f in knowledge_base:
            by_layer[f.get("layer") or "unclassified"].append(f)
        layer_order = ["molecular", "cellular", "tissue", "systemic", "clinical", "unclassified"]
        lines = []
        for layer in layer_order:
            facts = by_layer.get(layer)
            if not facts:
                continue
            lines.append(f"### {layer}")
            for f in facts:
                tags = []
                if f.get("system"):
                    tags.append(f.get("system"))
                if f.get("direction"):
                    tags.append(f.get("direction"))
                tag_str = f" ({', '.join(tags)})" if tags else ""
                lines.append(f"- [{f.get('evidence_id', '')}] {f.get('fact', '')}{tag_str}")
        return "\n".join(lines)

    @staticmethod
    def _get_supplemental_context(target: str) -> str:
        from dp_indicator.core.target_knowledge import get_target_profile
        profile = get_target_profile(target)
        ctx = profile.get("supplemental_target_disease_context", "")
        if ctx and ctx.strip():
            return f"Supplemental Target-Disease Background:\n{ctx.strip()}\n"
        return ""

    async def run(self, target: str, evidence_pool: list[dict],
                  direction: str = "target_to_indication",
                  focus: str = None, exclude: list[str] | str | None = None,
                  disease_background: dict = None,
                  knowledge_base: list[dict] | None = None) -> list[dict]:
        """Use single-step global reasoning with evidence and biological context.

        `knowledge_base` carries mechanistic facts
        distilled from the ENTIRE evidence pool, not just the raw top-N excerpts below.
        This is the primary signal for which mechanistic threads exist across the whole
        corpus; the raw excerpts remain for concrete quoting/grounding of whichever
        threads the model chooses to pursue.
        """
        from dp_indicator.core.target_knowledge import build_biological_context

        sorted_evidence = sorted(
            evidence_pool,
            key=known_disease_deprioritized_grade,
            reverse=True,
        )

        inclusion_count = sum(1 for e in evidence_pool if e.get("inclusion", True))
        if inclusion_count < 20:
            print(f"  ⚠️ 证据不足告警: 仅 {inclusion_count} 条有效证据(推荐≥20),假设置信度可能偏低")

        # Build a bounded evidence summary with source metadata.
        max_evidence_for_reasoning = 5
        # Reserve up to two slots for the best mechanism-bridge evidence
        # (see select_evidence_with_bridge_quota) so bridge_search_terms results aren't
        # crowded out by generic high-grade reviews in a pure global grade_score cut.
        selected_for_reasoning = select_evidence_with_bridge_quota(
            sorted_evidence, max_evidence_for_reasoning, min_bridge_slots=2
        )
        selected_ids = {id(e) for e in selected_for_reasoning}
        omitted_evidence = [e for e in sorted_evidence if id(e) not in selected_ids]
        truncated_evidence = []
        if omitted_evidence:
            n_omitted = len(omitted_evidence)
            truncated_evidence = [
                {"id": e.get("evidence_id", ""), "title": e.get("title", "")[:80], "grade": e.get("grade_score", 2)}
                for e in omitted_evidence
            ]
            n_bridge_selected = sum(1 for e in selected_for_reasoning if e.get("is_bridge_evidence"))
            print(f"  [heartbeat] ReasonerAgent: {n_omitted} low-grade evidence truncated for global reasoning "
                  f"(showing top {max_evidence_for_reasoning}, incl. {n_bridge_selected} bridge-reserved)", flush=True)

        evidence_summary = []
        for ev in selected_for_reasoning:
            entry = {
                "id": ev.get("evidence_id", ""),
                "source": ev.get("source_db", ""),
                "type": ev.get("evidence_type", ""),
                "grade": ev.get("grade_score", 2),
                "title": ev.get("title", "")[:120],
                "abstract": ev.get("abstract_snippet", "")[:1500],
            }
            if ev.get("is_bridge_evidence"):
                entry["note"] = "mechanism-bridge evidence: directly relevant to a target-specific mechanistic bridge axis (see Extended Mechanistic Bridges below); prefer citing this over generic evidence when it supports a bridge axis step"
            # Enrich with full-text deep reading if available
            fts = ev.get("full_text_summary")
            if fts and isinstance(fts, dict):
                kf = fts.get("key_findings", [])
                if kf:
                    entry["key_findings"] = kf[:3]
                if fts.get("experimental_model"):
                    entry["experimental_model"] = fts["experimental_model"]
                if fts.get("effect_size_note"):
                    entry["effect_size"] = fts["effect_size_note"]
            # Enrich with grader interpretation
            if ev.get("interpretation_note"):
                entry["interpretation_note"] = ev["interpretation_note"]
            if ev.get("relevance_to_target"):
                entry["relevance_to_target"] = ev["relevance_to_target"]
            # Include citation info if available
            sm = ev.get("source_metadata", {})
            if sm and sm.get("first_author"):
                entry["citation"] = f"{sm.get('first_author', '')} et al., {sm.get('journal_short', sm.get('journal', ''))} ({sm.get('year', '')})"
            evidence_summary.append(entry)

        bio_context = build_biological_context(target)

        focus_text = ""
        if focus:
            if isinstance(focus, list):
                focus_text = f"\nFocus areas: {', '.join(focus)}"
            else:
                focus_text = f"\nFocus: {focus}"

        exclude_text = ""
        if exclude:
            if isinstance(exclude, list):
                exclude_text = f"\nExclude areas: {', '.join(exclude)}"
            else:
                exclude_text = f"\nExclude: {exclude}"

        reasoning_guidance = self._get_reasoning_guidance(target)
        reasoning_guidance_text = ""
        if reasoning_guidance:
            # Keep up to four guidance bullets, including subtype-specific guidance.
            reasoning_guidance_text = "\n".join(f"- {g}" for g in reasoning_guidance[:4])

        intersection_guidance = self._get_intersection_guidance(target)

        supplemental_context = self._get_supplemental_context(target)

        # Inject disease background if available
        disease_bg_text = ""
        if disease_background:
            bg_parts = []
            for disease, info in disease_background.items():
                bg_parts.append(
                    f"{disease}: GWAS={info.get('gwas_count', 0)}, "
                    f"ClinicalTrials={info.get('clinical_trials_count', 0)}"
                )
                samples = info.get("gwas_samples", []) + info.get("ct_samples", [])
                if samples:
                    bg_parts.append(f"  samples: {'; '.join(samples[:4])}")
            disease_bg_text = "\n## Disease Background\n" + "\n".join(bg_parts)

        # Load prompt template
        prompt_template = self._load_prompt()
        prompt = prompt_template.replace("{{target}}", target)
        prompt = prompt.replace("{{n}}", str(len(evidence_summary)))
        prompt = prompt.replace("{{evidence_json}}", json.dumps(evidence_summary, ensure_ascii=False, indent=2))
        prompt = prompt.replace("{{bio_context}}", bio_context)
        prompt = prompt.replace("{{supplemental_context}}", supplemental_context)
        prompt = prompt.replace("{{focus_text}}", focus_text)
        prompt = prompt.replace("{{exclude_text}}", exclude_text)
        if disease_bg_text:
            prompt += disease_bg_text

        # Include the full-corpus knowledge base, covering every
        # evidence item that was retrieved and graded-in, not just the excerpts above.
        if knowledge_base:
            kb_text = self._format_knowledge_base(knowledge_base)
            prompt += (
                "\n\n## Full-Corpus Mechanistic Knowledge Base\n"
                f"The following {len(knowledge_base)} facts were distilled by reading EVERY "
                "retrieved evidence item for this target (not only the excerpts shown above). "
                "Treat this as your primary map of what the full literature corpus actually "
                "establishes about the target's biology across molecular/cellular/tissue/systemic "
                "layers. Look for mechanistic threads that recur across multiple independent facts "
                "here, especially threads with no corresponding entry under 'Known/Established "
                "Indications' — those are candidate novel indications. You MAY cite any evidence_id "
                "referenced below in your causal_chain steps, even if that item's full excerpt is not "
                "shown above.\n" + kb_text
            )

        # Append reasoning guidance and intersection guidance
        prompt += f"\n\n## Reasoning Guidance\n{reasoning_guidance_text}\n\n{intersection_guidance}"

        print(f"  [heartbeat] Global reasoning with {len(evidence_summary)} evidence items...", flush=True)
        t0 = time.time()
        try:
            result, _ = await asyncio.wait_for(
                self.llm.structured([
                    {"role": "system", "content": GLOBAL_REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ], max_tokens=REASONER_MAX_TOKENS, task=self.task),
                timeout=900,
            )
            elapsed = time.time() - t0
            print(f"  [heartbeat] Global reasoning done in {elapsed:.1f}s", flush=True)
        except asyncio.TimeoutError:
            print("  ⚠️ Global reasoning timeout after 900s, returning empty", flush=True)
            return []
        except Exception as e:
            # Network/HTTP errors from the LLM client (e.g. auth failure, connection reset,
            # non-retryable status codes) must not crash the whole pipeline. Degrade gracefully
            # by retrying once, then giving up and returning an empty hypothesis list.
            print(f"  ⚠️ Global reasoning request failed: {type(e).__name__}: {e}", flush=True)
            try:
                print("  [heartbeat] Retrying global reasoning once after error...", flush=True)
                result, _ = await asyncio.wait_for(
                    self.llm.structured([
                        {"role": "system", "content": GLOBAL_REASONING_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ], max_tokens=REASONER_MAX_TOKENS, task=self.task),
                    timeout=900,
                )
                elapsed = time.time() - t0
                print(f"  [heartbeat] Global reasoning done in {elapsed:.1f}s (after retry)", flush=True)
            except Exception as e2:
                print(f"  ⚠️ Global reasoning retry also failed: {type(e2).__name__}: {e2}, returning empty", flush=True)
                return []

        if isinstance(result, dict) and result.get("error") == "json_parse_failed":
            raw = result.get('raw', '')
            from pathlib import Path
            Path("debug_reasoner_raw.txt").write_text(raw, encoding="utf-8")
            print(f"  ⚠️ Global reasoning JSON parse failed, raw length={len(raw)}; saved debug_reasoner_raw.txt", flush=True)
            # Attempt one compact retry whenever structured output cannot be parsed.
            # max_evidence_for_reasoning=5 — so a truncated/malformed response used to just
            # give up immediately). The retry drops the knowledge base and raw excerpts to
            # the smallest useful set and explicitly asks for a smaller output shape.
            retry_pool = selected_for_reasoning if len(selected_for_reasoning) <= 30 else sorted_evidence[:30]
            print(f"  [heartbeat] Retrying global reasoning compactly with {len(retry_pool)} evidence items...", flush=True)
            return await self._retry_global_reasoning(target, retry_pool, bio_context, focus_text, exclude_text, reasoning_guidance_text, intersection_guidance)

        hypotheses = result.get("hypotheses", []) if isinstance(result, dict) else (result if isinstance(result, list) else [])
        # Drop malformed list items while retaining valid hypothesis objects.
        # rather than crashing the whole run on `hyp.get(...)` a few lines below.
        n_before = len(hypotheses)
        hypotheses = [h for h in hypotheses if isinstance(h, dict)]
        if len(hypotheses) < n_before:
            print(f"  ⚠️ Dropped {n_before - len(hypotheses)} malformed (non-object) hypothesis entries", flush=True)
        if not hypotheses:
            print(f"  ⚠️ No hypotheses generated", flush=True)
            return []

        print(f"  [heartbeat] Generated {len(hypotheses)} hypotheses", flush=True)

        # Normalize and validate
        valid_ids = {e.get("evidence_id", "") for e in evidence_pool}
        for hyp in hypotheses:
            if not hyp.get('indication') and hyp.get('indication_name'):
                hyp['indication'] = hyp['indication_name']
            if not hyp.get('statement') and hyp.get('one_sentence_statement'):
                hyp['statement'] = hyp['one_sentence_statement']

            # Normalize the supported causal-chain structures.
            chain = hyp.get("causal_chain", {})
            if isinstance(chain, list):
                # Backward compatibility: old list format → wrap in single axis
                hyp["causal_chain"] = {
                    "mechanism_axes": [
                        {
                            "axis_name": "primary",
                            "steps": [
                                {
                                    "layer": item.get("layer", item.get("level", "unknown")),
                                    "mechanism": item.get("mechanism", item.get("description", "")),
                                    "status": item.get("status", "inferred"),
                                    "evidence_ids": item.get("evidence_ids", []),
                                    "source_text": item.get("source_text", ""),
                                }
                                for item in chain
                            ]
                        }
                    ],
                    "cross_talk": []
                }
            elif isinstance(chain, dict):
                if "mechanism_axes" not in chain:
                    # Old dict format (L1, L2, L3, L4, L5 keys)
                    steps = []
                    for level in ["L1", "L2", "L3", "L4", "L5"]:
                        if level in chain:
                            entry = chain[level]
                            steps.append({
                                "layer": level,
                                "mechanism": entry.get("description", entry.get("mechanism", "")),
                                "status": entry.get("status", "inferred"),
                                "evidence_ids": entry.get("evidence_ids", []),
                                "source_text": entry.get("source_text", ""),
                            })
                    hyp["causal_chain"] = {
                        "mechanism_axes": [{"axis_name": "primary", "steps": steps}],
                        "cross_talk": []
                    }

            # Validate evidence IDs and downgrade unsupported claims
            chain = hyp.get("causal_chain", {})
            for axis in chain.get("mechanism_axes", []):
                for step in axis.get("steps", []):
                    claimed = step.get("evidence_ids", [])
                    verified = [cid for cid in claimed if cid in valid_ids]
                    step["evidence_ids"] = verified
                    if step.get("status") == "supported" and not verified:
                        step["status"] = "inferred"

        # Attach source metadata to each step
        hypotheses = await self._attach_sources(hypotheses, evidence_pool)

        # Record items excluded from the raw excerpt slice. When a knowledge base is
        # supplied, these items still contribute distilled
        # mechanistic facts to the Reasoner via KnowledgeSynthesizer, so "truncated" no longer
        # means "unseen by the Reasoner in any form".
        if truncated_evidence:
            kb_note = (
                f"; {len(knowledge_base)} of them (and all other included evidence) still "
                "contributed distilled facts via the Full-Corpus Mechanistic Knowledge Base"
                if knowledge_base else ""
            )
            for hyp in hypotheses:
                hyp["_truncated_evidence"] = {
                    "count": len(truncated_evidence),
                    "items": truncated_evidence[:20],  # cap for report size
                    "reason": (
                        f"Only top {max_evidence_for_reasoning} evidence items were shown to "
                        f"ReasonerAgent as raw excerpts{kb_note}"
                    )
                }

        hypotheses = await self._link_hypothesis_to_evidence(
            hypotheses, evidence_pool, target
        )
        self.audit.record("Reasoner", "hypothesize", "complete",
                          {"n_hypotheses": len(hypotheses)})
        return hypotheses

    async def _retry_global_reasoning(self, target: str, raw_evidence: list[dict],
                                       bio_context: str, focus_text: str,
                                       exclude_text: str, reasoning_guidance_text: str,
                                       intersection_guidance: str) -> list[dict]:
        """Retry global reasoning with a smaller, compact evidence slice and an explicit
        request to keep the output shape small — used when the first attempt's JSON was
        truncated/malformed (e.g. hit the output token budget)."""
        # Use the same compact per-item summary as the main call to keep retry prompts
        # bounded.
        evidence_summary = [
            {
                "id": ev.get("evidence_id", ""),
                "source": ev.get("source_db", ""),
                "grade": ev.get("grade_score", 2),
                "title": ev.get("title", "")[:120],
                "abstract": ev.get("abstract_snippet", "")[:400],
            }
            for ev in raw_evidence
        ]
        supplemental_context = self._get_supplemental_context(target)
        prompt_template = self._load_prompt()
        prompt = prompt_template.replace("{{target}}", target)
        prompt = prompt.replace("{{n}}", str(len(evidence_summary)))
        prompt = prompt.replace("{{evidence_json}}", json.dumps(evidence_summary, ensure_ascii=False, indent=2))
        prompt = prompt.replace("{{bio_context}}", bio_context)
        prompt = prompt.replace("{{supplemental_context}}", supplemental_context)
        prompt = prompt.replace("{{focus_text}}", focus_text)
        prompt = prompt.replace("{{exclude_text}}", exclude_text)
        prompt += f"\n\n## Reasoning Guidance\n{reasoning_guidance_text}\n\n{intersection_guidance}"
        prompt += (
            "\n\n## Output Size Constraint\nYour previous response was too long and got cut "
            "off. This time, limit each hypothesis to AT MOST 1 mechanism_axis (no cross_talk), "
            "and keep every mechanism string under 25 words, to guarantee the full JSON fits."
        )
        try:
            result, _ = await asyncio.wait_for(
                self.llm.structured([
                    {"role": "system", "content": GLOBAL_REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ], max_tokens=REASONER_MAX_TOKENS, task=self.task),
                timeout=900,
            )
            if isinstance(result, dict) and result.get("error") == "json_parse_failed":
                print(f"  ⚠️ Retry also failed to parse JSON", flush=True)
                return []
            hypotheses = result.get("hypotheses", []) if isinstance(result, dict) else []
            hypotheses = [h for h in hypotheses if isinstance(h, dict)]
            print(f"  [heartbeat] Retry generated {len(hypotheses)} hypotheses", flush=True)
            return hypotheses
        except asyncio.TimeoutError:
            print("  ⚠️ Retry global reasoning timeout", flush=True)
            return []
        except Exception as e:
            print(f"  ⚠️ Retry global reasoning error: {e}", flush=True)
            return []

    async def _attach_sources(self, hypotheses: list[dict],
                               evidence_pool: list[dict]) -> list[dict]:
        """Attach source metadata to each causal-chain step."""
        evidence_map = {e.get("evidence_id", ""): e for e in evidence_pool}

        for hyp in hypotheses:
            chain = hyp.get("causal_chain", {})
            for axis in chain.get("mechanism_axes", []):
                for step in axis.get("steps", []):
                    eids = step.get("evidence_ids", [])
                    step["sources"] = []
                    for eid in eids:
                        ev = evidence_map.get(eid)
                        if ev and ev.get("source_metadata"):
                            sm = ev["source_metadata"]
                            source_entry = {
                                "evidence_id": eid,
                                "source_text": step.get("source_text", ""),
                            }
                            # Copy all available metadata fields
                            for key in ["authors", "first_author", "year", "journal",
                                        "journal_short", "volume", "issue", "pages",
                                        "doi", "pmid", "pmcid", "database_name",
                                        "record_id", "record_url", "confidence_note"]:
                                if key in sm:
                                    source_entry[key] = sm[key]
                            # Add key_finding if full_text_summary exists
                            if ev.get("key_findings"):
                                source_entry["key_finding"] = ev["key_findings"][0]
                            step["sources"].append(source_entry)
        return hypotheses

    async def _link_hypothesis_to_evidence(self, hypotheses: list[dict],
                                           evidence_pool: list[dict],
                                           target: str) -> list[dict]:
        for hyp in hypotheses:
            indication = hyp.get("indication", hyp.get("indication_name", ""))
            chain = hyp.get("causal_chain", {})
            # Extract step dictionaries from every supported causal-chain shape.
            links = []
            if isinstance(chain, dict) and "mechanism_axes" in chain:
                for axis in chain.get("mechanism_axes", []):
                    links.extend(axis.get("steps", []))
            elif isinstance(chain, dict):
                links = list(chain.values())
            elif isinstance(chain, list):
                links = chain

            all_claimed_ids = set()
            for link in links:
                if isinstance(link, dict):
                    all_claimed_ids.update(link.get("evidence_ids", []))

            # Also search for target/indication mentions in evidence
            target_lower = target.lower()
            indication_lower = indication.lower()
            for ev in evidence_pool:
                text = f"{ev.get('title', '')} {ev.get('abstract_snippet', '')}".lower()
                ev_id = ev.get("evidence_id", "")
                if target_lower in text or indication_lower in text:
                    all_claimed_ids.add(ev_id)

            hyp["evidence_ids"] = list(all_claimed_ids)[:25]
            hyp["n_evidence"] = len(hyp["evidence_ids"])
            hyp["n_missing_links"] = sum(
                1 for link in links
                if isinstance(link, dict) and link.get("status") == "hypothesized"
            )
            hyp["target"] = target
        return hypotheses


class EvidenceMapper:
    """Map evidence to hypotheses with explicit reasoning."""
    def __init__(self, llm: object, audit: AuditRecorder,
                 model: str = "glm-5.1", task: str = "critic"):
        self.llm = llm
        self.audit = audit
        self.model = model
        self.task = task

    async def map_hypothesis(self, hypothesis: dict, evidence_pool: list[dict],
                             target: str) -> dict:
        """Map all evidence to a single hypothesis with explicit reasoning."""
        indication = hypothesis.get("indication", "")
        statement = hypothesis.get("statement", "")
        chain = hypothesis.get("causal_chain", {})

        # Build evidence text with citations
        # Prioritize high-grade, high-relevance items when the pool is large.
        max_evidence_for_mapping = 40
        sorted_ev = sorted(
            evidence_pool,
            key=lambda e: (known_disease_deprioritized_grade(e), e.get("relevance_to_target") == "high"),
            reverse=True,
        )
        # Reserve bridge-evidence slots consistently with ReasonerAgent.
        selected_ev = select_evidence_with_bridge_quota(
            sorted_ev, max_evidence_for_mapping, min_bridge_slots=4
        )
        selected_ev_ids = {id(e) for e in selected_ev}
        omitted_ev_mapping = [e for e in sorted_ev if id(e) not in selected_ev_ids]
        n_omitted_mapping = 0
        truncated_evidence_mapping = []
        if omitted_ev_mapping:
            n_omitted_mapping = len(omitted_ev_mapping)
            truncated_evidence_mapping = [
                {"id": e.get("evidence_id", ""), "title": e.get("title", "")[:80], "grade": e.get("grade_score", 2)}
                for e in omitted_ev_mapping
            ]
            print(f"  [heartbeat] EvidenceMapper: {n_omitted_mapping} low-grade evidence omitted for mapping (showing top {max_evidence_for_mapping})", flush=True)

        evidence_texts = []
        for ev in selected_ev:
            sm = ev.get("source_metadata", {})
            citation = ""
            if sm.get("first_author"):
                citation = f"[{sm.get('first_author', '')} et al., {sm.get('year', '')}]"
            evidence_texts.append({
                "id": ev.get("evidence_id", ""),
                "citation": citation,
                "title": ev.get("title", "")[:80],
                "abstract": ev.get("abstract_snippet", "")[:800],
                "type": ev.get("evidence_type", ""),
                "grade": ev.get("grade_score", 2),
            })

        # Summarize causal chain steps
        chain_summary = []
        for axis in chain.get("mechanism_axes", []):
            for step in axis.get("steps", []):
                chain_summary.append({
                    "layer": step.get("layer", ""),
                    "mechanism": step.get("mechanism", ""),
                    "status": step.get("status", ""),
                })

        prompt = f"""## Task
Map the following evidence items to the hypothesis, with explicit reasoning for each judgment.

## Target: {target}
## Indication: {indication}
## Hypothesis: {statement}

## Causal Chain Steps
{json.dumps(chain_summary, ensure_ascii=False, indent=2)}

## Evidence Pool ({len(evidence_texts)} items)
{json.dumps(evidence_texts, ensure_ascii=False, indent=2)}

## Instructions
For each evidence item, classify its relationship to the hypothesis:
- "direct_support": Directly supports a causal chain step
- "indirect_support": Supports background/context but not direct causality
- "contradicting": Findings conflict with the hypothesis
- "unrelated": No clear connection

For each "direct_support" item, specify which causal chain step it supports.
For each "contradicting" item, explain the contradiction.

**Tissue/Organ Context Check**
- If the evidence study was conducted in a different tissue/organ system than the hypothesis (e.g., CNS microglia vs renal macrophages), you MUST:
  1. Set the relationship to "indirect_support" (not "direct_support")
  2. Note the tissue mismatch in the rationale, e.g., "Study conducted in CNS microglia; extrapolation to renal macrophages is speculative"
  3. Only use "direct_support" if the evidence comes from the same tissue/organ system as the hypothesis

Return JSON:
{{
  "positive_evidence": [{{"id": "...", "step": "L1", "rationale": "..."}}],
  "indirect_evidence": [{{"id": "...", "rationale": "..."}}],
  "contradicting_evidence": [{{"id": "...", "rationale": "..."}}],
  "unrelated": ["id1", "id2"],
  "overall_assessment": "Brief assessment of evidence strength for this hypothesis"
}}"""

        try:
            result, _ = await asyncio.wait_for(
                self.llm.structured([
                    {"role": "system", "content": "You are a careful evidence analyst. Map evidence to hypotheses with explicit reasoning. Return JSON."},
                    {"role": "user", "content": prompt},
                ], max_tokens=4096, task=self.task),  # Accommodate per-item rationale
                # text for up to 40 evidence items.
                # mid-JSON on well-supported hypotheses, causing a parse failure that
                # silently zeroed out that hypothesis's entire evidence_mapping (observed
                # in an Option E run). The retry below is a safety net for cases that still
                # don't fit even at this larger budget.
                timeout=300,
            )
            if isinstance(result, dict) and result.get("error") == "json_parse_failed":
                print(f"  ⚠️ Evidence mapping JSON parse failed for {indication}, retrying with a compact prompt...", flush=True)
                result = await self._retry_compact_mapping(selected_ev, chain_summary, target, indication)
            elif isinstance(result, dict):
                # Attach the evidence-truncation record.
                if truncated_evidence_mapping:
                    result["_truncated_evidence"] = {
                        "count": len(truncated_evidence_mapping),
                        "items": truncated_evidence_mapping[:20],
                        "reason": f"Only top {max_evidence_for_mapping} evidence items were passed to EvidenceMapper"
                    }
            return result if isinstance(result, dict) else {}
        except asyncio.TimeoutError:
            print(f"  ⚠️ Evidence mapping timeout for {indication}", flush=True)
            return {"_source": "timeout", "positive_evidence": [], "contradicting_evidence": []}
        except Exception as e:
            print(f"  ⚠️ Evidence mapping failed for {indication}: {e}", flush=True)
            return {"_source": "error", "positive_evidence": [], "contradicting_evidence": []}

    async def _retry_compact_mapping(self, selected_ev: list[dict], chain_summary: list[dict],
                                      target: str, indication: str) -> dict:
        """Retry when the main mapping call returns truncated or malformed JSON.
        Cuts the evidence slice in half and drops long fields (rationale -> id-only lists,
        no abstracts) so the response has a much smaller, harder-to-truncate shape. This
        trades away per-citation rationale text for guaranteeing the hypothesis still gets
        SOME evidence_mapping instead of silently degrading to an empty one."""
        top_ev = selected_ev[: max(1, len(selected_ev) // 2)]
        evidence_texts = [
            {"id": ev.get("evidence_id", ""), "title": ev.get("title", "")[:60]}
            for ev in top_ev
        ]
        prompt = f"""## Task (compact retry — previous attempt's JSON was too long/malformed)
Classify each evidence item's relationship to the hypothesis. NO rationale text — IDs only.

## Target: {target}
## Indication: {indication}
## Causal Chain Steps
{json.dumps(chain_summary, ensure_ascii=False)}

## Evidence Pool ({len(evidence_texts)} items)
{json.dumps(evidence_texts, ensure_ascii=False)}

Return ONLY this compact JSON (id lists only, no rationale, no extra fields):
{{"positive_evidence": ["id1", "id2"], "indirect_evidence": ["id3"], "contradicting_evidence": [], "unrelated": ["id4"]}}"""
        try:
            result, _ = await asyncio.wait_for(
                self.llm.structured([
                    {"role": "system", "content": "You are a careful evidence analyst. Return only compact JSON, no rationale text."},
                    {"role": "user", "content": prompt},
                ], max_tokens=2048, task=self.task),
                timeout=180,
            )
            if isinstance(result, dict) and result.get("error") == "json_parse_failed":
                print(f"  ⚠️ Evidence mapping retry ALSO failed to parse for {indication} — "
                      f"this hypothesis's evidence_mapping will be empty and its evidence "
                      f"links may be under-reported in the report", flush=True)
                return {"_source": "parse_failed_after_retry", "positive_evidence": [], "contradicting_evidence": []}
            if not isinstance(result, dict):
                return {"_source": "parse_failed_after_retry", "positive_evidence": [], "contradicting_evidence": []}
            # Normalize the compact id-only shape into the standard positive_evidence
            # shape ({"id": ..., "rationale": ...}) so downstream consumers don't need
            # to special-case this fallback's output format.
            result["_source"] = "compact_retry"
            for key in ("positive_evidence", "indirect_evidence", "contradicting_evidence"):
                items = result.get(key, [])
                if items and isinstance(items[0], str):
                    result[key] = [{"id": i, "rationale": "(compact retry: no rationale captured)"} for i in items]
            return result
        except Exception as e:
            print(f"  ⚠️ Evidence mapping retry error for {indication}: {e}", flush=True)
            return {"_source": "retry_error", "positive_evidence": [], "contradicting_evidence": []}


class HypothesisCritic:
    """Deep-reasoning critic with evidence-mapping integration."""
    def __init__(self, llm: object, audit: AuditRecorder,
                 model: str = "glm-5.1", task: str = "critic"):
        self.llm = llm
        self.audit = audit
        self.model = model
        self.task = task
        self.evidence_mapper = EvidenceMapper(llm, audit, model, task)

    async def map_evidence(
        self,
        hypotheses: list[dict],
        evidence_pool: list[dict],
        target: str = "",
    ) -> list[dict]:
        semaphore = asyncio.Semaphore(3)

        async def _map_one(hypothesis: dict) -> None:
            async with semaphore:
                indication = hypothesis.get("indication", "")
                try:
                    hypothesis["evidence_mapping"] = (
                        await self.evidence_mapper.map_hypothesis(
                            hypothesis,
                            evidence_pool,
                            target,
                        )
                    )
                except Exception as exc:
                    print(
                        f"  ⚠️ Evidence mapping failed for "
                        f"{indication}: {exc}",
                        flush=True,
                    )
                    hypothesis["evidence_mapping"] = {"_source": "error"}

        await asyncio.gather(*[_map_one(item) for item in hypotheses])
        return hypotheses

    async def review_hypotheses(
        self,
        hypotheses: list[dict],
        evidence_pool: list[dict],
        target: str = "",
    ) -> list[dict]:
        semaphore = asyncio.Semaphore(3)

        async def _review_one(hypothesis: dict, index: int) -> None:
            indication = hypothesis.get("indication", "")
            print(
                f"  [heartbeat] Critic starting {index + 1}/"
                f"{len(hypotheses)}: {indication}...",
                flush=True,
            )
            async with semaphore:
                started = time.time()
                try:
                    hypothesis["critic_review"] = await self._review(
                        hypothesis,
                        evidence_pool,
                        target,
                    )
                except Exception as exc:
                    print(
                        f"  ⚠️ Critic review failed for "
                        f"{indication}: {exc}",
                        flush=True,
                    )
                    hypothesis["critic_review"] = self._review_fallback()
                elapsed = time.time() - started
                print(
                    f"  [heartbeat] Critic done {index + 1}/"
                    f"{len(hypotheses)}: {indication} in "
                    f"{elapsed:.1f}s",
                    flush=True,
                )

        await asyncio.gather(*[
            _review_one(hypothesis, index)
            for index, hypothesis in enumerate(hypotheses)
        ])
        return hypotheses

    async def run(
        self,
        hypotheses: list[dict],
        evidence_pool: list[dict],
        target: str = "",
    ) -> list[dict]:
        mapped = await self.map_evidence(
            hypotheses,
            evidence_pool,
            target,
        )
        return await self.review_hypotheses(
            mapped,
            evidence_pool,
            target,
        )

    def _review_fallback(self) -> dict:
        """Return a safe fallback when review fails."""
        return {
            "logical_completeness": {"reasoning": "Review failed", "conclusion": "Unable to assess"},
            "evidence_strength_review": {"reasoning": "Review failed", "conclusion": "Unable to assess"},
            "alternative_explanations": [],
            "causal_direction_check": "Unable to assess",
            "fatal_weakness": {"reasoning": "Review failed", "weakness": "Unknown"},
            "suggested_fix": "Please manually review this hypothesis",
            "confidence": 0.0,
            "_source": "timeout_fallback",
        }

    @staticmethod
    def normalize_review(review: dict) -> dict:
        normalized = copy.deepcopy(review)
        fatal = normalized.get("fatal_weakness")
        if isinstance(fatal, str):
            normalized["fatal_weakness"] = {"weakness": fatal}
        elif isinstance(fatal, dict) and not fatal.get("weakness"):
            weakness = next(
                (
                    fatal.get(key)
                    for key in (
                        "conclusion",
                        "assessment",
                        "reasoning",
                        "description",
                    )
                    if fatal.get(key)
                ),
                "",
            )
            if weakness:
                fatal["weakness"] = str(weakness)

        suggested = normalized.get("suggested_fix")
        if isinstance(suggested, dict):
            normalized["suggested_fix"] = str(
                next(
                    (
                        suggested.get(key)
                        for key in (
                            "recommendation",
                            "fix",
                            "conclusion",
                            "description",
                            "reasoning",
                        )
                        if suggested.get(key)
                    ),
                    "",
                )
            )
        return normalized

    async def _review(self, hypothesis: dict, evidence_pool: list[dict],
                      target: str) -> dict:
        """Review a hypothesis across six dimensions."""
        indication = hypothesis.get("indication", "")
        statement = hypothesis.get("statement", "")
        pred = hypothesis.get("falsifiable_prediction", "")
        chain = hypothesis.get("causal_chain", {})
        eh_mapping = hypothesis.get("evidence_mapping", {})

        # Build chain text for prompt
        chain_text = ""
        for i, axis in enumerate(chain.get("mechanism_axes", [])):
            chain_text += f"\nAxis {i+1}: {axis.get('axis_name', 'unknown')}\n"
            for step in axis.get("steps", []):
                chain_text += f"  - {step.get('layer', '?')}: {step.get('mechanism', '')} [{step.get('status', '?')}]\n"

        # Evidence mapping summary
        pos = eh_mapping.get("positive_evidence", [])[:8]
        neg = eh_mapping.get("contradicting_evidence", [])[:5]

        sys_prompt = (
            "You are a rigorous peer reviewer with expertise in immunology and drug discovery. "
            "Your job is to identify the most serious logical flaws. Before giving your final assessment, "
            "explicitly play devil's advocate: what is the strongest argument AGAINST this hypothesis? "
            "Then balance this against the supporting evidence."
        )

        user_prompt = f"""## Hypothesis
Target: {target}
Indication: {indication}
Statement: {statement[:400]}
Prediction: {pred[:300]}

## Causal Chain
{chain_text}

## Evidence Mapping Summary
Directly supporting evidence ({len(pos)} items):
{json.dumps(pos, ensure_ascii=False, indent=2)[:1500]}

Contradicting evidence ({len(neg)} items):
{json.dumps(neg, ensure_ascii=False, indent=2)[:800]}

## Review Tasks
Answer each question with reasoning + conclusion:

1. **Logical Completeness**: Does the causal chain have missing steps? Is every step from molecule to disease supported?
2. **Evidence Strength Review**:
   - "supported" steps: Does the evidence DIRECTLY support the causal claim, or just correlate?
   - "inferred" steps: Is the extrapolation reasonable? Is there stronger direct evidence being ignored?
   - "hypothesized" steps: Are there published alternative mechanisms?
3. **Alternative Explanations**: What other mechanisms could explain {target} ↔ {indication} association?
4. **Causal Direction**: Did we assume A→B? Could it be B→A or C→A&B?
5. **Fatal Weakness**: The single strongest reason this hypothesis might be wrong.
6. **Suggested Fix**: One concrete improvement to address the fatal weakness.

Return JSON with reasoning + conclusion for each dimension."""

        try:
            result, _ = await asyncio.wait_for(
                self.llm.structured([
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ], max_tokens=2048, task=self.task),
                timeout=180,
            )
            if isinstance(result, dict) and result.get("error") == "json_parse_failed":
                return self._review_fallback()
            if isinstance(result, dict) and "fatal_weakness" in result:
                normalized = self.normalize_review(result)
                fatal = normalized.get("fatal_weakness", {})
                if (
                    isinstance(fatal, dict)
                    and fatal.get("weakness")
                    and normalized.get("suggested_fix")
                ):
                    return normalized
        except asyncio.TimeoutError:
            print(f"  ⚠️ Review timeout for {indication}", flush=True)
        except Exception:
            pass
        return self._review_fallback()


class SemanticScorer:
    """LLM-based semantic scoring with retry handling."""
    def __init__(self, llm: object, audit: AuditRecorder,
                 model: str = "glm-5.1", task: str = "critic"):
        self.llm = llm
        self.audit = audit
        self.model = model
        self.task = task
        self.max_retries = 2

    async def score(self, hypothesis: dict, evidence_pool: list[dict],
                    target: str = "") -> dict:
        """Score a hypothesis with LLM reasoning. Retry on failure."""
        indication = hypothesis.get("indication", "")
        statement = hypothesis.get("statement", "")
        chain = hypothesis.get("causal_chain", {})
        critic_review = hypothesis.get("critic_review", {})
        eh_mapping = hypothesis.get("evidence_mapping", {})

        # Count supported/inferred/hypothesized steps
        step_counts = {"supported": 0, "inferred": 0, "hypothesized": 0}
        for axis in chain.get("mechanism_axes", []):
            for step in axis.get("steps", []):
                status = step.get("status", "inferred")
                step_counts[status] = step_counts.get(status, 0) + 1

        pos_count = len(eh_mapping.get("positive_evidence", []))
        neg_count = len(eh_mapping.get("contradicting_evidence", []))

        prompt = f"""## Task
Score the following hypothesis across 4 dimensions (0.0-1.0). Provide reasoning for each score.

## Target: {target}
## Indication: {indication}
## Hypothesis: {statement[:300]}

## Causal Chain Summary
- Supported steps: {step_counts['supported']}
- Inferred steps: {step_counts['inferred']}
- Hypothesized steps: {step_counts['hypothesized']}

## Evidence Summary
- Directly supporting evidence: {pos_count}
- Contradicting evidence: {neg_count}

## Critic Review Summary
- Fatal weakness: {critic_review.get('fatal_weakness', {}).get('weakness', 'N/A')[:200]}
- Confidence: {critic_review.get('confidence', 'N/A')}

## Scoring Dimensions

1. **G1 Rationality (0.0-1.0)**: How logically sound is the causal chain? Are the extrapolations reasonable?
2. **G2 Evidence Landscape (0.0-1.0)**: How well-studied are the target and disease? How scarce is the direct target-disease link?
   - High score = target well-studied + disease well-studied + direct link scarce
3. **G3 Falsifiability (0.0-1.0)**: Is the prediction specific, quantifiable, and truly falsifiable?
4. **G4 Feasibility (0.0-1.0)**: Druggability, safety, translational readiness.

For each dimension:
- Provide reasoning (2-3 sentences)
- Give a score (0.0-1.0, one decimal)

Return JSON:
{{
  "G1": {{"score": float, "rationale": str}},
  "G2": {{"score": float, "rationale": str}},
  "G3": {{"score": float, "rationale": str}},
  "G4": {{"score": float, "rationale": str}}
}}
Do NOT include an "overall" field - it will be computed from the weighted sum."""

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result, usage = await asyncio.wait_for(
                    self.llm.structured([
                        {"role": "system", "content": "You are an experienced drug discovery scoring committee member. Score carefully with reasoning."},
                        {"role": "user", "content": prompt},
                    ], max_tokens=1024, task=self.task),
                    timeout=180,
                )
                if isinstance(result, dict) and result.get("error") == "json_parse_failed":
                    last_error = "json_parse_failed"
                    if attempt < self.max_retries:
                        print(f"    ⚠️ Semantic scoring parse failed (attempt {attempt+1}), retrying...", flush=True)
                        continue
                    raise RuntimeError("JSON parse failed after retries")

                # Extract scores
                scores = {}
                for key in ["G1", "G2", "G3", "G4"]:
                    dim = result.get(key, {}) if isinstance(result, dict) else {}
                    if isinstance(dim, dict):
                        scores[key] = round(max(0.0, min(1.0, float(dim.get("score", 0.5)))), 3)
                        scores[f"{key}_rationale"] = dim.get("rationale", "")
                    else:
                        scores[key] = round(max(0.0, min(1.0, float(dim))), 3)

                # Compute overall from weighted sum (never trust LLM arithmetic)
                from dp_indicator.core.scoring import compute_overall
                scores["overall"] = compute_overall(scores)
                scores["overall_rationale"] = "Computed as Σ(weight_i × score_i)"

                self.audit.record("SemanticScorer", "score", "complete", {
                    "indication": indication, "scores": scores, "attempt": attempt + 1
                })
                return scores

            except asyncio.TimeoutError:
                last_error = "timeout"
                if attempt < self.max_retries:
                    print(f"    ⚠️ Semantic scoring timeout (attempt {attempt+1}), retrying...", flush=True)
                    continue
                raise RuntimeError(f"Semantic scoring timeout after {self.max_retries + 1} attempts")
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    print(f"    ⚠️ Semantic scoring error (attempt {attempt+1}): {e}, retrying...", flush=True)
                    continue
                raise RuntimeError(f"Semantic scoring failed after {self.max_retries + 1} attempts: {last_error}")


class RankerAgent:
    """Rank hypotheses using LLM semantic scoring with retry handling."""
    def __init__(self, audit: AuditRecorder, llm: object = None,
                 model: str = None):
        self.audit = audit
        self.llm = llm
        self.model = model
        # Fallback code scorer for documentation only; actual scoring uses LLM
        # scoring.py provides weights + compute_overall; no HypothesisScorer needed

    async def run(self, hypotheses: list[dict], evidence_pool: list[dict],
                  target: str = "") -> list[dict]:
        if not self.llm:
            raise RuntimeError("RankerAgent requires LLM for semantic scoring")

        relevant_evidence = [
            e for e in evidence_pool
            if e.get("relevance_to_target", "unknown") in ("high", "medium")
        ]
        if len(relevant_evidence) < 5:
            relevant_evidence = evidence_pool

        scorer = SemanticScorer(
            self.llm,
            self.audit,
            self.model or "glm-5.1",
            task="critic",
        )

        semaphore = asyncio.Semaphore(3)
        async def _score_one(hyp: dict) -> dict:
            async with semaphore:
                indication = hyp.get("indication", "")
                try:
                    scores = await scorer.score(hyp, relevant_evidence, target)
                    hyp["scores"] = scores
                    hyp["overall_score"] = scores["overall"]
                    hyp["scoring_method"] = "llm_semantic"
                except Exception as e:
                    print(f"  ⚠️ Semantic scoring failed for {indication}: {e}", flush=True)
                    # Failed semantic scoring is a pipeline failure; do not substitute
                    # a separate code-based score.
                    hyp["scores"] = {"G1": 0, "G2": 0, "G3": 0, "G4": 0, "overall": 0,
                                     "_error": str(e), "_source": "scoring_failed"}
                    hyp["overall_score"] = 0.0
                    hyp["scoring_method"] = "failed"
                hyp["n_evidence_scored"] = len(relevant_evidence)
                return hyp

        scored = await asyncio.gather(*[_score_one(h) for h in hypotheses])
        ranked = list(scored)
        ranked.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
        for i, h in enumerate(ranked):
            h["rank"] = i + 1
        self.audit.record("Ranker", "rank", "complete", {"n_hypotheses": len(ranked)})
        return ranked


class FullTextReader:
    """Deep-read evidence with multi-strategy retrieval and paywall fallback.

    Uses a four-strategy cascade: Europe PMC XML, PMC HTML, Unpaywall PDF,
    then Semantic Scholar PDF.
    - Graceful paywall handling: records paywall, falls back to extended abstract
    - Paywall evidence flagged with `_paywalled=True` for downstream agents
    - Configurable max_n (default 5 for Explore phase, higher for targeted reading)
    - LLM summary distinguishes full-text vs abstract sources
    """

    def __init__(self, llm: object, audit: AuditRecorder,
                 model: str = "glm-5.1", task: str = "critic",
                 max_global: int = 5):
        self.llm = llm
        self.audit = audit
        self.model = model
        self.task = task
        self.max_global = max_global
        self.fetcher = None  # lazy init

    async def _ensure_fetcher(self):
        if self.fetcher is None:
            from dp_indicator.clients.fulltext_fetcher import FullTextFetcher
            self.fetcher = FullTextFetcher()
        return self.fetcher

    @staticmethod
    def _select_top_evidence(evidence_pool: list[dict], max_n: int = 10,
                              prefer_pmcid: bool = False) -> list[dict]:
        """Select highest-value evidence for full-text reading.
        Priority: grade_score=4 > 3 > 2 > 1, then relevance, original research, unique study group.
        If prefer_pmcid=True, boost items with PMCID (more likely to have free full text).
        """
        scored = []
        for ev in evidence_pool:
            if not ev.get("inclusion", True):
                continue
            score = 0
            # Higher grade score means higher quality and therefore higher priority.
            grade = ev.get("grade_score", 2)
            if grade >= 4:
                score += 100
            elif grade == 3:
                score += 70
            elif grade == 2:
                score += 40
            elif grade == 1:
                score += 10  # very low quality, deprioritize
            if ev.get("relevance_to_target") == "high":
                score += 30
            ev_type = ev.get("evidence_type", "")
            if ev_type in ("RCT_human", "clinical_trial", "animal", "in_vitro"):
                score += 20
            if ev.get("source_db") in ("pubmed", "europe_pmc"):
                score += 10
            sm = ev.get("source_metadata", {})
            if prefer_pmcid and sm.get("pmcid", "").startswith("PMC"):
                score += 40  # strong boost for OA-likely items
            scored.append((score, ev))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate by study group (first_author + year)
        seen_groups = set()
        selected = []
        for score, ev in scored:
            sm = ev.get("source_metadata", {})
            first_author = sm.get("first_author", "")
            year = sm.get("year", "")
            group_key = f"{first_author}:{year}"
            if first_author and group_key in seen_groups:
                continue
            if first_author:
                seen_groups.add(group_key)
            selected.append(ev)
            if len(selected) >= max_n:
                break
        return selected

    async def _summarize_with_llm(self, ev: dict, full_text: str,
                                   source_type: str, paywalled: bool) -> dict:
        """Use LLM to extract structured summary from full text or abstract.

        Uses the 'fulltext' task model (deepseek-v3.2) for fast, structured extraction.
        Adapts the prompt based on source quality:
        - full_text_xml/html/pdf: ask for detailed extraction
        - extended_abstract: ask for conservative extraction, note limitations
        - abstract_only: minimal extraction, flag as low-confidence
        """
        title = ev.get("title", "")
        citation = ""
        sm = ev.get("source_metadata", {})
        if sm.get("first_author"):
            citation = f"{sm.get('first_author')} et al., {sm.get('journal_short', sm.get('journal', ''))} ({sm.get('year', '')})"

        safe_text = FullTextFetcher.smart_truncate(full_text, max_chars=6000) if source_type.startswith("full_text") else full_text[:3000]

        if source_type.startswith("full_text"):
            source_note = "You have the full text of this paper."
            confidence_instruction = "Extract detailed information from the full text. Include specific quantitative results if available."
        elif source_type == "extended_abstract":
            if paywalled:
                source_note = "The full text is behind a paywall. You only have the title and abstract."
                confidence_instruction = "Be conservative - only extract what is explicitly stated in the abstract. Flag any inferred details as 'abstract-based inference'."
            else:
                source_note = "You have an extended abstract (full text was not obtainable)."
                confidence_instruction = "Extract key findings from the abstract. Note that details may be limited."
        else:
            source_note = "Only minimal content is available."
            confidence_instruction = "Extract what you can but flag low confidence."

        prompt = f"""## Task
Read the following biomedical paper content and extract a structured summary.

## Paper
Title: {title}
Citation: {citation}
Content source: {source_type}
Paywalled: {paywalled}

## Content
{safe_text}

## Instructions
{source_note}
{confidence_instruction}

Return JSON:
{{
  "key_findings": ["finding 1", "finding 2"],
  "mechanisms": ["mechanism description 1"],
  "experimental_model": "Describe the model system used (cell line, animal, human cohort)",
  "effect_size": "Describe the magnitude of the observed effect (quantify if possible)",
  "limitations": "Key limitations of the study",
  "translation_relevance": "How does this finding translate to drug target discovery?"
}}"""

        try:
            result, _ = await asyncio.wait_for(
                self.llm.structured([
                    {"role": "system", "content": "You are a biomedical literature analyst. Extract key information accurately. Do not invent details not in the text."},
                    {"role": "user", "content": prompt},
                ], max_tokens=1024, task="fulltext"),
                timeout=120,
            )
            if isinstance(result, dict) and result.get("error") == "json_parse_failed":
                return {"_source": "parse_failed"}
            return result if isinstance(result, dict) else {}
        except asyncio.TimeoutError:
            return {"_source": "timeout"}
        except Exception as e:
            return {"_source": "error", "_error": str(e)}

    async def run(self, evidence_pool: list[dict],
                  target_evidence: list[dict] = None) -> list[dict]:
        """Deep-read evidence and enrich with structured summaries.

        Args:
            evidence_pool: Full pool (will be mutated in-place for enriched items)
            target_evidence: Optional explicit list to read (default: auto-select top-N)
        """
        fetcher = await self._ensure_fetcher()

        if target_evidence is not None:
            selected = target_evidence
        else:
            selected = self._select_top_evidence(evidence_pool, self.max_global, prefer_pmcid=True)

        print(f"  [heartbeat] FullTextReader: deep-reading {len(selected)} evidence items", flush=True)

        async def _process_one(ev: dict, idx: int) -> None:
            eid = ev.get("evidence_id", "")
            print(f"  [heartbeat] FullTextReader [{idx+1}/{len(selected)}]: {eid}...", flush=True)
            t0 = time.time()

            # Bound the complete four-strategy fetch cascade.
            # Each underlying HTTP request already carries its own httpx timeout, but a
            # misbehaving host (slow-drip response under the connection limit) could still
            # keep the cascade running well past what's useful. Bounding it here guarantees
            # this item can't stall the whole gather; on timeout we fall back to the
            # abstract we already have instead of hanging.
            try:
                fetch_result = await asyncio.wait_for(fetcher.fetch(ev), timeout=90)
                text = fetch_result["text"]
                source_type = fetch_result["source"]
                paywalled = fetch_result["paywalled"]
            except asyncio.TimeoutError:
                print(f"  [heartbeat] FullTextReader [{idx+1}] fetch timeout for {eid}, falling back to abstract", flush=True)
                text = ev.get("abstract_snippet", "")
                source_type = "abstract_only"
                paywalled = True

            is_extended_abstract = source_type in ("extended_abstract", "abstract_only")
            summary = await self._summarize_with_llm(ev, text, source_type, paywalled)
            elapsed = time.time() - t0

            source_label = f"{source_type}" + (" [paywalled]" if paywalled else "")
            print(f"  [heartbeat] FullTextReader [{idx+1}] done in {elapsed:.1f}s ({source_label})", flush=True)

            # Enrich evidence object
            ev["full_text_summary"] = summary
            ev["full_text_source"] = source_type
            ev["_paywalled"] = paywalled
            ev["key_findings"] = summary.get("key_findings", [])
            ev["mechanisms"] = summary.get("mechanisms", [])
            ev["experimental_model"] = summary.get("experimental_model", "")
            ev["effect_size_note"] = summary.get("effect_size", "")
            ev["limitations"] = summary.get("limitations", "")
            ev["translation_relevance"] = summary.get("translation_relevance", "")

            self.audit.record("FullTextReader", "deep_read", "complete",
                              {"evidence_id": eid, "source": source_type,
                               "paywalled": paywalled, "latency_s": elapsed})

        semaphore = asyncio.Semaphore(3)
        async def _bounded(ev, idx):
            async with semaphore:
                return await _process_one(ev, idx)

        tasks = [_bounded(ev, i) for i, ev in enumerate(selected)]
        await asyncio.gather(*tasks)

        await fetcher.close()

        n_enriched = sum(1 for ev in selected if not ev.get("full_text_summary", {}).get("_source"))
        n_paywalled = sum(1 for ev in selected if ev.get("_paywalled"))
        print(f"  ✅ FullTextReader: {n_enriched}/{len(selected)} enriched"
              f"{f', {n_paywalled} paywalled (used abstract fallback)' if n_paywalled else ''}", flush=True)
        return evidence_pool


class MetaCognitiveReflector:
    """LLM meta-cognitive reflection across all hypotheses."""

    def __init__(self, llm: object, audit: AuditRecorder,
                 model: str = "glm-5.1", task: str = "critic"):
        self.llm = llm
        self.audit = audit
        self.model = model
        self.task = task

    async def reflect(self, hypotheses: list[dict], evidence_pool: list[dict],
                      target: str) -> dict:
        """Generate meta-cognitive reflection on the hypothesis set."""
        # Build summary of hypotheses
        hyp_summaries = []
        for h in hypotheses:
            hyp_summaries.append({
                "rank": h.get("rank", "?"),
                "indication": h.get("indication", ""),
                "score": h.get("overall_score", 0),
                "statement": h.get("statement", "")[:200],
                "feasibility": h.get("feasibility_score", 0),
                "fatal_weakness": h.get("critic_review", {}).get("fatal_weakness", {}).get("weakness", "")[:150],
            })

        # Evidence landscape summary
        evidence_stats = {
            "total": len(evidence_pool),
            "by_type": {},
            "by_grade": {},
            "full_text_enriched": sum(1 for e in evidence_pool if e.get("full_text_summary")),
        }
        for ev in evidence_pool:
            et = ev.get("evidence_type", "unknown")
            evidence_stats["by_type"][et] = evidence_stats["by_type"].get(et, 0) + 1
            gs = ev.get("grade_score", 2)
            evidence_stats["by_grade"][gs] = evidence_stats["by_grade"].get(gs, 0) + 1

        prompt = f"""## Task
You are a senior scientific advisor reviewing a set of drug target hypotheses. Provide a meta-cognitive reflection that helps the research team prioritize and improve.

## Target: {target}

## Generated Hypotheses ({len(hyp_summaries)})
{json.dumps(hyp_summaries, ensure_ascii=False, indent=2)}

## Evidence Landscape
{json.dumps(evidence_stats, ensure_ascii=False, indent=2)}

## Reflection Questions
Answer each with honest, specific reasoning:

1. **Uncertainty Ranking**: Which hypothesis are you MOST uncertain about? What is the source of that uncertainty?
2. **Evidence Gaps**: If you had unlimited resources, what is the SINGLE most important piece of evidence to collect next? Why?
3. **Hypothesis Interconnections**: Do these hypotheses share mechanistic links? If one is validated, which others would be strengthened or weakened?
4. **Potential Biases**: What cognitive or literature biases might have influenced these hypotheses? (e.g., over-reliance on positive results, neglect of negative data)
5. **Counter-Intuitive Finding**: Is there any hypothesis whose implications contradict prevailing scientific consensus? Is this contradiction a warning sign or an opportunity?
6. **Recommended Next Step**: What is the ONE action the team should take first to reduce uncertainty?

Return JSON:
{{
  "uncertainty_ranking": {{"hypothesis": str, "reason": str}},
  "evidence_gap": {{"what": str, "why": str}},
  "interconnections": str,
  "potential_biases": [str],
  "counter_intuitive": {{"finding": str, "assessment": str}},
  "recommended_next_step": str,
  "overall_confidence": float (0-1)
}}"""

        try:
            result, _ = await asyncio.wait_for(
                self.llm.structured([
                    {"role": "system", "content": "You are a wise scientific advisor. Be honest about uncertainties and biases. Do not inflate confidence."},
                    {"role": "user", "content": prompt},
                ], max_tokens=1536, task=self.task),
                timeout=300,
            )
            if isinstance(result, dict) and result.get("error") == "json_parse_failed":
                return {"_source": "parse_failed"}
            self.audit.record("MetaCognitiveReflector", "reflect", "complete",
                              {"n_hypotheses": len(hypotheses)})
            return result if isinstance(result, dict) else {}
        except asyncio.TimeoutError:
            print("  ⚠️ Meta-cognitive reflection timeout", flush=True)
            return {"_source": "timeout"}
        except Exception as e:
            print(f"  ⚠️ Meta-cognitive reflection failed: {e}", flush=True)
            return {"_source": "error", "_error": str(e)}


class ExperimentDesigner:
    """Design validation experiments targeting the weakest links in each hypothesis."""

    def __init__(self, llm: object, audit: AuditRecorder,
                 model: str = "glm-5.1", task: str = "critic"):
        self.llm = llm
        self.audit = audit
        self.model = model
        self.task = task

    async def run(self, hypotheses: list[dict], evidence_pool: list[dict],
                  target: str = "") -> list[dict]:
        semaphore = asyncio.Semaphore(3)

        async def _design_one(hyp, idx):
            indication = hyp.get("indication", "")
            print(f"  [heartbeat] Design {idx+1}/{len(hypotheses)}: {indication}...", flush=True)
            async with semaphore:
                try:
                    exp = await asyncio.wait_for(
                        self._design_experiment(hyp, target, evidence_pool),
                        timeout=120,
                    )
                    exp["experiment_id"] = f"EXP-{idx+1}"
                    exp["hypothesis_id"] = hyp.get("hypothesis_id", indication)
                    exp["priority"] = "high" if hyp.get("overall_score", 0) > 0.7 else "medium"
                    return exp
                except asyncio.TimeoutError:
                    print(f"  ⚠️ Design timeout for {indication}", flush=True)
                except Exception as e:
                    print(f"  ⚠️ Design failed for {indication}: {e}", flush=True)
                return self._fallback_experiment(hyp, idx)

        tasks = [_design_one(h, i) for i, h in enumerate(hypotheses[:5])]
        results = await asyncio.gather(*tasks)
        self.audit.record("ExperimentDesigner", "run", "complete",
                          {"n_experiments": len(results)})
        return results

    async def _design_experiment(self, hyp: dict, target: str,
                                  evidence_pool: list[dict] = None) -> dict:
        indication = hyp.get("indication", "")
        statement = hyp.get("statement", "")
        chain = hyp.get("causal_chain", {})
        critic = hyp.get("critic_review", {})
        falsifiable = hyp.get("falsifiable_prediction", "")

        # Identify weakest links
        # Inspect every supported causal-chain structure when collecting weak links.
        weak_links = []
        if isinstance(chain, dict) and "mechanism_axes" in chain:
            for axis in chain.get("mechanism_axes", []):
                axis_name = axis.get("axis_name", "")
                for step in axis.get("steps", []):
                    if step.get("status") in ("hypothesized", "inferred"):
                        weak_links.append({
                            "level": step.get("layer", axis_name),
                            "description": step.get("mechanism", "")[:150],
                            "status": step.get("status"),
                            "has_evidence": bool(step.get("evidence_ids")),
                        })
        elif isinstance(chain, dict):
            for level, link in chain.items():
                if isinstance(link, dict) and link.get("status") in ("hypothesized", "inferred"):
                    weak_links.append({
                        "level": level,
                        "description": link.get("description", "")[:150],
                        "status": link.get("status"),
                        "has_evidence": bool(link.get("evidence_ids")),
                    })

        fatal_weakness = critic.get("fatal_weakness", {})
        suggested_fix = critic.get("suggested_fix", "")

        # Extract methodological hints from full-text-enriched evidence.
        method_hints = []
        if evidence_pool:
            # Find evidence related to this hypothesis' indication
            hyp_ev = [e for e in evidence_pool
                      if indication.lower() in e.get("title", "").lower()
                      or indication.lower() in e.get("abstract_snippet", "").lower()]
            for ev in hyp_ev[:5]:
                fts = ev.get("full_text_summary", {})
                if fts and isinstance(fts, dict) and not fts.get("_source"):
                    model = fts.get("experimental_model", "")
                    if model:
                        method_hints.append({
                            "citation": ev.get("source_metadata", {}).get("first_author", ""),
                            "model": model[:100],
                            "effect_size": fts.get("effect_size", "")[:80],
                        })
        method_hints_text = json.dumps(method_hints[:3], ensure_ascii=False) if method_hints else "N/A"

        prompt = f"""You are a drug discovery experimentalist. Design ONE validation experiment for this hypothesis.

Target: {target}
Indication: {indication}
Hypothesis: {statement[:300]}
Falsifiable prediction: {falsifiable[:200]}

Weakest causal chain links (most need validation):
{json.dumps(weak_links[:5], ensure_ascii=False, indent=2)}

Critic's fatal weakness: {fatal_weakness.get('weakness', 'N/A')[:200]}
Critic's suggested fix: {str(suggested_fix)[:200]}

Published experimental models for this indication (from full-text reading):
{method_hints_text}

Design a concrete experiment. Return JSON:
{{
  "title": str (concise experiment name),
  "model": str (animal model or cell system - reference published models if available),
  "intervention": str (what to administer/change),
  "readout": str (primary endpoint / measurement),
  "control": str (control group),
  "expected_supporting": str (result that supports hypothesis),
  "expected_refuting": str (result that refutes hypothesis),
  "timeline": str (estimated duration),
  "rationale": str (why this experiment addresses the weakest link)
}}"""

        result, _ = await self.llm.structured([
            {"role": "system", "content": "You are a senior experimentalist. Return ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ], max_tokens=1024, task=self.task)

        if isinstance(result, dict) and not result.get("error"):
            return result
        return self._fallback_experiment(hyp, 0)

    @staticmethod
    def _fallback_experiment(hyp: dict, idx: int) -> dict:
        pred = hyp.get("falsifiable_prediction", "")
        return {
            "experiment_id": f"EXP-{idx+1}",
            "hypothesis_id": hyp.get("hypothesis_id", hyp.get("indication", "")),
            "title": f"Validate {hyp.get('indication', '')}",
            "model": "To be determined",
            "intervention": "To be determined",
            "readout": pred[:100] if pred else "To be determined",
            "control": "Vehicle/wild-type",
            "expected_supporting": "Outcome consistent with prediction",
            "expected_refuting": "Outcome contradicts prediction",
            "timeline": "TBD",
            "rationale": "Fallback experiment design (LLM design failed)",
            "priority": "medium",
        }
