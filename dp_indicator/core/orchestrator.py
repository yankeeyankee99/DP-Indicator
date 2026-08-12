from __future__ import annotations
import asyncio
import copy
import json
import os
from pathlib import Path
from typing import Optional


class Orchestrator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("BH_API_KEY", "")
        from dp_indicator.agents.core_agents import AuditRecorder
        from dp_indicator.core.model_router import ModelRouter
        self.audit = AuditRecorder()
        self.model_router = ModelRouter(api_key=self.api_key)
        self._stage_checkpoint_dir = Path("checkpoints/stages")
        self._stage_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._last_knowledge_base: list[dict] = []

    # ── stage checkpoints (crash recovery, not HITL) ──

    def _save_stage_checkpoint(self, stage: str, data: dict):
        path = self._stage_checkpoint_dir / f"{stage}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"  [checkpoint] Saved {stage}: {path}", flush=True)
        except Exception as e:
            print(f"  [checkpoint] Warning: failed to save {stage}: {e}", flush=True)

    def _load_stage_checkpoint(self, stage: str) -> Optional[dict]:
        path = self._stage_checkpoint_dir / f"{stage}.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    # ── prefilter score validation ──

    @staticmethod
    def _spearman_rank(a: list[float], b: list[float]) -> float:
        n = len(a)
        if n < 3:
            return 0.0
        def rank(xs):
            n = len(xs)
            sorted_idx = sorted(range(n), key=lambda i: xs[i])
            ranks = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j + 1 < n and xs[sorted_idx[j + 1]] == xs[sorted_idx[i]]:
                    j += 1
                avg = (i + j) / 2.0 + 1
                for k in range(i, j + 1):
                    ranks[sorted_idx[k]] = avg
                i = j + 1
            return ranks
        ra, rb = rank(a), rank(b)
        mean_a = sum(ra) / n
        mean_b = sum(rb) / n
        num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
        den_a = sum((ra[i] - mean_a) ** 2 for i in range(n)) ** 0.5
        den_b = sum((rb[i] - mean_b) ** 2 for i in range(n)) ** 0.5
        if den_a == 0 or den_b == 0:
            return 0.0
        return num / (den_a * den_b)

    def _validate_prefilter(self, graded_pool: list[dict]):
        """Check correlation between _prefilter_score and LLM grade_score."""
        scored = [e for e in graded_pool if "_prefilter_score" in e and "grade_score" in e]
        if len(scored) < 10:
            return
        pre = [float(e["_prefilter_score"]) for e in scored]
        grd = [float(e.get("grade_score", 2)) for e in scored]
        rho = self._spearman_rank(pre, grd)
        print(f"  📊 预过滤 vs 实际分级 Spearman ρ={rho:.2f}")
        if rho < 0.3:
            print(f"  ⚠️ 预过滤分数与实际分级相关性低，采样可能存在偏差")

    # ── Phase 1: Explore ──

    # ── Intent parsing ──

    async def parse_intent(self, user_input: str) -> dict:
        """Parse natural language input into structured query using LLM, with regex fallback."""
        from dp_indicator.core.llm import LLMClient
        from dp_indicator.core.intent_parser import parse_query_with_llm

        llm = LLMClient(
            model=self.model_router.get_model_for_task("retriever"),
            api_key=self.api_key,
            semaphore=self.model_router.get_semaphore("retriever"),
            router=self.model_router,
        )
        try:
            query = await parse_query_with_llm(user_input, llm)
            if query.get("target"):
                print(f"  🎯 识别靶点: {query['target']}", flush=True)
            if query.get("focus_areas"):
                print(f"  📌 关注方向: {', '.join(query['focus_areas'])}", flush=True)
            if query.get("exclude_areas"):
                print(f"  🚫 排除方向: {', '.join(query['exclude_areas'])}", flush=True)
            return query
        finally:
            await llm.aclose()

    # ── Phase 1: Explore ──

    async def init(self, query: dict) -> dict:
        status = {
            "phase": "init",
            "target": query.get("target", ""),
            "status": "ok",
            "checks": {},
        }
        if not self.api_key:
            status["checks"]["api_key"] = "FAIL: BH_API_KEY not set"
            status["status"] = "error"
        else:
            status["checks"]["api_key"] = "OK"
        from dp_indicator.clients.databases import (
            PubMedClient, EuropePMCClient, ChEMBLClient, KEGGClient,
            UniProtClient, OpenTargetsClient, GWASCatalogClient,
            ClinicalTrialsClient, BioRxivClient, MONDOClient,
        )
        clients_to_test = {
            "pubmed": PubMedClient(),
            "europe_pmc": EuropePMCClient(),
            "chembl": ChEMBLClient(),
            "kegg": KEGGClient(),
            "uniprot": UniProtClient(),
            "opentargets": OpenTargetsClient(),
            "gwas": GWASCatalogClient(),
            "clinicaltrials": ClinicalTrialsClient(),
            "biorxiv": BioRxivClient(),
            "mondo": MONDOClient(),
        }
        for name, client in clients_to_test.items():
            try:
                if name in ("gwas", "clinicaltrials"):
                    results = await client.search("", disease="diabetes", max_results=1)
                else:
                    results = await client.search("test", max_results=1)
                status["checks"][name] = f"OK ({len(results)} results)"
            except Exception as e:
                status["checks"][name] = f"FAIL: {str(e)[:100]}"
            finally:
                client.close()
        return status

    async def explore(self, query: dict) -> list[dict]:
        from dp_indicator.agents.core_agents import RetrieverAgent, EvidenceGrader, EvidenceFilter
        from dp_indicator.core.llm import LLMClient
        from dp_indicator.clients.databases import (
            PubMedClient, EuropePMCClient, ChEMBLClient, KEGGClient,
            UniProtClient, OpenTargetsClient, GWASCatalogClient,
            ClinicalTrialsClient, BioRxivClient, MONDOClient,
        )
        retriever_model = self.model_router.get_model_for_task("retriever")
        retriever_llm = LLMClient(
            model=retriever_model,
            api_key=self.api_key,
            semaphore=self.model_router.get_semaphore("retriever"),
            router=self.model_router,
        )
        clients = {
            "pubmed": PubMedClient(),
            "europe_pmc": EuropePMCClient(),
            "chembl": ChEMBLClient(),
            "kegg": KEGGClient(),
            "uniprot": UniProtClient(),
            "opentargets": OpenTargetsClient(),
            "biorxiv": BioRxivClient(),
            "mondo": MONDOClient(),
        }
        try:
            retriever = RetrieverAgent(
                clients=clients, llm=retriever_llm, audit=self.audit,
                model=retriever_model, task="retriever",
            )
            focus_areas = query.get("focus_areas", [])
            print(f"  [heartbeat] Starting retrieval for target: {query['target']}...", flush=True)
            evidence_pool = await retriever.run(
                target=query["target"],
                synonyms=query.get("synonyms", []),
                direction=query.get("direction", "target_to_indication"),
                focus_areas=focus_areas,
            )
            log = getattr(retriever, '_last_retrieval_log', [])
            total_raw = sum(e.get('n_results', 0) for e in log)
            # Collect retrieval-stage statistics for report.json.
            pipeline_stats = {
                "total_raw_results": total_raw,
                "total_queries": len(log),
                "after_dedup": len(evidence_pool),
                "after_prefilter": 0,  # will be filled after pre_filter
                "after_grading": 0,  # will be filled after grading
            }
            print(f"  ✅ 检索完成: {len(evidence_pool)} 条去重证据(共 {len(log)} 次查询，{total_raw} 条原始结果)")
            if focus_areas:
                focus_hits = [e for e in log if 'focus' in e.get('type', '') and e.get('n_results', 0) > 0]
                if focus_hits:
                    print(f"  🎯 方向引导检索命中 {len(focus_hits)} 次")

            print(f"  [heartbeat] Running Tier 1 pre-filter on {len(evidence_pool)} items...", flush=True)
            evidence_pool = EvidenceFilter.pre_filter(evidence_pool, query["target"])
            pipeline_stats["after_prefilter"] = len(evidence_pool)
            print(f"  📏 Tier 1 预过滤后: {len(evidence_pool)} 条候选")

            max_evidence_for_grading = 150
            if len(evidence_pool) > max_evidence_for_grading:
                from collections import defaultdict
                by_source = defaultdict(list)
                for ev in evidence_pool:
                    by_source[ev.get("source_db", "unknown")].append(ev)
                for src in by_source:
                    by_source[src].sort(key=lambda e: e.get("_prefilter_score", 0), reverse=True)
                total = len(evidence_pool)
                quotas = {}
                for src, items in by_source.items():
                    prop = int(max_evidence_for_grading * len(items) / total)
                    quotas[src] = min(len(items), max(2, prop))
                while sum(quotas.values()) > max_evidence_for_grading:
                    reducible = [s for s in by_source if quotas[s] > 2]
                    if not reducible:
                        reducible = [s for s in by_source if quotas[s] > 1]
                    if not reducible:
                        break
                    largest = max(reducible, key=lambda s: quotas[s])
                    quotas[largest] -= 1
                while sum(quotas.values()) < max_evidence_for_grading:
                    added = False
                    for src in sorted(by_source.keys(), key=lambda s: len(by_source[s]) - quotas[s], reverse=True):
                        if quotas[src] < len(by_source[src]):
                            quotas[src] += 1
                            added = True
                            break
                    if not added:
                        break
                sampled = []
                for src in sorted(by_source.keys()):
                    items = by_source[src]
                    sampled.extend(items[:quotas[src]])
                evidence_pool = sampled
                source_summary = ", ".join(f"{src}:{quotas[src]}" for src in sorted(by_source.keys()))
                print(f"  📏 证据池分层采样: {len(evidence_pool)}/{total} 条({source_summary})")

                # sampling transparency
                for src in sorted(by_source.keys()):
                    orig = len(by_source[src])
                    kept = quotas[src]
                    dropped_pct = (1 - kept / orig) * 100 if orig > 0 else 0
                    if dropped_pct > 70:
                        print(f"  ⚠️ 采样偏差: {src} 丢弃 {dropped_pct:.0f}%({orig}→{kept})")

            grader_model = self.model_router.get_model_for_task("grader")
            grader_llm = LLMClient(
                model=grader_model,
                api_key=self.api_key,
                semaphore=self.model_router.get_semaphore("grader"),
                router=self.model_router,
            )
            grader = EvidenceGrader(
                llm=grader_llm, audit=self.audit,
                model=grader_model, task="grader",
            )
            print(f"  [heartbeat] Starting grading ({len(evidence_pool)} items)...", flush=True)
            graded_pool = await grader.run(evidence_pool)
            graded_pool = [e for e in graded_pool if e.get("inclusion", True)]
            inclusion_count = len(graded_pool)
            pipeline_stats["after_grading"] = inclusion_count
            print(f"  ✅ 分级完成: {inclusion_count} 条有效证据")

            if inclusion_count < 20:
                print(f"  ⚠️ 证据不足: 仅 {inclusion_count} 条(推荐≥20)")

            # prefilter score validation
            self._validate_prefilter(graded_pool)

            # Full-text deep reading
            try:
                from dp_indicator.agents.core_agents import FullTextReader
                reader_llm = LLMClient(
                    model=self.model_router.get_model_for_task("fulltext"),
                    api_key=self.api_key,
                    semaphore=self.model_router.get_semaphore("fulltext"),
                    router=self.model_router,
                )
                reader = FullTextReader(
                    llm=reader_llm, audit=self.audit,
                    model=self.model_router.get_model_for_task("fulltext"),
                    task="fulltext",
                    max_global=10,  # Bound global full-text reads while retaining coverage.
                )
                graded_pool = await reader.run(graded_pool)
                await reader_llm.aclose()
            except Exception as e:
                print(f"  ⚠️ FullTextReader failed: {e}", flush=True)

            # Record paywall coverage in the audit trail.
            n_paywalled = sum(1 for e in graded_pool if e.get("_paywalled"))
            if n_paywalled:
                print(f"  📋 付费墙报告: {n_paywalled} 篇无法获取全文，已使用摘要降级处理")

            # Distill the full evidence pool so downstream reasoning is not limited
            # to the small set of raw excerpts selected by grade score.
            # lottery. This reads every included item, not a top-N slice.
            knowledge_base = []
            try:
                from dp_indicator.agents.core_agents import KnowledgeSynthesizer
                synth_model = self.model_router.get_model_for_task("synthesizer")
                synth_llm = LLMClient(
                    model=synth_model,
                    api_key=self.api_key,
                    semaphore=self.model_router.get_semaphore("synthesizer"),
                    router=self.model_router,
                )
                try:
                    synthesizer = KnowledgeSynthesizer(
                        llm=synth_llm, audit=self.audit,
                        model=synth_model, task="synthesizer",
                    )
                    knowledge_base = await synthesizer.run(graded_pool, target=query["target"])
                finally:
                    await synth_llm.aclose()
            except Exception as e:
                print(f"  ⚠️ KnowledgeSynthesizer failed: {e}, continuing without knowledge base", flush=True)

            self.audit.record("Orchestrator", "explore", "complete",
                              {"n_evidence": len(graded_pool), "n_knowledge_facts": len(knowledge_base)})
            self._save_stage_checkpoint("explore", {
                "evidence_pool": graded_pool,
                "knowledge_base": knowledge_base,
                "query": query,
                "retrieval_log": log,
                "pipeline_stats": pipeline_stats,
            })
            self._pipeline_stats = pipeline_stats  # store for report generation
            self._last_knowledge_base = knowledge_base  # store for in-process hypothesize() call
            return graded_pool
        finally:
            await retriever_llm.aclose()
            if 'grader_llm' in locals():
                await grader_llm.aclose()
            for c in clients.values():
                c.close()

    # ── Phase 2: Hypothesize ──

    async def _run_evidence_verifier(
        self,
        hypotheses: list[dict],
        evidence_pool: list[dict],
        target: str,
    ) -> dict:
        from dp_indicator.agents.evidence_verifier import EvidenceVerifier
        from dp_indicator.core.llm import LLMClient

        verifier_llm = LLMClient(
            model=self.model_router.get_model_for_task("verifier"),
            api_key=self.api_key,
            semaphore=self.model_router.get_semaphore("verifier"),
            router=self.model_router,
        )
        try:
            verifier = EvidenceVerifier(
                llm=verifier_llm,
                audit=self.audit,
                model=self.model_router.get_model_for_task("verifier"),
                task="verifier",
            )
            return await verifier.verify(hypotheses, evidence_pool, target=target)
        finally:
            await verifier_llm.aclose()

    async def _verification_for_generation(
        self,
        hypotheses: list[dict],
        evidence_pool: list[dict],
        target: str,
    ) -> dict:
        return await self._run_evidence_verifier(
            hypotheses,
            evidence_pool,
            target,
        )

    @staticmethod
    def _apply_verification_adjustments(
        ranked: list[dict],
        adjustments: dict[str, float],
    ) -> list[dict]:
        for hypothesis in ranked:
            hid = hypothesis.get(
                "hypothesis_id",
                hypothesis.get("indication", ""),
            )
            adjustment = float(adjustments.get(hid, 0.0))
            if not adjustment:
                continue
            original = float(
                hypothesis.get(
                    "overall_score",
                    hypothesis.get("scores", {}).get("overall", 0.0),
                )
            )
            adjusted = max(0.0, min(1.0, original + adjustment))
            scores = hypothesis.setdefault("scores", {})
            scores["overall"] = adjusted
            scores["_verification_adjustment"] = adjustment
            hypothesis["overall_score"] = adjusted
            print(
                f"  [heartbeat] Verification adjustment for {hid}: "
                f"{original:.3f} -> {adjusted:.3f} ({adjustment:+.3f})",
                flush=True,
            )
        ranked.sort(
            key=lambda item: item.get("overall_score", 0.0),
            reverse=True,
        )
        for index, hypothesis in enumerate(ranked, start=1):
            hypothesis["rank"] = index
        return ranked

    async def _critic_for_generation(
        self,
        critic,
        hypotheses: list[dict],
        evidence_pool: list[dict],
        target: str,
    ) -> list[dict]:
        mapped = await critic.map_evidence(
            hypotheses,
            evidence_pool,
            target,
        )
        self._save_stage_checkpoint("post_mapping", {
            "hypotheses": copy.deepcopy(mapped),
            "evidence_pool": copy.deepcopy(evidence_pool),
            "target": target,
        })
        return await critic.review_hypotheses(
            mapped,
            evidence_pool,
            target,
        )

    async def hypothesize(self, query: dict, evidence_pool: list[dict],
                          knowledge_base: list[dict] | None = None) -> list[dict]:
        from dp_indicator.agents.core_agents import (
            ReasonerAgent, RankerAgent, HypothesisCritic, EvidenceGrader,
        )
        from dp_indicator.core.llm import LLMClient

        # Reuse the knowledge base built by explore() for one-shot pipeline runs;
        # standalone hypothesis generation passes it explicitly.
        # explicitly after loading it from the explore checkpoint.
        if knowledge_base is None:
            knowledge_base = self._last_knowledge_base or []

        reasoner_model = self.model_router.get_model_for_task("reasoner")
        reasoner_llm = LLMClient(
            model=reasoner_model,
            api_key=self.api_key,
            semaphore=self.model_router.get_semaphore("reasoner"),
            router=self.model_router,
            # Include the full-corpus knowledge base in addition to raw excerpts.
            # latency. The default 180s httpx timeout was too tight for this, causing
            # multiple wasted ReadTimeout retries (each burning ~180s) before eventually
            # succeeding. A longer per-request timeout lets the first attempt actually
            # finish instead of getting killed and retried from scratch.
            timeout=300,
        )
        pre_critic_llm = None
        critic_llm = None
        gwas_client = None
        ct_client = None
        disease_background = {}

        try:
            focus_areas = query.get("focus_areas", [])
            if focus_areas:
                from dp_indicator.clients.databases import GWASCatalogClient, ClinicalTrialsClient
                gwas_client = GWASCatalogClient()
                ct_client = ClinicalTrialsClient()
                for focus in focus_areas:
                    try:
                        gwas_results = await gwas_client.search("", disease=focus, max_results=10)
                        ct_results = await ct_client.search("", disease=focus, max_results=10)
                        disease_background[focus] = {
                            "gwas_count": len(gwas_results or []),
                            "clinical_trials_count": len(ct_results or []),
                            "gwas_samples": [r.get("title", "") for r in (gwas_results or [])[:3]],
                            "ct_samples": [r.get("title", "") for r in (ct_results or [])[:3]],
                        }
                        print(f"  '{focus}': GWAS={len(gwas_results or [])}, CT={len(ct_results or [])} → 疾病背景")
                    except Exception as e:
                        print(f"  '{focus}': 检索错误 {str(e)[:60]}")

            reasoner = ReasonerAgent(
                llm=reasoner_llm, audit=self.audit,
                model=reasoner_model, task="reasoner",
            )
            print(f"  Calling reasoner with {len(evidence_pool)} evidence items "
                  f"+ {len(knowledge_base)} knowledge-base facts...", flush=True)
            raw_hypotheses = await reasoner.run(
                target=query["target"],
                evidence_pool=evidence_pool,
                direction=query.get("direction", "target_to_indication"),
                focus=query.get("focus_areas"),
                exclude=query.get("exclude_areas"),
                disease_background=disease_background,
                knowledge_base=knowledge_base,
            )
            print(f"  reasoner done: {len(raw_hypotheses)} hypotheses", flush=True)

            # Normalize fields
            for hyp in raw_hypotheses:
                if not hyp.get('indication') and hyp.get('indication_name'):
                    hyp['indication'] = hyp['indication_name']
                if not hyp.get('statement') and hyp.get('one_sentence_statement'):
                    hyp['statement'] = hyp['one_sentence_statement']
                if isinstance(hyp.get('causal_chain'), list):
                    chain_list = hyp['causal_chain']
                    chain_dict = {}
                    for item in chain_list:
                        level = item.get('level', item.get('layer', 'unknown'))
                        chain_dict[level] = {
                            'description': item.get('description', item.get('mechanism', '')),
                            'status': item.get('status', 'inferred'),
                            'evidence_ids': item.get('evidence_ids', []),
                            'source_text': item.get('source_text', ''),
                        }
                    hyp['causal_chain'] = chain_dict

            # Validate evidence IDs
            # Validate IDs inside both mechanism-axis and mapping structures.
            # to actually re-verify anything for the current schema.
            valid_ids = {e.get("evidence_id", "") for e in evidence_pool}
            for hyp in raw_hypotheses:
                chain = hyp.get("causal_chain", {})
                if not isinstance(chain, dict):
                    continue
                if "mechanism_axes" in chain:
                    steps = [s for axis in chain.get("mechanism_axes", []) for s in axis.get("steps", [])]
                else:
                    steps = [link for link in chain.values() if isinstance(link, dict)]
                for step in steps:
                    claimed = step.get("evidence_ids", [])
                    verified = [cid for cid in claimed if cid in valid_ids]
                    step["evidence_ids"] = verified
                    if step.get("status") == "supported" and not verified:
                        step["status"] = "inferred"

            # Critic review
            critic_model = self.model_router.get_model_for_task("critic")
            critic_llm = LLMClient(
                model=critic_model,
                api_key=self.api_key,
                semaphore=self.model_router.get_semaphore("critic"),
                router=self.model_router,
            )
            critic = HypothesisCritic(
                llm=critic_llm, audit=self.audit,
                model=critic_model, task="critic",
            )
            print("  [heartbeat] Running critic review...", flush=True)
            reviewed = await self._critic_for_generation(
                critic,
                raw_hypotheses,
                evidence_pool,
                query["target"],
            )

            # Assess which evidence needs full-text reading.
            try:
                from dp_indicator.agents.evidence_prioritizer import EvidencePrioritizer
                prioritizer_llm = LLMClient(
                    model=self.model_router.get_model_for_task("prioritizer"),
                    api_key=self.api_key,
                    semaphore=self.model_router.get_semaphore("prioritizer"),
                    router=self.model_router,
                )
                try:
                    prioritizer = EvidencePrioritizer(
                        llm=prioritizer_llm, audit=self.audit,
                        model=self.model_router.get_model_for_task("prioritizer"),
                        task="prioritizer",
                    )
                    reading_queue = await prioritizer.prioritize(
                        evidence_pool, reviewed, target=query["target"]
                    )
                    # Execute full-text reading for priority items
                    if reading_queue:
                        from dp_indicator.agents.core_agents import FullTextReader
                        fulltext_llm = LLMClient(
                            model=self.model_router.get_model_for_task("fulltext"),
                            api_key=self.api_key,
                            semaphore=self.model_router.get_semaphore("fulltext"),
                            router=self.model_router,
                        )
                        try:
                            reader = FullTextReader(
                                llm=fulltext_llm, audit=self.audit,
                                model=self.model_router.get_model_for_task("fulltext"),
                                task="fulltext",
                                max_global=min(10, len(reading_queue)),
                            )
                            evidence_pool = await reader.run(evidence_pool, target_evidence=reading_queue)
                        finally:
                            await fulltext_llm.aclose()
                finally:
                    await prioritizer_llm.aclose()
            except Exception as e:
                print(f"  ⚠️ EvidencePrioritizer failed: {e}", flush=True)

            # Perform targeted full-text reading for evidence gaps.
            evidence_pool = await self._fill_evidence_gaps(
                reviewed, evidence_pool, query["target"], critic_llm
            )

            # Evidence Verifier: preserve raw output, sanitize deterministic hard
            # failures, and keep both raw/final reports for a complete audit trail.
            self._verification_report_raw = {}
            self._verification_report = {}
            self._verification_summary = ""
            self._verification_adjustments = {}
            self._verification_sanitization = []
            self._save_stage_checkpoint("pre_verification", {
                "hypotheses": copy.deepcopy(reviewed),
                "evidence_pool": evidence_pool,
                "query": query,
            })
            try:
                from dp_indicator.agents.evidence_verifier import (
                    filter_verification_report,
                    sanitize_hypotheses,
                )

                verification_result = await self._verification_for_generation(
                    reviewed,
                    evidence_pool,
                    query["target"],
                )
                raw_report = verification_result.get(
                    "verification_report",
                    {},
                )
                reviewed, sanitization_audit = sanitize_hypotheses(
                    reviewed,
                    verification_result,
                )
                final_report = filter_verification_report(
                    raw_report,
                    reviewed,
                )
                self._verification_report_raw = raw_report
                self._verification_report = final_report
                self._verification_adjustments = verification_result.get(
                    "score_adjustments",
                    {},
                )
                self._verification_sanitization = sanitization_audit
                self._verification_summary = (
                    verification_result.get("summary", "")
                    + f"\nSanitization actions: {len(sanitization_audit)}"
                ).strip()
                self._save_stage_checkpoint("verification", {
                    "raw_verification_report": raw_report,
                    "final_verification_report": final_report,
                    "score_adjustments": self._verification_adjustments,
                    "sanitization_audit": sanitization_audit,
                    "sanitized_hypotheses": reviewed,
                })
            except Exception as e:
                print(f"  ⚠️ EvidenceVerifier failed: {e}", flush=True)
                self._verification_report_raw = {}
                self._verification_report = {}
                self._verification_summary = (
                    "Verification skipped due to error"
                )
                self._verification_adjustments = {}
                self._verification_sanitization = []

            # Rank
            ranker = RankerAgent(
                audit=self.audit,
                llm=critic_llm, model=critic_model,
            )
            ranked = await ranker.run(reviewed, evidence_pool, target=query["target"])
            ranked = self._apply_verification_adjustments(
                ranked,
                self._verification_adjustments,
            )

            # feasibility_score from G4
            for h in ranked:
                scores = h.get("scores", {})
                h["feasibility_score"] = scores.get("G4", 0.0)

            # Meta-cognitive reflection
            try:
                from dp_indicator.agents.core_agents import MetaCognitiveReflector
                reflector = MetaCognitiveReflector(
                    llm=critic_llm, audit=self.audit,
                    model=critic_model, task="critic",
                )
                reflection = await reflector.reflect(ranked, evidence_pool, query["target"])
                if ranked:
                    ranked[0]["meta_reflection"] = reflection
            except Exception as e:
                print(f"  ⚠️ Meta-cognitive reflection failed: {e}", flush=True)

            self._save_stage_checkpoint("hypothesize", {
                "hypotheses": ranked,
                "evidence_pool": evidence_pool,
                "disease_background": disease_background,
                "query": query,
            })
            return ranked
        finally:
            await reasoner_llm.aclose()
            if pre_critic_llm:
                await pre_critic_llm.aclose()
            if critic_llm:
                await critic_llm.aclose()
            if gwas_client:
                gwas_client.close()
            if ct_client:
                ct_client.close()

    # ── Phase 3: Design experiments ──

    async def design(self, hypotheses: list[dict], evidence_pool: list[dict],
                     query: dict = None) -> list[dict]:
        from dp_indicator.agents.core_agents import ExperimentDesigner
        from dp_indicator.core.llm import LLMClient

        critic_model = self.model_router.get_model_for_task("critic")
        critic_llm = LLMClient(
            model=critic_model,
            api_key=self.api_key,
            semaphore=self.model_router.get_semaphore("critic"),
            router=self.model_router,
        )
        try:
            # Fetch method literature for experiment design.
            evidence_pool = await self._fetch_method_literature(
                hypotheses, evidence_pool, query.get("target", "") if query else ""
            )

            designer = ExperimentDesigner(
                llm=critic_llm, audit=self.audit,
                model=critic_model, task="critic",
            )
            print("  [heartbeat] Designing experiments...", flush=True)
            experiments = await designer.run(hypotheses, evidence_pool, target=query.get("target", "") if query else "")

            self._save_stage_checkpoint("design", {
                "hypotheses": hypotheses,
                "evidence_pool": evidence_pool,
                "experiments": experiments,
            })
            return experiments
        finally:
            await critic_llm.aclose()

    # ── Phase 4: Report ──

    # ── Targeted full-text reading helpers ──

    async def _fill_evidence_gaps(self, hypotheses: list[dict],
                                  evidence_pool: list[dict],
                                  target: str, critic_llm) -> list[dict]:
        """Detect evidence gaps in hypothesis causal chains and fetch targeted full text.

        For each hypothesis, identify causal chain links with status='inferred'
        (no supporting evidence). Search PubMed for the specific mechanism + indication,
        fetch full text of top results, and add to evidence pool.
        """
        from dp_indicator.agents.core_agents import FullTextReader
        from dp_indicator.clients.databases import PubMedClient

        # Collect all gap queries
        gap_queries = []
        for hyp in hypotheses[:5]:  # top-5 hypotheses
            chain = hyp.get("causal_chain", {})
            indication = hyp.get("indication", "")
            if not isinstance(chain, dict):
                continue
            # Handle every supported causal-chain structure.
            if "mechanism_axes" in chain:
                for axis in chain.get("mechanism_axes", []):
                    for step in axis.get("steps", []):
                        status = step.get("status", "inferred")
                        if status in ("inferred", "hypothesized"):
                            desc = (step.get("description") or step.get("mechanism", ""))[:100]
                            if desc and indication:
                                gap_queries.append({
                                    "indication": indication,
                                    "mechanism": desc,
                                    "query": f"{target} {indication} {desc}".strip()[:200],
                                })
            else:
                for level, link in chain.items():
                    if isinstance(link, dict) and link.get("status") in ("inferred", "hypothesized"):
                        desc = link.get("description", "")[:100]
                        if desc and indication:
                            gap_queries.append({
                                "indication": indication,
                                "mechanism": desc,
                                "query": f"{target} {indication} {desc}".strip()[:200],
                            })

        if not gap_queries:
            print("  [heartbeat] No evidence gaps detected — skipping targeted full-text fetch", flush=True)
            return evidence_pool

        # Deduplicate queries
        seen = set()
        unique_queries = []
        for q in gap_queries:
            key = q["query"].lower()
            if key not in seen:
                seen.add(key)
                unique_queries.append(q)

        # Limit to top 8 gap queries to control cost
        unique_queries = unique_queries[:8]
        print(f"  [heartbeat] Evidence gap detected: {len(unique_queries)} targeted queries", flush=True)

        # Search PubMed for each gap query
        pubmed = PubMedClient()
        new_evidence = []
        try:
            for q in unique_queries:
                try:
                    # Bound each gap search so a hung PubMed connection cannot stall
                    # the post-critic stage (the search also has httpx-level
                    # timeouts + retry, this is an outer safety net consistent with the
                    # other LLM/network stages).
                    results = await asyncio.wait_for(
                        pubmed.search(q["query"], max_results=5), timeout=60
                    )
                    # Filter out already-known evidence
                    existing_ids = {e.get("evidence_id") for e in evidence_pool}
                    fresh = [e for e in results if e.get("evidence_id") not in existing_ids]
                    if fresh:
                        print(f"    → '{q['indication']}': {len(fresh)} new evidence", flush=True)
                        new_evidence.extend(fresh[:3])  # top-3 per query
                except Exception as e:
                    print(f"    → '{q['indication']}': search error {str(e)[:60]}", flush=True)
        finally:
            pubmed.close()

        if not new_evidence:
            print("  [heartbeat] No new evidence found for gaps", flush=True)
            return evidence_pool

        # Quick-grade the new evidence with proper grader LLM
        from dp_indicator.agents.core_agents import EvidenceGrader
        from dp_indicator.core.llm import LLMClient
        grader_llm = LLMClient(
            model=self.model_router.get_model_for_task("grader"),
            api_key=self.api_key,
            semaphore=self.model_router.get_semaphore("grader"),
            router=self.model_router,
        )
        try:
            grader = EvidenceGrader(
                llm=grader_llm, audit=self.audit,
                model=self.model_router.get_model_for_task("grader"),
                task="grader",
            )
            graded_new = await grader.run(new_evidence)
        finally:
            await grader_llm.aclose()
        graded_new = [e for e in graded_new if e.get("inclusion", True)]

        # Full-text read the best new evidence (use fulltext LLM)
        if graded_new:
            # Create dedicated fulltext LLM client
            fulltext_llm = LLMClient(
                model=self.model_router.get_model_for_task("fulltext"),
                api_key=self.api_key,
                semaphore=self.model_router.get_semaphore("fulltext"),
                router=self.model_router,
            )
            try:
                reader = FullTextReader(
                    llm=fulltext_llm, audit=self.audit,
                model=self.model_router.get_model_for_task("fulltext"),
                task="fulltext",
                max_global=min(10, len(graded_new)),
            )
                graded_new = await reader.run(graded_new)
            finally:
                await fulltext_llm.aclose()

        evidence_pool.extend(graded_new)
        print(f"  ✅ Gap fill: added {len(graded_new)} new evidence (pool now {len(evidence_pool)})", flush=True)
        return evidence_pool

    async def _fetch_method_literature(self, hypotheses: list[dict],
                                      evidence_pool: list[dict],
                                      target: str) -> list[dict]:
        """Fetch methodological literature for experiment design.

        For each hypothesis, search PubMed for experimental protocols
        related to the indication + target, and full-text read the best ones
        to extract model/dose/readout information.
        """
        from dp_indicator.agents.core_agents import FullTextReader
        from dp_indicator.clients.databases import PubMedClient

        method_queries = []
        for hyp in hypotheses[:5]:
            indication = hyp.get("indication", "")
            if indication:
                method_queries.append({
                    "indication": indication,
                    "query": f"{target} {indication} experimental model protocol"[:200],
                })

        if not method_queries:
            return evidence_pool

        print(f"  [heartbeat] Fetching method literature: {len(method_queries)} queries", flush=True)

        pubmed = PubMedClient()
        new_evidence = []
        try:
            for q in method_queries:
                try:
                    results = await pubmed.search(q["query"], max_results=3)
                    existing_ids = {e.get("evidence_id") for e in evidence_pool}
                    fresh = [e for e in results if e.get("evidence_id") not in existing_ids]
                    if fresh:
                        new_evidence.extend(fresh[:2])  # top-2 per query
                except Exception:
                    pass
        finally:
            pubmed.close()

        if not new_evidence:
            return evidence_pool

        # Quick-grade
        from dp_indicator.agents.core_agents import EvidenceGrader
        from dp_indicator.core.llm import LLMClient
        grader_llm = LLMClient(
            model=self.model_router.get_model_for_task("grader"),
            api_key=self.api_key,
            semaphore=self.model_router.get_semaphore("grader"),
            router=self.model_router,
        )
        try:
            grader = EvidenceGrader(
                llm=grader_llm, audit=self.audit,
                model=self.model_router.get_model_for_task("grader"),
                task="grader",
            )
            graded_new = await grader.run(new_evidence)
            graded_new = [e for e in graded_new if e.get("inclusion", True)]
        finally:
            await grader_llm.aclose()

        if graded_new:
            # Use fulltext LLM for full-text reading
            fulltext_llm = LLMClient(
                model=self.model_router.get_model_for_task("fulltext"),
                api_key=self.api_key,
                semaphore=self.model_router.get_semaphore("fulltext"),
                router=self.model_router,
            )
            try:
                reader = FullTextReader(
                    llm=fulltext_llm, audit=self.audit,
                    model=self.model_router.get_model_for_task("fulltext"),
                    task="fulltext",
                    max_global=min(5, len(graded_new)),
                )
                graded_new = await reader.run(graded_new)
                evidence_pool.extend(graded_new)
                print(f"  ✅ Method literature: added {len(graded_new)} items", flush=True)
            finally:
                await fulltext_llm.aclose()

        return evidence_pool

    # ── Phase 4: Report ──

    def generate_report(self, hypotheses: list[dict], experiments: list[dict],
                        query: dict) -> dict:
        from dp_indicator.reporting.generator import ReportGenerator
        degradation_notes = []
        explore_ckpt = self._load_stage_checkpoint("explore")
        if explore_ckpt:
            retrieval_log = explore_ckpt.get("retrieval_log", [])
            db_errors = [e for e in retrieval_log if e.get("status") == "error"]
            if db_errors:
                degradation_notes.append(
                    f"数据库检索异常: {len(db_errors)} 次({', '.join(set(d['db'] for d in db_errors))})"
                )
        query = dict(query) if query else {}
        if degradation_notes:
            query["_degradation_notes"] = degradation_notes
        # Add pipeline statistics to the report payload.
        pipeline_stats = getattr(self, "_pipeline_stats", None)
        if not pipeline_stats and explore_ckpt:
            pipeline_stats = explore_ckpt.get("pipeline_stats")
        if pipeline_stats:
            query["_pipeline_stats"] = pipeline_stats
        # Add verification results to the report payload.
        verification_report = getattr(self, "_verification_report", None)
        if verification_report:
            query["_verification_report"] = verification_report
        verification_summary = getattr(self, "_verification_summary", None)
        if verification_summary:
            query["_verification_summary"] = verification_summary
        verification_report_raw = getattr(
            self,
            "_verification_report_raw",
            None,
        )
        if verification_report_raw:
            query["_verification_report_raw"] = verification_report_raw
        verification_adjustments = getattr(
            self,
            "_verification_adjustments",
            None,
        )
        if verification_adjustments:
            query["_verification_adjustments"] = verification_adjustments
        verification_sanitization = getattr(
            self,
            "_verification_sanitization",
            None,
        )
        if verification_sanitization:
            query["_verification_sanitization"] = verification_sanitization
        gen = ReportGenerator()
        return gen.generate(hypotheses, experiments, query)
