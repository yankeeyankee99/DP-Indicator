"""Evidence Prioritizer - evaluates which evidence items need full-text reading.

After ReasonerAgent generates hypotheses using abstracts, this module:
1. Identifies evidence directly cited in hypothesis causal chains
2. Uses LLM to assess each evidence item's importance for full-text verification
3. Produces a priority queue for targeted full-text reading

Design principles:
- Quality over speed: uses glm-5.1 for nuanced judgment
- Hypothesis-aware: importance depends on which hypothesis the evidence supports
- Paywall-aware: tracks whether full text is likely accessible
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EvidencePrioritizer:
    """Evaluate evidence importance and generate a full-text reading queue."""

    def __init__(self, llm: object, audit: object,
                 model: str = "glm-5.1", task: str = "reasoner"):
        self.llm = llm
        self.audit = audit
        self.model = model
        self.task = task

    async def prioritize(self, evidence_pool: list[dict],
                         hypotheses: list[dict],
                         target: str = "") -> list[dict]:
        """Assess each evidence item and mark priority_fulltext if needed.

        Returns the evidence_pool with added fields:
            - priority_fulltext: bool
            - priority_reason: str
            - priority_score: int (0-10)

        Also returns a reading queue (sorted list of evidence IDs).
        """
        print(f"  [heartbeat] EvidencePrioritizer: starting on {len(evidence_pool)} evidence items...", flush=True)
        # Step 1: Identify evidence directly cited in causal chains
        cited_ids = set()
        for hyp in hypotheses[:5]:
            chain = hyp.get("causal_chain", {})
            if isinstance(chain, dict):
                if "mechanism_axes" in chain:
                    for axis in chain.get("mechanism_axes", []):
                        for step in axis.get("steps", []):
                            cited_ids.update(step.get("evidence_ids", []))
                else:
                    for link in chain.values():
                        if isinstance(link, dict):
                            cited_ids.update(link.get("evidence_ids", []))
            # Also check evidence_mapping
            eh = hyp.get("evidence_mapping", {})
            for pos in eh.get("positive_evidence", []):
                if isinstance(pos, dict) and pos.get("id"):
                    cited_ids.add(pos["id"])

        # Step 2: Quick heuristic pre-scoring (no LLM needed)
        pre_scored = []
        for ev in evidence_pool:
            score = 0
            reasons = []

            eid = ev.get("evidence_id", "")
            if eid in cited_ids:
                score += 4
                reasons.append("directly cited in causal chain")

            grade = ev.get("grade_score", 2)
            if grade >= 3:
                score += 3
                reasons.append(f"high GRADE quality ({grade})")

            ev_type = ev.get("evidence_type", "")
            if ev_type in ("RCT_human", "clinical_trial", "animal", "in_vitro"):
                score += 2
                reasons.append(f"experimental study ({ev_type})")

            abstract = ev.get("abstract_snippet", "")
            if len(abstract) > 500:
                score += 1
                reasons.append("substantial abstract available")

            # Check for quantitative data
            abstract_lower = abstract.lower()
            quant_markers = ["ic50", "ec50", "p<", "p =", "fold change", "ci95",
                             "hazard ratio", "odds ratio", "ki =", "kd ="]
            if any(m in abstract_lower for m in quant_markers):
                score += 2
                reasons.append("contains quantitative data")

            pre_scored.append({
                "evidence": ev,
                "score": min(score, 10),
                "reasons": reasons,
            })

        # Step 3: LLM-based refinement for borderline cases (score 3-7)
        borderline = [item for item in pre_scored if 3 <= item["score"] <= 7]
        if borderline and self.llm:
            print(f"  [heartbeat] EvidencePrioritizer: LLM-refining {len(borderline)} borderline items...", flush=True)
            await self._llm_refine(borderline, hypotheses[:5], target)
            print(f"  [heartbeat] EvidencePrioritizer: refinement done", flush=True)

        # Step 4: Mark priority_fulltext and build reading queue
        reading_queue = []
        for item in pre_scored:
            ev = item["evidence"]
            score = item["score"]
            reasons = item["reasons"]

            is_priority = score >= 4
            ev["priority_fulltext"] = is_priority
            ev["priority_reason"] = "; ".join(reasons) if reasons else "low priority"
            ev["priority_score"] = score

            if is_priority:
                reading_queue.append(ev)

        # Sort reading queue by priority score descending
        reading_queue.sort(key=lambda e: e.get("priority_score", 0), reverse=True)

        # Limit queue size to control cost (max 15 items)
        reading_queue = reading_queue[:15]

        self.audit.record("EvidencePrioritizer", "prioritize", "complete", {
            "n_pool": len(evidence_pool),
            "n_priority": len([e for e in evidence_pool if e.get("priority_fulltext")]),
            "n_queue": len(reading_queue),
            "n_cited": len(cited_ids),
        })

        print(f"  [heartbeat] EvidencePrioritizer: {len(reading_queue)} items in reading queue "
              f"(of {len(evidence_pool)} total, {len(cited_ids)} cited)", flush=True)

        return reading_queue

    async def _llm_refine(self, borderline_items: list[dict],
                          hypotheses: list[dict], target: str):
        """Use LLM to refine priority scores for borderline evidence.

        Evaluates each evidence item's relevance to the hypotheses.
        """
        sem = asyncio.Semaphore(3)  # Concurrency limit for glm-5.1
        tasks = [self._refine_one(item, hypotheses, target, sem) for item in borderline_items]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _refine_one(self, item: dict, hypotheses: list[dict],
                          target: str, sem: asyncio.Semaphore):
        """Refine priority score for a single evidence item."""
        async with sem:
            ev = item["evidence"]
            eid = ev.get("evidence_id", "")
            title = ev.get("title", "")[:200]
            abstract = ev.get("abstract_snippet", "")[:800]
            ev_type = ev.get("evidence_type", "")
            grade = ev.get("grade_score", 2)

            # Build hypothesis summary
            hyp_summaries = []
            for h in hypotheses[:3]:
                ind = h.get("indication", "")
                stmt = h.get("statement", "")[:150]
                hyp_summaries.append(f"- [{ind}] {stmt}")
            hyp_text = "\n".join(hyp_summaries)

            prompt = f"""## Task
Evaluate whether this evidence item is important enough to warrant full-text reading for verifying hypotheses.

## Evidence
- ID: {eid}
- Title: {title}
- Type: {ev_type}
- GRADE: {grade}
- Abstract (truncated): {abstract}

## Hypotheses
{hyp_text}

## Evaluation Criteria
Score 0-10 for full-text priority:
- 8-10: Directly cited in causal chain AND contains experimental data
- 5-7: Relevant to hypothesis mechanism, may contain supporting data
- 3-4: Tangentially related, background context only
- 0-2: Unlikely to add value beyond abstract

Consider:
1. Does the abstract mention the target ({target})?
2. Does it contain experimental validation (not just review/commentary)?
3. Is it directly relevant to any hypothesis indication?
4. Could full text reveal data that changes hypothesis assessment?

Return JSON: {{"score": int, "reason": "str"}}"""

            try:
                # Bound each LLM call so a hung connection under rate-limit or
                # server instability cannot
                # block this coroutine indefinitely — and because prioritize() only
                # printed a heartbeat at its very end, the whole run appeared frozen
                # after "Critic done" with zero progress output. Bounding each refine
                # call means a stuck item degrades to its heuristic score instead of
                # stalling the pipeline.
                result, _ = await asyncio.wait_for(
                    self.llm.structured([
                        {"role": "system", "content": "You are a literature triage specialist. Be conservative - only recommend full-text reading when the abstract suggests important mechanistic or experimental data."},
                        {"role": "user", "content": prompt},
                    ], max_tokens=256, task=self.task, temperature=0),
                    timeout=90,
                )
                if isinstance(result, dict) and not result.get("error"):
                    llm_score = int(result.get("score", item["score"]))
                    # Blend LLM score with heuristic score
                    item["score"] = round((llm_score + item["score"]) / 2)
                    llm_reason = result.get("reason", "")
                    if llm_reason:
                        item["reasons"].append(f"LLM: {llm_reason[:100]}")
            except asyncio.TimeoutError:
                # LLM call exceeded the per-item budget - keep heuristic score
                logger.debug(f"LLM refine timeout for {eid}")
                print(f"  [heartbeat] EvidencePrioritizer: LLM refine timeout for {eid}, using heuristic score", flush=True)
            except Exception as e:
                # LLM call failed (HTTP error, network error, etc.) - keep heuristic score
                logger.debug(f"LLM refine failed for {eid}: {e}")
                print(f"  [heartbeat] EvidencePrioritizer: LLM refine skipped for {eid} ({type(e).__name__})", flush=True)
