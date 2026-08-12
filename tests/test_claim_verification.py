import asyncio
import json
import unittest

from dp_indicator.agents.claim_verifier import (
    ClaimVerifier,
    aggregate_bridge_claims,
    aggregate_claim_verdict,
    aggregate_parent_claims,
    build_claim_candidates,
    merge_bridge_grounding,
    quote_is_in_source,
    recheck_sanitized_hypotheses,
    validate_grounding_record,
)
from dp_indicator.agents.evidence_verifier import EvidenceVerifier
from dp_indicator.agents.evidence_verifier import (
    compute_claim_score_adjustments,
    compute_gap_acceptance,
)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def structured(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        response = self.responses.pop(0)
        prompt = messages[-1]["content"]
        if "Claims:\n" in prompt and isinstance(response, dict):
            claims_text = prompt.split("Claims:\n", 1)[1].split(
                "\n\nAllowed verdicts",
                1,
            )[0]
            claims = json.loads(claims_text)
            for index, item in enumerate(response.get("items", [])):
                if index >= len(claims):
                    break
                if item.get("claim_id") in {
                    None,
                    "",
                    "ignored-by-contract",
                }:
                    item["claim_id"] = claims[index]["claim_id"]
        return response, {"total_tokens": 1}


class FakeAudit:
    def record(self, *args, **kwargs):
        return None


class FakeCache:
    def __init__(self, metadata):
        self.metadata = metadata

    async def fetch_batch(self, evidence_ids):
        return {
            evidence_id: self.metadata[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in self.metadata
        }

    async def close(self):
        return None


class HangingLLM:
    async def structured(self, messages, **kwargs):
        await asyncio.sleep(10)


class ConcurrentLLM:
    def __init__(self, candidate_id):
        self.candidate_id = candidate_id
        self.active = 0
        self.max_active = 0

    async def structured(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "Split each biomedical statement" in prompt:
            return {
                "items": [
                    {
                        "candidate_id": self.candidate_id,
                        "claims": ["Blockade reduces cytokines."],
                    }
                ]
            }, {}
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.02)
            claims_text = prompt.split("Claims:\n", 1)[1].split(
                "\n\nAllowed verdicts",
                1,
            )[0]
            claims = json.loads(claims_text)
            return {
                "items": [
                    {
                        "claim_id": claim["claim_id"],
                        "verdict": "supported",
                        "quote": "Blockade reduces cytokines",
                        "reason": "Exact support.",
                    }
                    for claim in claims
                ]
            }, {}
        finally:
            self.active -= 1


class SlowBatchLLM:
    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def structured(self, messages, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.02)
            return {"items": []}, {}
        finally:
            self.active -= 1


class GroundingBatchLLM:
    def __init__(self, omit_after_first=False):
        self.batch_sizes = []
        self.calls = 0
        self.omit_after_first = omit_after_first

    async def structured(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        claims_text = prompt.split("Claims:\n", 1)[1].split(
            "\n\nAllowed verdicts",
            1,
        )[0]
        claims = json.loads(claims_text)
        self.batch_sizes.append(len(claims))
        self.calls += 1
        selected = claims
        if self.omit_after_first and self.calls == 1:
            selected = claims[:1]
        return {
            "items": [
                {
                    "claim_id": claim["claim_id"],
                    "verdict": "supported",
                    "quote": "exact source words",
                    "reason": "Exact support.",
                }
                for claim in selected
            ]
        }, {}


class FakeGapRetriever:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def retrieve(self, claims, hypotheses, target):
        self.calls.append((claims, hypotheses, target))
        if self.error:
            raise self.error
        return self.result


class ClaimContractTests(unittest.TestCase):
    def test_gap_acceptance_requires_retained_verified_final_claims(self):
        before = [{"claim_id": f"A{index}"} for index in range(13)]
        claims = [
            {
                "claim_id": "A0",
                "verdict": "supported",
                "evidence_results": [{
                    "evidence_id": "PMID:1",
                    "verdict": "supported",
                    "quote_verified": True,
                    "evidence_role": "bridge_evidence",
                }],
            },
            {
                "claim_id": "A1",
                "verdict": "supported",
                "evidence_results": [{
                    "evidence_id": "PMID:2",
                    "verdict": "supported",
                    "quote_verified": True,
                    "evidence_role": "bridge_evidence",
                }],
            },
            {
                "claim_id": "A2",
                "verdict": "unverifiable",
                "evidence_results": [{
                    "evidence_id": "PMID:2",
                    "verdict": "supported",
                    "quote_verified": True,
                    "evidence_role": "bridge_evidence",
                }],
            },
        ]
        retrieval = {"groups": [{
            "selected": {
                "decision": "retain_partial",
                "evidence_ids": ["PMID:1"],
            },
        }]}

        low = compute_gap_acceptance(before, claims, retrieval)
        self.assertEqual(low["uncited_atomic_before"], 13)
        self.assertEqual(low["supported_or_partial_after"], 1)
        self.assertAlmostEqual(low["coverage_gain"], 1 / 13)
        self.assertFalse(low["acceptance_passed"])

        retrieval["groups"][0]["selected"]["evidence_ids"] = [
            "PMID:1", "PMID:2"
        ]
        boundary = compute_gap_acceptance(
            before[:10], claims + [{
                "claim_id": "A3",
                "verdict": "partial",
                "evidence_results": [{
                    "evidence_id": "PMID:2",
                    "verdict": "partial",
                    "quote_verified": True,
                    "evidence_role": "bridge_evidence",
                }],
            }],
            retrieval,
        )
        self.assertAlmostEqual(boundary["coverage_gain"], 0.3)
        self.assertTrue(boundary["acceptance_passed"])
        self.assertEqual(
            compute_gap_acceptance([], claims, retrieval)["coverage_gain"],
            0.0,
        )
    def setUp(self):
        self.hypotheses = [
            {
                "hypothesis_id": "H1",
                "indication": "Example disease",
                "causal_chain": {
                    "mechanism_axes": [
                        {
                            "axis_name": "primary",
                            "steps": [
                                {
                                    "layer": "L1",
                                    "mechanism": (
                                        "Kv1.3 blockade reduces inflammatory "
                                        "cytokine release and tissue injury."
                                    ),
                                    "status": "supported",
                                    "evidence_ids": ["PMID:1"],
                                }
                            ],
                        }
                    ]
                },
                "evidence_mapping": {
                    "positive_evidence": [
                        {
                            "id": "PMID:1",
                            "rationale": "The study reports lower cytokine release.",
                        }
                    ],
                    "indirect_evidence": [],
                    "contradicting_evidence": [
                        {
                            "id": "PMID:2",
                            "rationale": "Blockade did not improve tissue injury.",
                        }
                    ],
                },
            }
        ]

    def test_candidates_have_stable_ids_and_origin_paths(self):
        first = build_claim_candidates(self.hypotheses)
        second = build_claim_candidates(self.hypotheses)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(first[0]["hypothesis_id"], "H1")
        self.assertEqual(first[0]["origin"]["kind"], "causal_step")
        self.assertEqual(first[0]["expected_relation"], "support")
        self.assertEqual(first[2]["expected_relation"], "contradict")
        self.assertTrue(first[0]["claim_id"].startswith("CLM-"))

    def test_mapping_claims_are_capped_per_hypothesis(self):
        hypothesis = self.hypotheses[0]
        mapping = hypothesis["evidence_mapping"]
        for bucket in (
            "positive_evidence",
            "indirect_evidence",
            "contradicting_evidence",
        ):
            mapping[bucket] = [
                {
                    "id": f"{bucket}:{index}",
                    "rationale": f"{bucket} rationale {index}",
                }
                for index in range(20)
            ]

        candidates = build_claim_candidates([hypothesis])
        by_bucket = {}
        for candidate in candidates:
            bucket = candidate["origin"].get("bucket", "causal_step")
            by_bucket[bucket] = by_bucket.get(bucket, 0) + 1

        self.assertEqual(by_bucket["causal_step"], 1)
        self.assertEqual(by_bucket["positive_evidence"], 12)
        self.assertEqual(by_bucket["indirect_evidence"], 3)
        self.assertEqual(by_bucket["contradicting_evidence"], 5)

    def test_quote_must_be_present_after_whitespace_normalization(self):
        source = "Kv1.3 blockade reduced\n inflammatory cytokine release."

        self.assertTrue(
            quote_is_in_source(
                "reduced inflammatory cytokine release",
                source,
            )
        )
        self.assertFalse(
            quote_is_in_source("prevented all tissue injury", source)
        )

    def test_fabricated_quote_cannot_remain_supported(self):
        record = validate_grounding_record(
            {
                "evidence_id": "PMID:1",
                "verdict": "supported",
                "quote": "prevented all tissue injury",
                "reason": "Claimed by reviewer",
            },
            "Kv1.3 blockade reduced inflammatory cytokine release.",
        )

        self.assertEqual(record["verdict"], "unsupported")
        self.assertFalse(record["quote_verified"])
        self.assertEqual(record["reason"], "quote_not_found_in_source")

    def test_missing_source_is_explicitly_unverifiable(self):
        record = validate_grounding_record(
            {
                "evidence_id": "PMID:1",
                "verdict": "supported",
                "quote": "some quote",
            },
            "",
        )

        self.assertEqual(record["verdict"], "unverifiable")
        self.assertFalse(record["quote_verified"])

    def test_claim_aggregation_preserves_mixed_evidence(self):
        verdict = aggregate_claim_verdict(
            [
                {"verdict": "supported"},
                {"verdict": "contradicted"},
            ]
        )

        self.assertEqual(verdict, "mixed")

    def test_bridge_aggregate_uses_selected_set_union_coverage(self):
        claims = [
            {
                "claim_id": f"A{index}",
                "parent_claim_id": "P1",
                "hypothesis_id": "H1",
                "origin": {
                    "kind": "causal_step",
                    "axis_index": 0,
                    "step_index": 0,
                },
                "verdict": "unverifiable",
                "evidence_results": [],
            }
            for index in range(1, 4)
        ]
        retrieval = {
            "groups": [{
                "hypothesis_id": "H1",
                "parent_claim_id": "P1",
                "origin": claims[0]["origin"],
                "selected": {
                    "evidence_ids": ["PMID:1", "PMID:2"],
                    "coverage": 2 / 3,
                    "decision": "retain_supported",
                    "verified_spans": [
                        {
                            "claim_id": "A1",
                            "evidence_id": "PMID:1",
                            "quote": "supports A1",
                            "quote_verified": True,
                        },
                        {
                            "claim_id": "A2",
                            "evidence_id": "PMID:2",
                            "quote": "supports A2",
                            "quote_verified": True,
                        },
                        {
                            "claim_id": "A2",
                            "evidence_id": "PMID:2",
                            "quote": "supports A2",
                            "quote_verified": True,
                        },
                    ],
                },
            }],
        }

        aggregates = aggregate_bridge_claims(claims, retrieval)

        self.assertEqual(len(aggregates), 1)
        aggregate = aggregates[0]
        self.assertEqual(aggregate["aggregate_kind"], "bridge_set")
        self.assertEqual(aggregate["evidence_ids"], ["PMID:1", "PMID:2"])
        self.assertEqual(aggregate["coverage"], 0.666667)
        self.assertEqual(aggregate["decision"], "retain_supported")
        self.assertEqual(
            [span["claim_id"] for span in aggregate["verified_spans"]],
            ["A1", "A2"],
        )

    def test_empty_bridge_retrieval_does_not_change_direct_aggregates(self):
        claims = self._atomic_claims(["supported", "unsupported"])
        before = aggregate_parent_claims(claims)

        self.assertEqual(aggregate_bridge_claims(claims, {"groups": []}), [])
        self.assertEqual(aggregate_parent_claims(claims), before)

    def test_bridge_merge_updates_only_unverifiable_atomic_claims(self):
        claims = [
            {
                "claim_id": "A1",
                "parent_claim_id": "P1",
                "hypothesis_id": "H1",
                "verdict": "unverifiable",
                "evidence_results": [],
            },
            {
                "claim_id": "A2",
                "parent_claim_id": "P1",
                "hypothesis_id": "H1",
                "verdict": "unsupported",
                "evidence_results": [{
                    "evidence_id": "PMID:direct",
                    "verdict": "unsupported",
                }],
            },
        ]
        retrieval = {"groups": [{
            "selected": {
                "decision": "retain_supported",
                "evidence_ids": ["PMID:bridge"],
            },
            "grounded_claims": [
                {
                    "claim_id": "A1",
                    "verdict": "supported",
                    "evidence_results": [{
                        "evidence_id": "PMID:bridge",
                        "verdict": "supported",
                        "quote": "bridge quote",
                        "quote_verified": True,
                    }],
                },
                {
                    "claim_id": "A2",
                    "verdict": "supported",
                    "evidence_results": [{
                        "evidence_id": "PMID:bridge",
                        "verdict": "supported",
                        "quote": "bridge quote",
                        "quote_verified": True,
                    }],
                },
            ],
        }]}

        merge_bridge_grounding(claims, retrieval)

        self.assertEqual(claims[0]["verdict"], "supported")
        self.assertEqual(len(claims[0]["evidence_results"]), 1)
        self.assertEqual(claims[1]["verdict"], "unsupported")
        self.assertEqual(
            claims[1]["evidence_results"][0]["evidence_id"],
            "PMID:direct",
        )

    def test_remove_bridge_group_cannot_change_atomic_verdict_or_results(self):
        claims = [{
            "claim_id": "A1",
            "parent_claim_id": "P1",
            "hypothesis_id": "H1",
            "verdict": "unverifiable",
            "evidence_results": [],
        }]
        retrieval = {"groups": [{
            "selected": {
                "decision": "remove",
                "evidence_ids": ["PMID:bridge"],
            },
            "grounded_claims": [{
                "claim_id": "A1",
                "evidence_results": [{
                    "evidence_id": "PMID:bridge",
                    "verdict": "supported",
                    "quote": "bridge quote",
                    "quote_verified": True,
                }],
            }],
        }]}

        merge_bridge_grounding(claims, retrieval)

        self.assertEqual(claims[0]["verdict"], "unverifiable")
        self.assertEqual(claims[0]["evidence_results"], [])

    def test_parent_aggregation_retains_sixty_percent_coverage(self):
        claims = self._atomic_claims(
            ["supported", "supported", "unsupported"]
        )

        aggregate = aggregate_parent_claims(claims)[0]

        self.assertAlmostEqual(aggregate["coverage"], 2 / 3)
        self.assertEqual(aggregate["decision"], "retain_supported")
        self.assertEqual(len(aggregate["verified_spans"]), 2)

    def test_parent_aggregation_retains_thirty_percent_as_partial(self):
        claims = self._atomic_claims(
            ["partial", "partial", "unsupported"]
        )

        aggregate = aggregate_parent_claims(claims)[0]

        self.assertAlmostEqual(aggregate["coverage"], 1 / 3)
        self.assertEqual(aggregate["decision"], "retain_partial")

    def test_parent_aggregation_tracks_unverifiable_separately(self):
        claims = self._atomic_claims(["unverifiable", "unverifiable"])

        aggregate = aggregate_parent_claims(claims)[0]

        self.assertIsNone(aggregate["coverage"])
        self.assertEqual(aggregate["unverifiable_ratio"], 1.0)
        self.assertEqual(aggregate["decision"], "unverifiable")

    def test_parent_aggregation_removes_explicit_contradiction(self):
        claims = self._atomic_claims(["supported", "contradicted"])

        aggregate = aggregate_parent_claims(claims)[0]

        self.assertTrue(aggregate["has_contradiction"])
        self.assertEqual(aggregate["decision"], "remove")

    @staticmethod
    def _atomic_claims(verdicts):
        claims = []
        for index, verdict in enumerate(verdicts):
            claims.append(
                {
                    "claim_id": f"CLM-{index}",
                    "parent_claim_id": "PARENT-1",
                    "hypothesis_id": "H1",
                    "origin": {
                        "kind": "causal_step",
                        "axis_index": 0,
                        "step_index": 0,
                        "layer": "L1",
                    },
                    "expected_relation": "support",
                    "evidence_results": [
                        {
                            "evidence_id": "PMID:1",
                            "verdict": verdict,
                            "quote": f"quote {index}",
                            "quote_verified": verdict
                            in {"supported", "partial"},
                        }
                    ],
                }
            )
        return claims

    def test_recheck_downgrades_supported_step_without_verified_span(self):
        hypotheses = self.hypotheses
        checked, audit = recheck_sanitized_hypotheses(hypotheses)

        step = checked[0]["causal_chain"]["mechanism_axes"][0]["steps"][0]
        self.assertEqual(step["status"], "inferred")
        self.assertEqual(hypotheses[0]["causal_chain"]["mechanism_axes"][0]["steps"][0]["status"], "supported")
        self.assertEqual(audit[0]["action"], "downgrade_step")

    def test_offline_grounding_challenge_set_metrics(self):
        cases = [
            {
                "label": "exact_support",
                "source": "Blockade reduced cytokine release.",
                "record": {
                    "verdict": "supported",
                    "quote": "reduced cytokine release",
                },
                "expected": "supported",
                "is_error": False,
            },
            {
                "label": "partial_support",
                "source": "Blockade reduced cytokine release.",
                "record": {
                    "verdict": "partial",
                    "quote": "reduced cytokine release",
                },
                "expected": "partial",
                "is_error": False,
            },
            {
                "label": "fabricated_quote",
                "source": "Blockade reduced cytokine release.",
                "record": {
                    "verdict": "supported",
                    "quote": "eliminated all tissue injury",
                },
                "expected": "unsupported",
                "is_error": True,
            },
            {
                "label": "missing_source",
                "source": "",
                "record": {
                    "verdict": "supported",
                    "quote": "some finding",
                },
                "expected": "unverifiable",
                "is_error": True,
            },
            {
                "label": "unsupported_claim",
                "source": "No treatment effect was observed.",
                "record": {
                    "verdict": "unsupported",
                    "quote": "",
                },
                "expected": "unsupported",
                "is_error": True,
            },
            {
                "label": "exact_contradiction",
                "source": "Blockade increased cytokine release.",
                "record": {
                    "verdict": "contradicted",
                    "quote": "increased cytokine release",
                },
                "expected": "contradicted",
                "is_error": True,
            },
        ]

        checked = []
        for case in cases:
            with self.subTest(case=case["label"]):
                result = validate_grounding_record(
                    case["record"],
                    case["source"],
                )
                self.assertEqual(result["verdict"], case["expected"])
                checked.append((case, result))

        errors = [item for item in checked if item[0]["is_error"]]
        residual_errors = [
            item for item in errors if item[1]["verdict"] == "supported"
        ]
        valid = [item for item in checked if not item[0]["is_error"]]
        false_deletions = [
            item
            for item in valid
            if item[1]["verdict"] in {"unsupported", "unverifiable"}
        ]
        self.assertEqual(len(errors), 4)
        self.assertEqual(len(residual_errors), 0)
        self.assertEqual(len(false_deletions), 0)


class ClaimVerifierTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.hypotheses = [
            {
                "hypothesis_id": "H1",
                "indication": "Example disease",
                "causal_chain": {
                    "mechanism_axes": [
                        {
                            "axis_name": "primary",
                            "steps": [
                                {
                                    "layer": "L1",
                                    "mechanism": (
                                        "Blockade reduces cytokines and "
                                        "prevents tissue injury."
                                    ),
                                    "status": "supported",
                                    "evidence_ids": ["PMID:1"],
                                }
                            ],
                        }
                    ]
                },
                "evidence_mapping": {
                    "positive_evidence": [],
                    "indirect_evidence": [],
                    "contradicting_evidence": [],
                },
            }
        ]

    async def test_verify_decomposes_then_binds_exact_source_quotes(self):
        llm = FakeLLM(
            [
                {
                    "items": [
                        {
                            "candidate_id": build_claim_candidates(
                                self.hypotheses
                            )[0]["claim_id"],
                            "claims": [
                                "Blockade reduces cytokines.",
                                "Blockade prevents tissue injury.",
                            ],
                        }
                    ]
                },
                {
                    "items": [
                        {
                            "claim_id": "ignored-by-contract",
                            "verdict": "supported",
                            "quote": "Blockade reduced cytokines",
                            "reason": "Directly stated.",
                        },
                        {
                            "claim_id": "ignored-by-contract",
                            "verdict": "supported",
                            "quote": "prevented all tissue injury",
                            "reason": "Not actually stated.",
                        },
                    ]
                },
            ]
        )
        verifier = ClaimVerifier(llm=llm, task="reasoner")

        result = await verifier.verify(
            self.hypotheses,
            {
                "PMID:1": {
                    "title": "Study",
                    "abstract": (
                        "Blockade reduced cytokines in treated cells. "
                        "Tissue injury was not measured."
                    ),
                }
            },
            target="Kv1.3",
        )

        self.assertEqual(len(result["claims"]), 2)
        self.assertNotEqual(
            result["claims"][0]["claim_id"],
            result["claims"][1]["claim_id"],
        )
        self.assertEqual(result["claims"][0]["verdict"], "supported")
        self.assertEqual(result["claims"][1]["verdict"], "unsupported")
        self.assertTrue(
            result["claims"][0]["evidence_results"][0]["quote_verified"]
        )
        self.assertFalse(
            result["claims"][1]["evidence_results"][0]["quote_verified"]
        )

    async def test_ground_existing_claims_skips_decomposition_and_copies_input(self):
        claims = [{
            "claim_id": "A1",
            "parent_claim_id": "P1",
            "hypothesis_id": "H1",
            "text": "Treatment improves outcome",
            "expected_relation": "support",
            "evidence_ids": ["PMID:1"],
            "origin": {"kind": "causal_step"},
        }]
        llm = GroundingBatchLLM()
        verifier = ClaimVerifier(llm=llm)

        grounded = await verifier.ground_existing_claims(
            claims,
            {"PMID:1": {
                "title": "Study",
                "abstract": "The study reports exact source words.",
            }},
            "Kv1.3",
        )

        self.assertEqual(len(grounded), 1)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(grounded[0]["verdict"], "supported")
        self.assertNotIn("evidence_results", claims[0])

    async def test_missing_source_is_unverifiable_without_grounding_call(self):
        llm = FakeLLM([{"items": []}])
        verifier = ClaimVerifier(llm=llm)

        result = await verifier.verify(
            self.hypotheses,
            {},
            target="Kv1.3",
        )

        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(result["claims"][0]["verdict"], "unverifiable")
        self.assertEqual(
            result["claims"][0]["evidence_results"][0]["reason"],
            "source_unavailable",
        )

    async def test_decomposition_is_chunked_to_avoid_oversized_outputs(self):
        hypotheses = [
            {
                "hypothesis_id": "H1",
                "causal_chain": {
                    "mechanism_axes": [
                        {
                            "steps": [
                                {
                                    "layer": f"L{index}",
                                    "mechanism": f"Atomic claim {index}.",
                                    "evidence_ids": [],
                                }
                                for index in range(21)
                            ]
                        }
                    ]
                },
                "evidence_mapping": {},
            }
        ]
        candidates = build_claim_candidates(hypotheses)
        llm = FakeLLM(
            [
                {
                    "items": [
                        {
                            "candidate_id": candidate["claim_id"],
                            "claims": [candidate["text"]],
                        }
                        for candidate in candidates[:20]
                    ]
                },
                {
                    "items": [
                        {
                            "candidate_id": candidates[20]["claim_id"],
                            "claims": [candidates[20]["text"]],
                        }
                    ]
                },
            ]
        )
        verifier = ClaimVerifier(llm=llm)

        result = await verifier.verify(hypotheses, {}, target="Kv1.3")

        self.assertEqual(len(result["claims"]), 21)
        self.assertEqual(len(llm.calls), 2)

    async def test_decomposition_timeout_falls_back_to_original_claim(self):
        verifier = ClaimVerifier(
            llm=HangingLLM(),
            call_timeout_seconds=0.01,
        )

        result = await verifier.verify(
            self.hypotheses,
            {},
            target="Kv1.3",
        )

        self.assertEqual(len(result["claims"]), 1)
        self.assertEqual(
            result["claims"][0]["text"],
            self.hypotheses[0]["causal_chain"]["mechanism_axes"][0][
                "steps"
            ][0]["mechanism"],
        )
        self.assertEqual(result["claims"][0]["verdict"], "unverifiable")

    async def test_evidence_verifier_merges_injected_gap_retrieval(self):
        hypotheses = self.hypotheses
        step = hypotheses[0]["causal_chain"]["mechanism_axes"][0]["steps"][0]
        step["evidence_ids"] = []
        candidate_id = build_claim_candidates(hypotheses)[0]["claim_id"]
        llm = FakeLLM([{
            "items": [{
                "candidate_id": candidate_id,
                "claims": ["Blockade reduces cytokines."],
            }],
        }])
        retrieval = {
            "groups": [{
                "hypothesis_id": "H1",
                "parent_claim_id": candidate_id,
                "origin": {
                    "kind": "causal_step",
                    "axis_index": 0,
                    "step_index": 0,
                },
                "grounded_claims": [],
                "selected": {
                    "evidence_ids": ["PMID:9"],
                    "coverage": 0.6,
                    "decision": "retain_supported",
                    "verified_spans": [{
                        "claim_id": "A1",
                        "evidence_id": "PMID:9",
                        "quote": "exact bridge words",
                        "quote_verified": True,
                    }],
                },
            }],
            "evidence": [{
                "evidence_id": "PMID:9",
                "title": "Bridge study",
                "abstract_snippet": "exact bridge words",
                "source_metadata": {"pmid": "9"},
            }],
            "summary": {"resolved_parent_steps": 1, "errors": []},
        }
        gap_retriever = FakeGapRetriever(retrieval)
        verifier = EvidenceVerifier(
            llm=llm,
            audit=FakeAudit(),
            cache_client=FakeCache({}),
            gap_retriever=gap_retriever,
        )

        async def no_v4(*args, **kwargs):
            return {}

        verifier._v4_check_indication_relevance = no_v4
        result = await verifier.verify(hypotheses, [], target="Kv1.3")

        report = result["verification_report"]
        self.assertEqual(len(gap_retriever.calls), 1)
        self.assertEqual(report["gap_retrieval"], retrieval)
        self.assertEqual(
            report["claim_grounding"]["bridge_aggregates"][0][
                "aggregate_kind"
            ],
            "bridge_set",
        )
        self.assertEqual(
            result["verified_hypotheses"][0]["causal_chain"][
                "mechanism_axes"
            ][0]["steps"][0]["evidence_ids"],
            ["PMID:9"],
        )

    async def test_disabled_gap_retrieval_preserves_empty_legacy_path(self):
        hypotheses = self.hypotheses
        hypotheses[0]["causal_chain"]["mechanism_axes"][0]["steps"][0][
            "evidence_ids"
        ] = []
        candidate_id = build_claim_candidates(hypotheses)[0]["claim_id"]
        llm = FakeLLM([{
            "items": [{
                "candidate_id": candidate_id,
                "claims": ["Blockade reduces cytokines."],
            }],
        }])
        gap_retriever = FakeGapRetriever({"unexpected": True})
        verifier = EvidenceVerifier(
            llm=llm,
            audit=FakeAudit(),
            cache_client=FakeCache({}),
            gap_retriever=gap_retriever,
            enable_gap_retrieval=False,
        )

        async def no_v4(*args, **kwargs):
            return {}

        verifier._v4_check_indication_relevance = no_v4
        result = await verifier.verify(hypotheses, [], target="Kv1.3")

        self.assertEqual(gap_retriever.calls, [])
        self.assertEqual(
            result["verification_report"]["gap_retrieval"],
            {
                "groups": [],
                "evidence": [],
                "summary": {
                    "uncited_atomic_before": 1,
                    "supported_or_partial_after": 0,
                    "coverage_gain": 0.0,
                    "acceptance_threshold": 0.30,
                    "acceptance_passed": False,
                },
            },
        )
        self.assertEqual(
            result["verification_report"]["claim_grounding"][
                "bridge_aggregates"
            ],
            [],
        )

    async def test_gap_retrieval_exception_is_recorded_and_verification_continues(self):
        hypotheses = self.hypotheses
        hypotheses[0]["causal_chain"]["mechanism_axes"][0]["steps"][0][
            "evidence_ids"
        ] = []
        candidate_id = build_claim_candidates(hypotheses)[0]["claim_id"]
        llm = FakeLLM([{
            "items": [{
                "candidate_id": candidate_id,
                "claims": ["Blockade reduces cytokines."],
            }],
        }])
        verifier = EvidenceVerifier(
            llm=llm,
            audit=FakeAudit(),
            cache_client=FakeCache({}),
            gap_retriever=FakeGapRetriever(error=RuntimeError("offline")),
        )

        async def no_v4(*args, **kwargs):
            return {}

        verifier._v4_check_indication_relevance = no_v4
        result = await verifier.verify(hypotheses, [], target="Kv1.3")

        errors = result["verification_report"]["gap_retrieval"]["summary"][
            "errors"
        ]
        self.assertEqual(len(errors), 1)
        self.assertIn("offline", errors[0])
        self.assertIn("summary", result)

    async def test_decomposition_batches_run_with_bounded_concurrency(self):
        hypotheses = [
            {
                "hypothesis_id": "H1",
                "causal_chain": {
                    "mechanism_axes": [
                        {
                            "steps": [
                                {
                                    "layer": f"L{index}",
                                    "mechanism": f"Claim {index}.",
                                    "evidence_ids": [],
                                }
                                for index in range(41)
                            ]
                        }
                    ]
                },
                "evidence_mapping": {},
            }
        ]
        llm = SlowBatchLLM()
        verifier = ClaimVerifier(
            llm=llm,
            call_timeout_seconds=1,
            max_concurrency=3,
        )

        result = await verifier.verify(hypotheses, {}, target="Kv1.3")

        self.assertEqual(len(result["claims"]), 41)
        self.assertEqual(llm.max_active, 3)

    async def test_grounding_requests_run_with_bounded_concurrency(self):
        hypotheses = self.hypotheses
        hypotheses[0]["causal_chain"]["mechanism_axes"][0]["steps"][0][
            "evidence_ids"
        ] = ["PMID:1", "PMID:2", "PMID:3", "PMID:4"]
        candidate_id = build_claim_candidates(hypotheses)[0]["claim_id"]
        llm = ConcurrentLLM(candidate_id)
        verifier = ClaimVerifier(
            llm=llm,
            call_timeout_seconds=1,
            max_concurrency=3,
        )
        metadata = {
            evidence_id: {
                "title": "Study",
                "abstract": "Blockade reduces cytokines in treated cells.",
            }
            for evidence_id in [
                "PMID:1",
                "PMID:2",
                "PMID:3",
                "PMID:4",
            ]
        }

        result = await verifier.verify(
            hypotheses,
            metadata,
            target="Kv1.3",
        )

        self.assertEqual(result["claims"][0]["verdict"], "supported")
        self.assertEqual(llm.max_active, 3)

    async def test_grounding_chunks_each_source_to_six_claims(self):
        claims = [
            {
                "claim_id": f"CLM-{index}",
                "parent_claim_id": "PARENT-1",
                "hypothesis_id": "H1",
                "origin": {"kind": "causal_step"},
                "text": f"Claim {index}",
                "expected_relation": "support",
                "evidence_ids": ["PMID:1"],
            }
            for index in range(7)
        ]
        llm = GroundingBatchLLM()
        verifier = ClaimVerifier(llm=llm, call_timeout_seconds=1)

        await verifier._ground_claims(
            claims,
            {
                "PMID:1": {
                    "title": "Study",
                    "abstract": "The article contains exact source words.",
                }
            },
            target="Kv1.3",
        )

        self.assertEqual(sorted(llm.batch_sizes), [1, 6])
        self.assertTrue(
            all(claim["verdict"] == "supported" for claim in claims)
        )

    async def test_grounding_retries_only_claim_ids_missing_from_response(self):
        claims = [
            {
                "claim_id": f"CLM-{index}",
                "parent_claim_id": "PARENT-1",
                "hypothesis_id": "H1",
                "origin": {"kind": "causal_step"},
                "text": f"Claim {index}",
                "expected_relation": "support",
                "evidence_ids": ["PMID:1"],
            }
            for index in range(2)
        ]
        llm = GroundingBatchLLM(omit_after_first=True)
        verifier = ClaimVerifier(llm=llm, call_timeout_seconds=1)

        await verifier._ground_claims(
            claims,
            {
                "PMID:1": {
                    "title": "Study",
                    "abstract": "The article contains exact source words.",
                }
            },
            target="Kv1.3",
        )

        self.assertEqual(llm.batch_sizes, [2, 1])
        self.assertEqual(llm.calls, 2)
        self.assertTrue(
            all(claim["verdict"] == "supported" for claim in claims)
        )
        self.assertFalse(
            any(
                result["reason"] == "grounding_result_missing"
                for claim in claims
                for result in claim["evidence_results"]
            )
        )

    async def test_evidence_verifier_uses_claim_grounding_for_v2_v3(self):
        candidate_id = build_claim_candidates(self.hypotheses)[0]["claim_id"]
        llm = FakeLLM(
            [
                {
                    "items": [
                        {
                            "candidate_id": candidate_id,
                            "claims": ["Blockade reduces cytokines."],
                        }
                    ]
                },
                {
                    "items": [
                        {
                            "verdict": "supported",
                            "quote": "Blockade reduces cytokines",
                            "reason": "Exact support.",
                        }
                    ]
                },
            ]
        )
        metadata = {
            "PMID:1": {
                "title": "Study",
                "abstract": "Blockade reduces cytokines in treated cells.",
            }
        }
        verifier = EvidenceVerifier(
            llm=llm,
            audit=FakeAudit(),
            cache_client=FakeCache(metadata),
        )

        async def no_v4(*args, **kwargs):
            return {}

        async def legacy_must_not_run(*args, **kwargs):
            raise AssertionError("legacy V2/V3 verifier was called")

        verifier._v4_check_indication_relevance = no_v4
        verifier._v2_check_citation_accuracy = legacy_must_not_run
        verifier._v3_check_description_accuracy = legacy_must_not_run

        result = await verifier.verify(
            self.hypotheses,
            [{"evidence_id": "PMID:1"}],
            target="Kv1.3",
        )

        report = result["verification_report"]
        self.assertEqual(
            report["claim_grounding"]["claims"][0]["verdict"],
            "supported",
        )
        self.assertEqual(report["v2_citation_accuracy"], {})
        self.assertEqual(report["v3_description_accuracy"], {})
        self.assertIn("Claim Grounding", result["summary"])
        self.assertIn("1/1 supported", result["summary"])

    def test_score_penalty_counts_each_claim_once(self):
        adjustments = compute_claim_score_adjustments(
            [
                {
                    "claim_id": "CLM-1",
                    "hypothesis_id": "H1",
                    "verdict": "unsupported",
                    "evidence_results": [
                        {"evidence_id": "PMID:1", "verdict": "unsupported"},
                        {"evidence_id": "PMID:2", "verdict": "unsupported"},
                    ],
                }
            ],
            {},
            [{"hypothesis_id": "H1"}],
        )

        self.assertEqual(adjustments, {"H1": -0.13})

    def test_score_penalty_is_proportional_not_issue_count_based(self):
        claims = [
            {
                "claim_id": f"CLM-{index}",
                "hypothesis_id": "H1",
                "verdict": "unsupported" if index < 2 else "supported",
            }
            for index in range(10)
        ]

        adjustments = compute_claim_score_adjustments(
            claims,
            {},
            [{"hypothesis_id": "H1"}],
        )

        self.assertAlmostEqual(adjustments["H1"], -0.026)

    async def test_evidence_pool_abstract_is_source_fallback(self):
        candidate_id = build_claim_candidates(self.hypotheses)[0]["claim_id"]
        llm = FakeLLM(
            [
                {
                    "items": [
                        {
                            "candidate_id": candidate_id,
                            "claims": ["Blockade reduces cytokines."],
                        }
                    ]
                },
                {
                    "items": [
                        {
                            "verdict": "supported",
                            "quote": "Blockade reduces cytokines",
                            "reason": "Found in retrieved abstract.",
                        }
                    ]
                },
            ]
        )
        verifier = EvidenceVerifier(
            llm=llm,
            audit=FakeAudit(),
            cache_client=FakeCache({}),
        )

        async def no_v4(*args, **kwargs):
            return {}

        verifier._v4_check_indication_relevance = no_v4
        result = await verifier.verify(
            self.hypotheses,
            [
                {
                    "evidence_id": "PMID:1",
                    "title": "Retrieved study",
                    "abstract_snippet": (
                        "Blockade reduces cytokines in treated cells."
                    ),
                }
            ],
            target="Kv1.3",
        )

        claim = result["verification_report"]["claim_grounding"]["claims"][0]
        self.assertEqual(claim["verdict"], "supported")
        self.assertEqual(len(llm.calls), 2)


if __name__ == "__main__":
    unittest.main()
