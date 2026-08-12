import asyncio
import json
import unittest

from dp_indicator.agents.claim_verifier import ClaimVerifier
from dp_indicator.agents.gap_evidence_retriever import (
    GapEvidenceRetriever,
    collect_uncited_parent_groups,
    deduplicate_candidates,
    fallback_query,
    select_bridge_set,
)


class GapGroupingTests(unittest.TestCase):
    def test_groups_only_unverifiable_claims_without_evidence_results(self):
        claims = [
            {
                "claim_id": "A1",
                "parent_claim_id": "P1",
                "hypothesis_id": "H1",
                "verdict": "unverifiable",
                "evidence_results": [],
                "text": "Outcome improves",
                "origin": {"kind": "causal_step", "axis_index": 0, "step_index": 4},
            },
            {
                "claim_id": "A2",
                "parent_claim_id": "P2",
                "hypothesis_id": "H1",
                "verdict": "unsupported",
                "evidence_results": [],
                "text": "Unsupported mechanism",
                "origin": {"kind": "causal_step", "axis_index": 0, "step_index": 1},
            },
        ]

        groups = collect_uncited_parent_groups(claims)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["parent_claim_id"], "P1")
        self.assertEqual([c["claim_id"] for c in groups[0]["claims"]], ["A1"])

    def test_group_limit_is_deterministic(self):
        claims = [
            {
                "claim_id": f"A{i}",
                "parent_claim_id": f"P{i}",
                "hypothesis_id": "H1",
                "verdict": "unverifiable",
                "evidence_results": [],
                "text": f"Claim {i}",
                "origin": {"kind": "causal_step", "axis_index": 0, "step_index": i},
            }
            for i in range(25)
        ]

        groups = collect_uncited_parent_groups(claims, max_groups=19)

        self.assertEqual(len(groups), 19)
        self.assertEqual(groups[0]["parent_claim_id"], "P0")
        self.assertEqual(groups[-1]["parent_claim_id"], "P18")

    def test_max_groups_cannot_exceed_global_cap(self):
        claims = [
            {
                "claim_id": f"A{i}",
                "parent_claim_id": f"P{i}",
                "hypothesis_id": "H1",
                "verdict": "unverifiable",
                "evidence_results": [],
                "text": f"Claim {i}",
                "origin": {"kind": "causal_step", "axis_index": 0, "step_index": i},
            }
            for i in range(25)
        ]

        groups = collect_uncited_parent_groups(claims, max_groups=100)

        self.assertEqual(len(groups), 19)

    def test_negative_max_groups_returns_empty(self):
        claims = [
            {
                "claim_id": "A1",
                "parent_claim_id": "P1",
                "hypothesis_id": "H1",
                "verdict": "unverifiable",
                "evidence_results": [],
                "text": "Claim",
                "origin": {"kind": "causal_step", "axis_index": 0, "step_index": 0},
            }
        ]

        groups = collect_uncited_parent_groups(claims, max_groups=-1)

        self.assertEqual(groups, [])


class BridgeSelectionTests(unittest.TestCase):
    def test_verified_spans_keep_only_supported_or_partial_verdicts(self):
        selected = select_bridge_set(
            [{"claim_id": "A1"}, {"claim_id": "A2"}],
            [{
                "evidence_id": "PMID:1",
                "results": [
                    {
                        "claim_id": "A1",
                        "verdict": "supported",
                        "quote": "supported quote",
                        "quote_verified": True,
                    },
                    {
                        "claim_id": "A2",
                        "verdict": "unsupported",
                        "quote": "unsupported quote",
                        "quote_verified": True,
                    },
                ],
            }],
        )

        self.assertEqual(selected["decision"], "retain_partial")
        self.assertEqual(
            selected["verified_spans"],
            [{
                "claim_id": "A1",
                "evidence_id": "PMID:1",
                "verdict": "supported",
                "quote": "supported quote",
                "quote_verified": True,
                "evidence_role": "bridge_evidence",
            }],
        )

    def test_deduplicates_by_pmid_doi_then_title(self):
        items = [
            {"evidence_id": "PMID:1", "title": "A Study", "source_metadata": {"doi": "10/x"}},
            {"evidence_id": "EPMC:1", "title": "A study.", "source_metadata": {"pmid": "1"}},
            {"evidence_id": "DOI:10/x", "title": "Different", "source_metadata": {"doi": "10/x"}},
        ]
        self.assertEqual(len(deduplicate_candidates(items)), 1)

    def test_transitive_deduplication_merges_keys_from_skipped_records(self):
        items = [
            {"evidence_id": "PMID:1", "title": "First", "source_metadata": {}},
            {
                "evidence_id": "EPMC:99",
                "title": "Second",
                "source_metadata": {"pmid": "1", "doi": "10/chain"},
            },
            {"evidence_id": "DOI:10/chain", "title": "Third", "source_metadata": {}},
        ]

        self.assertEqual(len(deduplicate_candidates(items)), 1)

    def test_connected_components_deduplicate_transitive_bridge(self):
        items = [
            {"evidence_id": "PMID:1", "title": "First", "source_metadata": {}},
            {"evidence_id": "DOI:10/y", "title": "Second", "source_metadata": {}},
            {
                "evidence_id": "EPMC:99",
                "title": "Bridge",
                "source_metadata": {"pmid": "1", "doi": "10/y"},
            },
        ]

        deduped = deduplicate_candidates(items)

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["evidence_id"], "PMID:1")

    def test_deduplication_is_independent_of_input_order(self):
        base = [
            {"evidence_id": "PMID:1", "title": "First", "source_metadata": {}},
            {"evidence_id": "DOI:10/y", "title": "Second", "source_metadata": {}},
            {
                "evidence_id": "EPMC:99",
                "title": "Bridge",
                "source_metadata": {"pmid": "1", "doi": "10/y"},
            },
        ]

        self.assertEqual(len(deduplicate_candidates(base)), 1)
        self.assertEqual(len(deduplicate_candidates(list(reversed(base)))), 1)

    def test_selects_papers_by_union_coverage_without_double_counting(self):
        claims = [
            {"claim_id": "A1", "weight": 1.0},
            {"claim_id": "A2", "weight": 1.0},
            {"claim_id": "A3", "weight": 1.0},
        ]
        candidates = [
            {
                "evidence_id": "PMID:1",
                "results": [
                    {
                        "claim_id": "A1",
                        "verdict": "supported",
                        "quote_verified": True,
                        "quote": "Outcome improves in treated group",
                    },
                    {"claim_id": "A2", "verdict": "unsupported", "quote_verified": False},
                ],
            },
            {
                "evidence_id": "PMID:2",
                "results": [
                    {
                        "claim_id": "A2",
                        "verdict": "supported",
                        "quote_verified": True,
                        "quote": "Mechanism confirmed in vitro",
                    },
                    {"claim_id": "A1", "verdict": "unsupported", "quote_verified": False},
                ],
            },
        ]

        selected = select_bridge_set(claims, candidates)

        self.assertEqual(selected["evidence_ids"], ["PMID:1", "PMID:2"])
        self.assertAlmostEqual(selected["coverage"], 2 / 3)
        self.assertEqual(selected["decision"], "retain_supported")
        self.assertGreater(len(selected["verified_spans"]), 0)
        for span in selected["verified_spans"]:
            self.assertTrue(span["quote"])
            self.assertEqual(span["evidence_role"], "bridge_evidence")

    def test_unverified_positive_verdict_does_not_contribute_to_coverage(self):
        claims = [{"claim_id": "A1", "weight": 1.0}]
        candidates = [{
            "evidence_id": "PMID:1",
            "results": [{
                "claim_id": "A1",
                "verdict": "supported",
                "quote_verified": False,
            }],
        }]

        selected = select_bridge_set(claims, candidates)

        self.assertEqual(selected["evidence_ids"], [])
        self.assertAlmostEqual(selected["coverage"], 0.0)
        self.assertEqual(selected["decision"], "remove")
        self.assertEqual(selected["verified_spans"], [])

    def test_coverage_threshold_at_sixty_percent_is_retain_supported(self):
        claims = [{"claim_id": f"A{i}", "weight": 1.0} for i in range(5)]
        candidates = [{
            "evidence_id": "PMID:1",
            "results": [
                {
                    "claim_id": f"A{i}",
                    "verdict": "supported",
                    "quote_verified": i < 3,
                    "quote": f"quote {i}",
                }
                for i in range(5)
            ],
        }]

        selected = select_bridge_set(claims, candidates)

        self.assertAlmostEqual(selected["coverage"], 0.6)
        self.assertEqual(selected["decision"], "retain_supported")

    def test_coverage_threshold_at_thirty_percent_is_retain_partial(self):
        claims = [{"claim_id": f"A{i}", "weight": 1.0} for i in range(10)]
        candidates = [{
            "evidence_id": "PMID:1",
            "results": [
                {
                    "claim_id": f"A{i}",
                    "verdict": "supported" if i < 3 else "unsupported",
                    "quote_verified": i < 3,
                    "quote": f"quote {i}",
                }
                for i in range(10)
            ],
        }]

        selected = select_bridge_set(claims, candidates)

        self.assertAlmostEqual(selected["coverage"], 0.3)
        self.assertEqual(selected["decision"], "retain_partial")

    def test_coverage_below_thirty_percent_is_remove(self):
        claims = [{"claim_id": f"A{i}", "weight": 1.0} for i in range(10)]
        candidates = [{
            "evidence_id": "PMID:1",
            "results": [
                {
                    "claim_id": f"A{i}",
                    "verdict": "supported" if i < 2 else "unsupported",
                    "quote_verified": i < 2,
                    "quote": f"quote {i}",
                }
                for i in range(10)
            ],
        }]

        selected = select_bridge_set(claims, candidates)

        self.assertAlmostEqual(selected["coverage"], 0.2)
        self.assertEqual(selected["decision"], "remove")

    def test_selects_at_most_three_papers(self):
        claims = [{"claim_id": f"A{i}", "weight": 1.0} for i in range(5)]
        candidates = [
            {
                "evidence_id": f"PMID:{i}",
                "results": [{
                    "claim_id": f"A{i}",
                    "verdict": "supported",
                    "quote_verified": True,
                    "quote": f"quote {i}",
                }],
            }
            for i in range(5)
        ]

        selected = select_bridge_set(claims, candidates)

        self.assertEqual(len(selected["evidence_ids"]), 3)

    def test_max_papers_cannot_exceed_global_cap(self):
        claims = [{"claim_id": f"A{i}", "weight": 1.0} for i in range(5)]
        candidates = [
            {
                "evidence_id": f"PMID:{i}",
                "results": [{
                    "claim_id": f"A{i}",
                    "verdict": "supported",
                    "quote_verified": True,
                    "quote": f"quote {i}",
                }],
            }
            for i in range(5)
        ]

        selected = select_bridge_set(claims, candidates, max_papers=10)

        self.assertEqual(len(selected["evidence_ids"]), 3)

    def test_contradicted_candidate_is_not_selected_as_positive_bridge(self):
        claims = [{"claim_id": "A1", "weight": 1.0}]
        candidates = [{
            "evidence_id": "PMID:3",
            "results": [{
                "claim_id": "A1",
                "verdict": "contradicted",
                "quote_verified": True,
            }],
        }]
        selected = select_bridge_set(claims, candidates)
        self.assertEqual(selected["evidence_ids"], [])
        self.assertEqual(selected["decision"], "remove")


class FakePubMed:
    def __init__(self, results_by_query=None, failing_queries=None):
        self.results_by_query = results_by_query or {}
        self.failing_queries = set(failing_queries or [])
        self.queries = []

    async def search(self, query, max_results=5):
        self.queries.append((query, max_results))
        if query in self.failing_queries:
            raise RuntimeError(f"search failed: {query}")
        return self.results_by_query.get(query, [])


class PlanningGroundingLLM:
    def __init__(self, plans=None, timeout_planning=False):
        self.plans = plans
        self.timeout_planning = timeout_planning
        self.planning_calls = 0

    async def structured(self, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "Plan bounded PubMed queries" in prompt:
            self.planning_calls += 1
            if self.timeout_planning:
                await asyncio.sleep(10)
            return {"groups": self.plans}, {}
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
                    "quote": "exact bridge words",
                    "reason": "Exact support.",
                }
                for claim in claims
            ]
        }, {}


class ExplodingIterable(list):
    def __iter__(self):
        raise RuntimeError("malformed planner iterable")


def _gap_claim(parent="P1", claim_id="A1", hypothesis_id="H1"):
    return {
        "claim_id": claim_id,
        "parent_claim_id": parent,
        "hypothesis_id": hypothesis_id,
        "verdict": "unverifiable",
        "evidence_results": [],
        "text": "Treatment improves outcome",
        "expected_relation": "support",
        "origin": {"kind": "causal_step", "axis_index": 0, "step_index": 2},
    }


async def _permissive_relevance(group, candidates, target):
    return {
        str(candidate["evidence_id"]): {
            "relevance": "direct",
            "reason": "offline test fixture",
        }
        for candidate in candidates
    }


class GapEvidenceRetrieverTests(unittest.IsolatedAsyncioTestCase):
    async def test_relevance_fail_closed_and_preserves_rejected_metadata(self):
        cases = {
            "absent": None,
            "missing_id": lambda group, candidates, target: {},
            "invalid": lambda group, candidates, target: {
                "PMID:1": {"relevance": "invalid", "reason": "bad value"}
            },
            "unknown": lambda group, candidates, target: {
                "PMID:1": {"relevance": "unknown", "reason": "uncertain"}
            },
            "empty": lambda group, candidates, target: {
                "PMID:1": {"relevance": "", "reason": "empty value"}
            },
        }
        for name, checker in cases.items():
            with self.subTest(name=name):
                llm = PlanningGroundingLLM([{
                    "parent_claim_id": "P1", "queries": ["one"],
                }])
                result = await GapEvidenceRetriever(
                    llm=llm,
                    search_client=FakePubMed({"one": [{
                        "evidence_id": "PMID:1",
                        "title": "Rejected bridge",
                        "abstract_snippet": "exact bridge words",
                        "source_metadata": {"pmid": "1", "doi": "10/test"},
                    }]}),
                    claim_verifier=ClaimVerifier(llm=llm),
                    bridge_relevance_checker=checker,
                ).retrieve(
                    [_gap_claim()],
                    [{"hypothesis_id": "H1", "indication": "Disease"}],
                    "Kv1.3",
                )

                group = result["groups"][0]
                self.assertEqual(group["selected"]["decision"], "remove")
                self.assertEqual(group["candidate_evidence"], [])
                self.assertEqual(group["rejected_candidates"][0]["evidence_id"], "PMID:1")
                self.assertEqual(
                    group["rejected_candidates"][0]["source_metadata"]["doi"],
                    "10/test",
                )
                self.assertEqual(
                    group["rejected_candidates"][0]["relevance"]["relevance"],
                    "unknown"
                    if name in {"absent", "missing_id", "empty"}
                    else name,
                )

    async def test_remove_group_does_not_reduce_unresolved_atomic_summary(self):
        class LowCoverageGrounder:
            async def ground_existing_claims(self, claims, metadata, target):
                grounded = []
                for index, claim in enumerate(claims):
                    result = dict(claim)
                    result["evidence_results"] = [{
                        "evidence_id": "PMID:1",
                        "verdict": "supported" if index == 0 else "unsupported",
                        "quote": "exact bridge words" if index == 0 else "",
                        "quote_verified": index == 0,
                    }]
                    result["verdict"] = result["evidence_results"][0]["verdict"]
                    grounded.append(result)
                return grounded

        llm = PlanningGroundingLLM([{
            "parent_claim_id": "P1", "queries": ["one"],
        }])
        claims = [
            _gap_claim(claim_id=f"A{index}") for index in range(5)
        ]
        result = await GapEvidenceRetriever(
            llm=llm,
            search_client=FakePubMed({"one": [{
                "evidence_id": "PMID:1",
                "title": "Low coverage bridge",
                "abstract_snippet": "exact bridge words",
                "source_metadata": {"pmid": "1"},
            }]}),
            claim_verifier=LowCoverageGrounder(),
            bridge_relevance_checker=_permissive_relevance,
        ).retrieve(
            claims,
            [{"hypothesis_id": "H1", "indication": "Disease"}],
            "Kv1.3",
        )

        self.assertEqual(result["groups"][0]["selected"]["decision"], "remove")
        self.assertEqual(result["summary"]["unresolved_atomic_claims"], 5)

    async def test_retained_partial_group_reduces_unresolved_summary(self):
        class PartialGrounder:
            async def ground_existing_claims(self, claims, metadata, target):
                grounded = []
                for index, claim in enumerate(claims):
                    result = dict(claim)
                    result["evidence_results"] = [{
                        "evidence_id": "PMID:1",
                        "verdict": "supported" if index == 0 else "unsupported",
                        "quote": "exact bridge words" if index == 0 else "",
                        "quote_verified": index == 0,
                    }]
                    result["verdict"] = result["evidence_results"][0]["verdict"]
                    grounded.append(result)
                return grounded

        llm = PlanningGroundingLLM([{
            "parent_claim_id": "P1", "queries": ["one"],
        }])
        result = await GapEvidenceRetriever(
            llm=llm,
            search_client=FakePubMed({"one": [{
                "evidence_id": "PMID:1",
                "title": "Partial bridge",
                "abstract_snippet": "exact bridge words",
                "source_metadata": {"pmid": "1"},
            }]}),
            claim_verifier=PartialGrounder(),
            bridge_relevance_checker=_permissive_relevance,
        ).retrieve(
            [_gap_claim(claim_id=f"A{index}") for index in range(3)],
            [{"hypothesis_id": "H1", "indication": "Disease"}],
            "Kv1.3",
        )

        self.assertEqual(
            result["groups"][0]["selected"]["decision"],
            "retain_partial",
        )
        self.assertEqual(result["summary"]["unresolved_atomic_claims"], 2)

    async def test_deduplicated_candidates_are_hard_capped_before_grounding(self):
        llm = PlanningGroundingLLM([{
            "parent_claim_id": "P1",
            "queries": ["planned one", "planned two"],
        }])
        records = [
            {
                "evidence_id": f"PMID:{index}",
                "title": f"Bridge study {index}",
                "abstract_snippet": "The paper contains exact bridge words.",
                "source_metadata": {"pmid": str(index)},
            }
            for index in range(10)
        ]
        pubmed = FakePubMed({
            "planned one": records[:5],
            "planned two": records[5:],
        })
        retriever = GapEvidenceRetriever(
            llm=llm,
            search_client=pubmed,
            claim_verifier=ClaimVerifier(llm=llm),
            bridge_relevance_checker=_permissive_relevance,
        )

        result = await retriever.retrieve(
            [_gap_claim()],
            [{"hypothesis_id": "H1", "indication": "Example disease"}],
            "Kv1.3",
        )

        group = result["groups"][0]
        self.assertEqual(
            [item["evidence_id"] for item in group["candidate_evidence"]],
            ["PMID:0", "PMID:1", "PMID:2"],
        )
        self.assertEqual(
            [item["evidence_id"] for item in group["grounded_candidates"]],
            ["PMID:0", "PMID:1", "PMID:2"],
        )

    async def test_planner_failure_and_group_grounding_failure_are_structured(self):
        class FailingFirstGrounder:
            def __init__(self):
                self.calls = 0

            async def ground_existing_claims(self, claims, metadata, target):
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError("offline grounding")
                return await ClaimVerifier(
                    llm=PlanningGroundingLLM([])
                ).ground_existing_claims(claims, metadata, target)

        claims = [
            _gap_claim(parent="P1", claim_id="A1", hypothesis_id="H1"),
            _gap_claim(parent="P2", claim_id="A2", hypothesis_id="H2"),
        ]
        fallback_one = fallback_query("Disease one", [claims[0]])
        fallback_two = fallback_query("Disease two", [claims[1]])
        pubmed = FakePubMed({
            fallback_one: [{
                "evidence_id": "PMID:1",
                "title": "First",
                "abstract_snippet": "exact bridge words",
                "source_metadata": {"pmid": "1"},
            }],
            fallback_two: [{
                "evidence_id": "PMID:2",
                "title": "Second",
                "abstract_snippet": "exact bridge words",
                "source_metadata": {"pmid": "2"},
            }],
        })
        retriever = GapEvidenceRetriever(
            llm=PlanningGroundingLLM(timeout_planning=True),
            search_client=pubmed,
            claim_verifier=FailingFirstGrounder(),
            bridge_relevance_checker=_permissive_relevance,
            planning_timeout_seconds=0.01,
        )

        result = await retriever.retrieve(
            claims,
            [
                {"hypothesis_id": "H1", "indication": "Disease one"},
                {"hypothesis_id": "H2", "indication": "Disease two"},
            ],
            "Kv1.3",
        )

        self.assertEqual(
            result["groups"][0]["selected"]["decision"],
            "remove",
        )
        self.assertEqual(
            result["groups"][1]["selected"]["evidence_ids"],
            ["PMID:2"],
        )
        errors = result["summary"]["errors"]
        self.assertTrue(any(error["stage"] == "planner_timeout" for error in errors))
        self.assertTrue(
            any(
                error["stage"] == "grounding"
                and error["parent_claim_id"] == "P1"
                for error in errors
            )
        )

    async def test_bridge_relevance_removes_mismatch_and_audits_failure(self):
        async def relevance(group, candidates, target):
            if group["parent_claim_id"] == "P1":
                return {
                    "PMID:1": {
                        "relevance": "mismatch",
                        "reason": "wrong tissue",
                    }
                }
            return {
                "PMID:2": {"relevance": "direct", "reason": "same tissue"}
            }

        claims = [
            _gap_claim(parent="P1", claim_id="A1", hypothesis_id="H1"),
            _gap_claim(parent="P2", claim_id="A2", hypothesis_id="H2"),
        ]
        pubmed = FakePubMed({
            "one": [{
                "evidence_id": "PMID:1", "title": "Mismatch",
                "abstract_snippet": "exact bridge words",
                "source_metadata": {"pmid": "1"},
            }],
            "two": [{
                "evidence_id": "PMID:2", "title": "Relevant",
                "abstract_snippet": "exact bridge words",
                "source_metadata": {"pmid": "2"},
            }],
        })
        llm = PlanningGroundingLLM([
            {"parent_claim_id": "P1", "queries": ["one"]},
            {"parent_claim_id": "P2", "queries": ["two"]},
        ])
        retriever = GapEvidenceRetriever(
            llm=llm, search_client=pubmed,
            claim_verifier=ClaimVerifier(llm=llm),
            bridge_relevance_checker=relevance,
        )

        result = await retriever.retrieve(
            claims,
            [
                {"hypothesis_id": "H1", "indication": "Disease one"},
                {"hypothesis_id": "H2", "indication": "Disease two"},
            ],
            "Kv1.3",
        )

        self.assertEqual(result["groups"][0]["selected"]["evidence_ids"], [])
        self.assertEqual(result["groups"][1]["selected"]["evidence_ids"], ["PMID:2"])
        self.assertEqual(
            result["groups"][0]["bridge_relevance"]["PMID:1"]["relevance"],
            "mismatch",
        )

    async def test_bridge_relevance_exception_removes_only_its_group(self):
        async def relevance(group, candidates, target):
            if group["parent_claim_id"] == "P1":
                raise TimeoutError("relevance timed out")
            return {
                "PMID:2": {"relevance": "direct", "reason": "same tissue"}
            }

        claims = [
            _gap_claim(parent="P1", claim_id="A1", hypothesis_id="H1"),
            _gap_claim(parent="P2", claim_id="A2", hypothesis_id="H2"),
        ]
        pubmed = FakePubMed({
            "one": [{
                "evidence_id": "PMID:1", "title": "First",
                "abstract_snippet": "exact bridge words",
                "source_metadata": {"pmid": "1"},
            }],
            "two": [{
                "evidence_id": "PMID:2", "title": "Second",
                "abstract_snippet": "exact bridge words",
                "source_metadata": {"pmid": "2"},
            }],
        })
        llm = PlanningGroundingLLM([
            {"parent_claim_id": "P1", "queries": ["one"]},
            {"parent_claim_id": "P2", "queries": ["two"]},
        ])
        result = await GapEvidenceRetriever(
            llm=llm, search_client=pubmed,
            claim_verifier=ClaimVerifier(llm=llm),
            bridge_relevance_checker=relevance,
        ).retrieve(
            claims,
            [
                {"hypothesis_id": "H1", "indication": "Disease one"},
                {"hypothesis_id": "H2", "indication": "Disease two"},
            ],
            "Kv1.3",
        )

        self.assertEqual(result["groups"][0]["selected"]["decision"], "remove")
        self.assertEqual(result["groups"][1]["selected"]["evidence_ids"], ["PMID:2"])
        self.assertTrue(
            any(
                error["stage"] == "bridge_relevance"
                and error["parent_claim_id"] == "P1"
                for error in result["summary"]["errors"]
            )
        )
        self.assertEqual(
            result["groups"][0]["rejected_candidates"][0]["relevance"][
                "reason"
            ],
            "relevance_checker_failed",
        )

    async def test_malformed_planner_shapes_use_fallback_without_raising(self):
        malformed_plans = [
            None,
            {"not": "a list"},
            ["not a dict", 7, None],
            [{"parent_claim_id": "P1", "queries": None}],
            [{"parent_claim_id": "P1", "queries": "character query"}],
            [{
                "parent_claim_id": "P1",
                "queries": [None, 7, "", "   "],
            }],
        ]
        claims = [_gap_claim()]
        expected = fallback_query("Example disease", claims)

        for plans in malformed_plans:
            with self.subTest(plans=plans):
                pubmed = FakePubMed()
                retriever = GapEvidenceRetriever(
                    llm=PlanningGroundingLLM(plans),
                    search_client=pubmed,
                    claim_verifier=ClaimVerifier(
                        llm=PlanningGroundingLLM([]),
                    ),
                )

                result = await retriever.retrieve(
                    claims,
                    [{
                        "hypothesis_id": "H1",
                        "indication": "Example disease",
                    }],
                    "Kv1.3",
                )

                self.assertEqual(result["groups"][0]["queries"], [expected])
                self.assertEqual(pubmed.queries, [(expected, 5)])

    async def test_planner_filters_invalid_queries_without_stringifying_them(self):
        plans = [{
            "parent_claim_id": "P1",
            "queries": [" valid one ", None, 7, "", "valid two"],
        }]
        pubmed = FakePubMed()
        retriever = GapEvidenceRetriever(
            llm=PlanningGroundingLLM(plans),
            search_client=pubmed,
            claim_verifier=ClaimVerifier(llm=PlanningGroundingLLM([])),
        )

        result = await retriever.retrieve(
            [_gap_claim()],
            [{"hypothesis_id": "H1", "indication": "Example disease"}],
            "Kv1.3",
        )

        self.assertEqual(
            result["groups"][0]["queries"],
            ["valid one", "valid two"],
        )
        self.assertEqual(
            pubmed.queries,
            [("valid one", 5), ("valid two", 5)],
        )

    async def test_planner_parse_exception_is_isolated_to_fallback(self):
        claims = [_gap_claim()]
        expected = fallback_query("Example disease", claims)
        pubmed = FakePubMed()
        retriever = GapEvidenceRetriever(
            llm=PlanningGroundingLLM(ExplodingIterable()),
            search_client=pubmed,
            claim_verifier=ClaimVerifier(llm=PlanningGroundingLLM([])),
        )

        result = await retriever.retrieve(
            claims,
            [{"hypothesis_id": "H1", "indication": "Example disease"}],
            "Kv1.3",
        )

        self.assertEqual(result["groups"][0]["queries"], [expected])
        self.assertEqual(pubmed.queries, [(expected, 5)])

    async def test_retrieval_bounds_queries_rejects_empty_abstract_and_labels_bridge(self):
        llm = PlanningGroundingLLM([{
            "parent_claim_id": "P1",
            "queries": ["planned one", "planned two", "ignored third"],
        }])
        pubmed = FakePubMed({
            "planned one": [{
                "evidence_id": "PMID:1",
                "title": "Bridge study",
                "abstract_snippet": "The paper contains exact bridge words.",
                "source_metadata": {"pmid": "1"},
            }],
            "planned two": [{
                "evidence_id": "PMID:2",
                "title": "No abstract",
                "abstract_snippet": "",
                "source_metadata": {"pmid": "2"},
            }],
        })
        retriever = GapEvidenceRetriever(
            llm=llm,
            search_client=pubmed,
            claim_verifier=ClaimVerifier(llm=llm),
            bridge_relevance_checker=_permissive_relevance,
        )

        result = await retriever.retrieve(
            [_gap_claim()],
            [{"hypothesis_id": "H1", "indication": "Example disease"}],
            "Kv1.3",
        )

        self.assertEqual(llm.planning_calls, 1)
        self.assertLessEqual(len(pubmed.queries), 2)
        self.assertTrue(all(limit == 5 for _, limit in pubmed.queries))
        group = result["groups"][0]
        self.assertEqual(
            [item["evidence_id"] for item in group["candidate_evidence"]],
            ["PMID:1"],
        )
        self.assertEqual(group["selected"]["evidence_ids"], ["PMID:1"])
        self.assertEqual(group["selected"]["evidence_role"], "bridge_evidence")
        self.assertEqual(
            group["candidate_evidence"][0]["retrieval_reason"],
            "uncited_causal_gap",
        )
        self.assertEqual(
            group["candidate_evidence"][0]["bridge_parent_claim_id"],
            "P1",
        )
        self.assertEqual(result["summary"]["gap_parent_steps"], 1)
        self.assertEqual(result["summary"]["searched_parent_steps"], 1)
        self.assertEqual(result["summary"]["resolved_parent_steps"], 1)
        self.assertEqual(result["summary"]["unresolved_atomic_claims"], 0)
        self.assertEqual(result["summary"]["errors"], [])

    async def test_query_planning_timeout_uses_deterministic_fallback(self):
        llm = PlanningGroundingLLM(timeout_planning=True)
        pubmed = FakePubMed()
        retriever = GapEvidenceRetriever(
            llm=llm,
            search_client=pubmed,
            claim_verifier=ClaimVerifier(llm=llm),
            planning_timeout_seconds=0.01,
        )
        claims = [_gap_claim()]

        result = await retriever.retrieve(
            claims,
            [{"hypothesis_id": "H1", "indication": "Example disease"}],
            "Kv1.3",
        )

        expected = fallback_query("Example disease", claims)
        self.assertEqual(pubmed.queries, [(expected, 5)])
        self.assertEqual(result["groups"][0]["queries"], [expected])

    async def test_incomplete_plans_fall_back_per_group_and_search_errors_continue(self):
        llm = PlanningGroundingLLM([{
            "parent_claim_id": "P1",
            "queries": ["broken query"],
        }])
        claims = [
            _gap_claim(parent="P1", claim_id="A1", hypothesis_id="H1"),
            _gap_claim(parent="P2", claim_id="A2", hypothesis_id="H2"),
        ]
        fallback = fallback_query("Disease two", [claims[1]])
        pubmed = FakePubMed(
            {fallback: [{
                "evidence_id": "PMID:9",
                "title": "Second group bridge",
                "abstract_snippet": "The paper contains exact bridge words.",
                "source_metadata": {"pmid": "9"},
            }]},
            failing_queries={"broken query"},
        )
        retriever = GapEvidenceRetriever(
            llm=llm,
            search_client=pubmed,
            claim_verifier=ClaimVerifier(llm=llm),
            bridge_relevance_checker=_permissive_relevance,
        )

        result = await retriever.retrieve(
            claims,
            [
                {"hypothesis_id": "H1", "indication": "Disease one"},
                {"hypothesis_id": "H2", "indication": "Disease two"},
            ],
            "Kv1.3",
        )

        self.assertEqual(len(result["groups"]), 2)
        self.assertEqual(result["groups"][1]["queries"], [fallback])
        self.assertEqual(
            result["groups"][1]["selected"]["evidence_ids"],
            ["PMID:9"],
        )
        self.assertEqual(len(result["summary"]["errors"]), 1)
        self.assertEqual(result["summary"]["errors"][0]["stage"], "retrieval")
        self.assertIn(
            "broken query",
            result["summary"]["errors"][0]["message"],
        )


if __name__ == "__main__":
    unittest.main()
