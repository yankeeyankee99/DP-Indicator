import unittest

from dp_indicator.agents.evidence_verifier import (
    apply_bridge_evidence,
    claim_grounding_to_legacy_issues,
    filter_verification_report,
    sanitize_hypotheses,
)


class EvidenceSanitizerTests(unittest.TestCase):
    def setUp(self):
        self.hypotheses = [
            {
                "hypothesis_id": "H1",
                "indication": "Example Disease",
                "causal_chain": {
                    "mechanism_axes": [
                        {
                            "steps": [
                                {
                                    "layer": "L1",
                                    "status": "supported",
                                    "evidence_ids": [
                                        "PMID:missing",
                                        "PMID:fabricated",
                                    ],
                                },
                                {
                                    "layer": "L2",
                                    "status": "supported",
                                    "evidence_ids": ["PMID:partial"],
                                },
                                {
                                    "layer": "L3",
                                    "status": "supported",
                                    "evidence_ids": ["PMID:good"],
                                },
                            ]
                        }
                    ]
                },
                "evidence_mapping": {
                    "positive_evidence": [
                        {"id": "PMID:missing", "rationale": "missing"},
                        {"id": "PMID:fabricated", "rationale": "fabricated"},
                        {"id": "PMID:bad-description", "rationale": "wrong"},
                        {"id": "PMID:over", "rationale": "too strong"},
                        {"id": "PMID:mismatch", "rationale": "other tissue"},
                        {"id": "PMID:good", "rationale": "supported"},
                        {"id": "PMID:good", "rationale": "duplicate"},
                    ],
                    "indirect_evidence": [],
                    "contradicting_evidence": [],
                },
            }
        ]
        self.verification_result = {
            "verification_report": {
                "v1_id_existence": {
                    "PMID:missing": {"exists": False},
                    "PMID:fabricated": {"exists": True},
                    "PMID:partial": {"exists": True},
                    "PMID:bad-description": {"exists": True},
                    "PMID:over": {"exists": True},
                    "PMID:mismatch": {"exists": True},
                    "PMID:good": {"exists": True},
                },
                "v2_citation_accuracy": {
                    "H1": [
                        {
                            "evidence_id": "PMID:fabricated",
                            "step": "L1",
                            "issue": "Citation fabricated: not in abstract",
                            "severity": "high",
                        },
                        {
                            "evidence_id": "PMID:partial",
                            "step": "L2",
                            "issue": "Citation partial: extra claim",
                            "severity": "medium",
                        },
                    ]
                },
                "v3_description_accuracy": {
                    "H1": [
                        {
                            "evidence_id": "PMID:bad-description",
                            "issue": "Rationale misattributed: wrong article",
                            "severity": "high",
                        },
                        {
                            "evidence_id": "PMID:over",
                            "issue": "Rationale overinterpreted: too strong",
                            "severity": "medium",
                        },
                    ]
                },
                "v4_indication_relevance": {
                    "H1": [
                        {
                            "evidence_id": "PMID:mismatch",
                            "issue": "Tissue/organ mismatch: other organ",
                            "severity": "high",
                        }
                    ]
                },
                "v5_count_consistency": {},
            },
            "score_adjustments": {"H1": -0.15},
            "summary": "raw summary",
        }

    def test_apply_bridge_evidence_attaches_by_origin_and_deduplicates(self):
        hypotheses = [{
            "hypothesis_id": "H1",
            "cited_evidence_ids": ["PMID:direct"],
            "causal_chain": {
                "mechanism_axes": [{
                    "steps": [
                        {
                            "layer": "L1",
                            "status": "inferred",
                            "evidence_ids": ["PMID:direct"],
                            "sources": [{
                                "evidence_id": "PMID:direct",
                                "source_text": "direct source",
                            }],
                        },
                        {
                            "layer": "L2",
                            "status": "inferred",
                            "evidence_ids": [],
                        },
                    ],
                }],
            },
            "evidence_mapping": {},
        }]
        evidence_pool = [{"evidence_id": "PMID:direct", "title": "Direct"}]
        retrieval = {
            "groups": [
                {
                    "hypothesis_id": "H1",
                    "origin": {
                        "kind": "causal_step",
                        "axis_index": 0,
                        "step_index": 0,
                    },
                    "selected": {
                        "evidence_ids": ["PMID:1"],
                        "coverage": 0.6,
                        "decision": "retain_supported",
                        "verified_spans": [{
                            "claim_id": "A1",
                            "evidence_id": "PMID:1",
                            "quote": "supported bridge span",
                            "quote_verified": True,
                        }],
                    },
                },
                {
                    "hypothesis_id": "H1",
                    "origin": {
                        "kind": "causal_step",
                        "axis_index": 0,
                        "step_index": 1,
                    },
                    "selected": {
                        "evidence_ids": ["PMID:2"],
                        "coverage": 0.3,
                        "decision": "retain_partial",
                        "verified_spans": [{
                            "claim_id": "A2",
                            "evidence_id": "PMID:2",
                            "quote": "partial bridge span",
                            "quote_verified": True,
                        }],
                    },
                },
            ],
            "evidence": [
                {
                    "evidence_id": "PMID:1",
                    "title": "Bridge one",
                    "abstract_snippet": "supported bridge span",
                    "source_metadata": {
                        "pmid": "1",
                        "journal": "Journal One",
                    },
                },
                {
                    "evidence_id": "PMID:2",
                    "title": "Bridge two",
                    "abstract_snippet": "partial bridge span",
                    "source_metadata": {"pmid": "2"},
                },
                {
                    "evidence_id": "PMID:1",
                    "title": "Duplicate bridge one",
                    "source_metadata": {"pmid": "1"},
                },
            ],
        }

        apply_bridge_evidence(hypotheses, evidence_pool, retrieval)
        apply_bridge_evidence(hypotheses, evidence_pool, retrieval)

        steps = hypotheses[0]["causal_chain"]["mechanism_axes"][0]["steps"]
        self.assertEqual(
            steps[0]["evidence_ids"],
            ["PMID:direct", "PMID:1"],
        )
        self.assertEqual(steps[0]["status"], "supported")
        self.assertEqual(steps[1]["evidence_ids"], ["PMID:2"])
        self.assertEqual(steps[1]["status"], "inferred")
        self.assertEqual(
            hypotheses[0]["cited_evidence_ids"],
            ["PMID:direct", "PMID:1", "PMID:2"],
        )
        self.assertEqual(
            [source["evidence_id"] for source in steps[0]["sources"]],
            ["PMID:direct", "PMID:1"],
        )
        self.assertEqual(
            steps[0]["sources"][1]["evidence_role"],
            "bridge_evidence",
        )
        self.assertEqual(
            steps[0]["sources"][1]["retrieval_reason"],
            "uncited_causal_gap",
        )
        self.assertEqual(
            steps[0]["verified_spans"][0]["evidence_id"],
            "PMID:1",
        )
        self.assertEqual(
            [item["evidence_id"] for item in evidence_pool],
            ["PMID:direct", "PMID:1", "PMID:2"],
        )

    def test_sanitizer_removes_only_hard_failures_and_preserves_input(self):
        original = repr(self.hypotheses)

        sanitized, audit = sanitize_hypotheses(
            self.hypotheses,
            self.verification_result,
        )

        self.assertEqual(repr(self.hypotheses), original)
        steps = sanitized[0]["causal_chain"]["mechanism_axes"][0]["steps"]
        self.assertEqual(steps[0]["evidence_ids"], [])
        self.assertEqual(steps[0]["status"], "inferred")
        self.assertEqual(steps[1]["evidence_ids"], ["PMID:partial"])
        self.assertEqual(steps[1]["status"], "inferred")
        self.assertEqual(steps[2]["evidence_ids"], ["PMID:good"])
        self.assertEqual(steps[2]["status"], "supported")

        positive = sanitized[0]["evidence_mapping"]["positive_evidence"]
        self.assertEqual(
            [item["id"] for item in positive],
            ["PMID:over", "PMID:mismatch", "PMID:good"],
        )
        self.assertEqual(
            positive[0]["verification_status"],
            "overinterpreted",
        )
        self.assertEqual(
            positive[1]["verification_relevance"],
            "mismatch",
        )
        self.assertGreaterEqual(len(audit), 6)

    def test_filtered_report_contains_only_evidence_remaining_in_output(self):
        sanitized, _ = sanitize_hypotheses(
            self.hypotheses,
            self.verification_result,
        )

        final_report = filter_verification_report(
            self.verification_result["verification_report"],
            sanitized,
        )

        self.assertNotIn(
            "PMID:missing",
            final_report["v1_id_existence"],
        )
        self.assertNotIn(
            "PMID:fabricated",
            final_report["v1_id_existence"],
        )
        self.assertNotIn(
            "PMID:bad-description",
            final_report["v1_id_existence"],
        )
        self.assertEqual(
            final_report["v2_citation_accuracy"]["H1"][0]["evidence_id"],
            "PMID:partial",
        )
        self.assertEqual(
            final_report["v3_description_accuracy"]["H1"][0]["evidence_id"],
            "PMID:over",
        )
        self.assertEqual(
            final_report["v4_indication_relevance"]["H1"][0]["evidence_id"],
            "PMID:mismatch",
        )

    def test_claim_grounding_binds_verified_span_and_removes_bad_source(self):
        hypotheses = [
            {
                "hypothesis_id": "H1",
                "causal_chain": {
                    "mechanism_axes": [
                        {
                            "steps": [
                                {
                                    "layer": "L1",
                                    "mechanism": "A compound claim.",
                                    "status": "supported",
                                    "evidence_ids": ["PMID:good", "PMID:bad"],
                                }
                            ]
                        }
                    ]
                },
                "evidence_mapping": {
                    "positive_evidence": [
                        {"id": "PMID:good", "rationale": "good"},
                        {"id": "PMID:bad", "rationale": "bad"},
                    ],
                    "indirect_evidence": [],
                    "contradicting_evidence": [],
                },
            }
        ]
        claims = [
            {
                "claim_id": "CLM-step",
                "hypothesis_id": "H1",
                "origin": {
                    "kind": "causal_step",
                    "axis_index": 0,
                    "step_index": 0,
                    "layer": "L1",
                },
                "verdict": "supported",
                "evidence_results": [
                    {
                        "evidence_id": "PMID:good",
                        "verdict": "supported",
                        "quote": "exact source words",
                        "quote_verified": True,
                    },
                    {
                        "evidence_id": "PMID:bad",
                        "verdict": "unsupported",
                        "quote": "",
                        "quote_verified": False,
                    },
                ],
            },
            {
                "claim_id": "CLM-map",
                "hypothesis_id": "H1",
                "origin": {
                    "kind": "evidence_mapping",
                    "bucket": "positive_evidence",
                    "item_index": 1,
                },
                "verdict": "unsupported",
                "evidence_results": [
                    {
                        "evidence_id": "PMID:bad",
                        "verdict": "unsupported",
                        "quote": "",
                        "quote_verified": False,
                    }
                ],
            },
        ]

        sanitized, audit = sanitize_hypotheses(
            hypotheses,
            {
                "verification_report": {
                    "v1_id_existence": {
                        "PMID:good": {"exists": True},
                        "PMID:bad": {"exists": True},
                    },
                    "claim_grounding": {"claims": claims},
                    "v2_citation_accuracy": {},
                    "v3_description_accuracy": {},
                    "v4_indication_relevance": {},
                }
            },
        )

        step = sanitized[0]["causal_chain"]["mechanism_axes"][0]["steps"][0]
        self.assertEqual(step["evidence_ids"], ["PMID:good"])
        self.assertEqual(step["status"], "supported")
        self.assertEqual(
            step["verified_spans"][0]["quote"],
            "exact source words",
        )
        positive = sanitized[0]["evidence_mapping"]["positive_evidence"]
        self.assertEqual([item["id"] for item in positive], ["PMID:good"])
        self.assertTrue(
            any(item["reason"] == "unsupported_claim" for item in audit)
        )

    def test_compound_step_retains_citation_at_two_thirds_coverage(self):
        hypotheses = [
            {
                "hypothesis_id": "H1",
                "causal_chain": {
                    "mechanism_axes": [
                        {
                            "steps": [
                                {
                                    "layer": "L1",
                                    "mechanism": "A compound mechanism.",
                                    "status": "supported",
                                    "evidence_ids": ["PMID:1"],
                                }
                            ]
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
        claims = []
        for index, verdict in enumerate(
            ["supported", "supported", "unsupported"]
        ):
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
                    "verdict": verdict,
                    "evidence_results": [
                        {
                            "evidence_id": "PMID:1",
                            "verdict": verdict,
                            "quote": f"supported quote {index}",
                            "quote_verified": verdict == "supported",
                        }
                    ],
                }
            )

        sanitized, _ = sanitize_hypotheses(
            hypotheses,
            {
                "verification_report": {
                    "v1_id_existence": {"PMID:1": {"exists": True}},
                    "claim_grounding": {"claims": claims},
                    "v2_citation_accuracy": {
                        "H1": [
                            {
                                "evidence_id": "PMID:1",
                                "step": "L1",
                                "severity": "high",
                                "issue": "Derived atomic compatibility issue",
                            }
                        ]
                    },
                    "v3_description_accuracy": {},
                    "v4_indication_relevance": {},
                }
            },
        )

        step = sanitized[0]["causal_chain"]["mechanism_axes"][0]["steps"][0]
        self.assertEqual(step["evidence_ids"], ["PMID:1"])
        self.assertEqual(step["status"], "supported")
        self.assertEqual(len(step["verified_spans"]), 2)

    def test_compound_step_removes_citation_below_thirty_percent(self):
        hypotheses = [
            {
                "hypothesis_id": "H1",
                "causal_chain": {
                    "mechanism_axes": [
                        {
                            "steps": [
                                {
                                    "layer": "L1",
                                    "status": "supported",
                                    "evidence_ids": ["PMID:1"],
                                }
                            ]
                        }
                    ]
                },
                "evidence_mapping": {},
            }
        ]
        claims = []
        for index, verdict in enumerate(
            ["partial", "unsupported", "unsupported"]
        ):
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
                    "verdict": verdict,
                    "evidence_results": [
                        {
                            "evidence_id": "PMID:1",
                            "verdict": verdict,
                            "quote_verified": False,
                        }
                    ],
                }
            )

        sanitized, _ = sanitize_hypotheses(
            hypotheses,
            {
                "verification_report": {
                    "v1_id_existence": {"PMID:1": {"exists": True}},
                    "claim_grounding": {"claims": claims},
                    "v2_citation_accuracy": {},
                    "v3_description_accuracy": {},
                    "v4_indication_relevance": {},
                }
            },
        )

        step = sanitized[0]["causal_chain"]["mechanism_axes"][0]["steps"][0]
        self.assertEqual(step["evidence_ids"], [])
        self.assertEqual(step["status"], "inferred")

    def test_bridge_aggregate_preserves_bridge_without_rescuing_bad_direct_claim(self):
        hypotheses = [{
            "hypothesis_id": "H1",
            "causal_chain": {
                "mechanism_axes": [{
                    "steps": [{
                        "layer": "L1",
                        "status": "supported",
                        "evidence_ids": ["PMID:bad", "PMID:bridge"],
                    }],
                }],
            },
            "evidence_mapping": {},
        }]
        direct_claim = {
            "claim_id": "A1",
            "parent_claim_id": "P1",
            "hypothesis_id": "H1",
            "origin": {
                "kind": "causal_step",
                "axis_index": 0,
                "step_index": 0,
                "layer": "L1",
            },
            "verdict": "unsupported",
            "evidence_results": [{
                "evidence_id": "PMID:bad",
                "verdict": "unsupported",
                "quote_verified": False,
            }],
        }
        bridge_aggregate = {
            "aggregate_kind": "bridge_set",
            "hypothesis_id": "H1",
            "parent_claim_id": "P1",
            "origin": direct_claim["origin"],
            "evidence_ids": ["PMID:bridge"],
            "coverage": 0.6,
            "decision": "retain_supported",
            "verified_spans": [{
                "claim_id": "A1",
                "evidence_id": "PMID:bridge",
                "quote": "bridge support",
                "quote_verified": True,
                "evidence_role": "bridge_evidence",
            }],
        }

        sanitized, _ = sanitize_hypotheses(
            hypotheses,
            {
                "verification_report": {
                    "v1_id_existence": {
                        "PMID:bad": {"exists": True},
                        "PMID:bridge": {"exists": True},
                    },
                    "claim_grounding": {
                        "claims": [direct_claim],
                        "bridge_aggregates": [bridge_aggregate],
                    },
                    "v2_citation_accuracy": {},
                    "v3_description_accuracy": {},
                    "v4_indication_relevance": {},
                },
            },
        )

        step = sanitized[0]["causal_chain"]["mechanism_axes"][0]["steps"][0]
        self.assertEqual(step["evidence_ids"], ["PMID:bridge"])
        self.assertEqual(step["status"], "supported")
        self.assertEqual(
            step["verified_spans"][0]["evidence_id"],
            "PMID:bridge",
        )

    def test_claim_grounding_derives_legacy_issue_views(self):
        v2, v3 = claim_grounding_to_legacy_issues(
            [
                {
                    "hypothesis_id": "H1",
                    "origin": {"kind": "causal_step", "layer": "L1"},
                    "verdict": "partial",
                    "evidence_results": [
                        {
                            "evidence_id": "PMID:1",
                            "verdict": "partial",
                            "reason": "Only one clause is supported.",
                        }
                    ],
                },
                {
                    "hypothesis_id": "H1",
                    "origin": {"kind": "evidence_mapping"},
                    "verdict": "unsupported",
                    "evidence_results": [
                        {
                            "evidence_id": "PMID:2",
                            "verdict": "unsupported",
                            "reason": "No support in abstract.",
                        }
                    ],
                },
            ]
        )

        self.assertEqual(v2["H1"][0]["severity"], "medium")
        self.assertEqual(v2["H1"][0]["step"], "L1")
        self.assertEqual(v3["H1"][0]["severity"], "high")

    def test_filtered_report_removes_grounding_for_deleted_evidence(self):
        sanitized = [
            {
                "hypothesis_id": "H1",
                "causal_chain": {
                    "mechanism_axes": [
                        {
                            "steps": [
                                {
                                    "layer": "L1",
                                    "evidence_ids": ["PMID:good"],
                                }
                            ]
                        }
                    ]
                },
                "evidence_mapping": {
                    "positive_evidence": [
                        {"id": "PMID:good", "rationale": "good"}
                    ]
                },
            }
        ]
        raw = {
            "v1_id_existence": {
                "PMID:good": {"exists": True},
                "PMID:bad": {"exists": True},
            },
            "claim_grounding": {
                "claims": [
                    {
                        "claim_id": "CLM-1",
                        "hypothesis_id": "H1",
                        "verdict": "mixed",
                        "evidence_results": [
                            {
                                "evidence_id": "PMID:good",
                                "verdict": "supported",
                            },
                            {
                                "evidence_id": "PMID:bad",
                                "verdict": "contradicted",
                            },
                        ],
                    }
                ],
                "by_hypothesis": {},
            },
            "v2_citation_accuracy": {},
            "v3_description_accuracy": {},
            "v4_indication_relevance": {},
        }

        filtered = filter_verification_report(raw, sanitized)

        claim = filtered["claim_grounding"]["claims"][0]
        self.assertEqual(claim["verdict"], "supported")
        self.assertEqual(
            [item["evidence_id"] for item in claim["evidence_results"]],
            ["PMID:good"],
        )
        self.assertEqual(
            filtered["claim_grounding"]["by_hypothesis"]["H1"][0][
                "claim_id"
            ],
            "CLM-1",
        )

    def test_filtered_report_keeps_bridge_aggregate_and_direct_verdict(self):
        sanitized = [{
            "hypothesis_id": "H1",
            "causal_chain": {
                "mechanism_axes": [{
                    "steps": [{
                        "evidence_ids": ["PMID:bridge"],
                    }],
                }],
            },
            "evidence_mapping": {},
        }]
        raw = {
            "v1_id_existence": {},
            "claim_grounding": {
                "claims": [{
                    "claim_id": "A1",
                    "hypothesis_id": "H1",
                    "verdict": "unsupported",
                    "evidence_results": [
                        {
                            "evidence_id": "PMID:direct",
                            "verdict": "unsupported",
                        },
                        {
                            "evidence_id": "PMID:bridge",
                            "verdict": "supported",
                            "quote_verified": True,
                            "evidence_role": "bridge_evidence",
                        },
                    ],
                }],
                "bridge_aggregates": [{
                    "aggregate_kind": "bridge_set",
                    "hypothesis_id": "H1",
                    "parent_claim_id": "P1",
                    "evidence_ids": ["PMID:bridge"],
                    "decision": "retain_supported",
                    "verified_spans": [{
                        "evidence_id": "PMID:bridge",
                        "quote": "bridge quote",
                        "quote_verified": True,
                    }],
                }],
            },
            "v2_citation_accuracy": {},
            "v3_description_accuracy": {},
            "v4_indication_relevance": {},
        }

        filtered = filter_verification_report(raw, sanitized)

        claim = filtered["claim_grounding"]["claims"][0]
        self.assertEqual(claim["verdict"], "unsupported")
        self.assertEqual(
            filtered["claim_grounding"]["bridge_aggregates"][0][
                "evidence_ids"
            ],
            ["PMID:bridge"],
        )


if __name__ == "__main__":
    unittest.main()
