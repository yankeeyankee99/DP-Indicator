import unittest

from dp_indicator.core.orchestrator import Orchestrator


class VerificationRerankTests(unittest.IsolatedAsyncioTestCase):
    async def test_generation_always_invokes_verifier(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        calls = []

        async def fake_verify(hypotheses, evidence_pool, target):
            calls.append((hypotheses, evidence_pool, target))
            return {"verification_report": {}, "summary": "checked"}

        orchestrator._run_evidence_verifier = fake_verify
        result = await orchestrator._verification_for_generation(
            [{"rank": 1}],
            [{"evidence_id": "PMID:1"}],
            "Kv1.3",
        )

        self.assertEqual(result["summary"], "checked")
        self.assertEqual(len(calls), 1)

    def test_verification_penalties_apply_after_scoring_and_rerank(self):
        ranked = [
            {
                "hypothesis_id": "H1",
                "rank": 1,
                "overall_score": 0.8,
                "scores": {"overall": 0.8, "G1": 0.9},
            },
            {
                "hypothesis_id": "H2",
                "rank": 2,
                "overall_score": 0.7,
                "scores": {"overall": 0.7, "G1": 0.8},
            },
        ]

        adjusted = Orchestrator._apply_verification_adjustments(
            ranked,
            {"H1": -0.15},
        )

        self.assertEqual(
            [item["hypothesis_id"] for item in adjusted],
            ["H2", "H1"],
        )
        self.assertEqual([item["rank"] for item in adjusted], [1, 2])
        self.assertAlmostEqual(adjusted[1]["overall_score"], 0.65)
        self.assertAlmostEqual(
            adjusted[1]["scores"]["overall"],
            0.65,
        )
        self.assertAlmostEqual(
            adjusted[1]["scores"]["_verification_adjustment"],
            -0.15,
        )
        self.assertAlmostEqual(adjusted[1]["scores"]["G1"], 0.9)


if __name__ == "__main__":
    unittest.main()
