import unittest
from unittest.mock import AsyncMock

from dp_indicator.agents.core_agents import (
    AuditRecorder,
    HypothesisCritic,
)
from dp_indicator.core.orchestrator import Orchestrator


class CriticAgentTests(unittest.IsolatedAsyncioTestCase):
    def make_critic(self):
        critic = HypothesisCritic(
            llm=object(),
            audit=AuditRecorder(),
            model="test-model",
            task="critic",
        )
        critic.evidence_mapper.map_hypothesis = AsyncMock(
            return_value={
                "positive_evidence": [{"id": "PMID:1"}],
                "contradicting_evidence": [],
            }
        )
        critic._review = AsyncMock(
            return_value={
                "fatal_weakness": {"weakness": "Missing causal bridge"},
                "suggested_fix": "Measure the bridge directly",
                "alternative_explanations": ["Reverse causality"],
            }
        )
        return critic

    async def test_map_evidence_does_not_call_review(self):
        critic = self.make_critic()
        hypotheses = [{"hypothesis_id": "H1", "indication": "Disease"}]

        result = await critic.map_evidence(hypotheses, [], "Kv1.3")

        critic.evidence_mapper.map_hypothesis.assert_awaited_once()
        critic._review.assert_not_awaited()
        self.assertIn("evidence_mapping", result[0])
        self.assertNotIn("critic_review", result[0])

    async def test_review_does_not_remap_evidence(self):
        critic = self.make_critic()
        hypotheses = [
            {
                "hypothesis_id": "H1",
                "indication": "Disease",
                "evidence_mapping": {"positive_evidence": []},
            }
        ]

        result = await critic.review_hypotheses(
            hypotheses,
            [],
            "Kv1.3",
        )

        critic.evidence_mapper.map_hypothesis.assert_not_awaited()
        critic._review.assert_awaited_once()
        self.assertIn("critic_review", result[0])

    def test_review_normalization_accepts_string_weakness(self):
        normalized = HypothesisCritic.normalize_review(
            {
                "fatal_weakness": "The causal bridge is indirect",
                "suggested_fix": {
                    "recommendation": "Measure the bridge directly"
                },
            }
        )

        self.assertEqual(
            normalized["fatal_weakness"]["weakness"],
            "The causal bridge is indirect",
        )
        self.assertEqual(
            normalized["suggested_fix"],
            "Measure the bridge directly",
        )

    async def test_orchestrator_maps_then_reviews_once(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        checkpoints = {}
        orchestrator._save_stage_checkpoint = (
            lambda name, value: checkpoints.setdefault(name, value)
        )
        critic = self.make_critic()

        reviewed = await orchestrator._critic_for_generation(
            critic,
            [{"hypothesis_id": "H1", "indication": "Disease"}],
            [],
            "Kv1.3",
        )

        critic.evidence_mapper.map_hypothesis.assert_awaited_once()
        critic._review.assert_awaited_once()
        self.assertNotIn(
            "critic_review",
            checkpoints["post_mapping"]["hypotheses"][0],
        )
        self.assertIn("critic_review", reviewed[0])


if __name__ == "__main__":
    unittest.main()
