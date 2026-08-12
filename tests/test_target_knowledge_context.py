import unittest

from dp_indicator.core.target_knowledge import (
    build_biological_context,
    clear_yaml_cache,
    get_out_of_scope_keywords,
)


class TargetKnowledgeContextTests(unittest.TestCase):
    def setUp(self):
        clear_yaml_cache()

    def test_production_context_contains_scope_and_novelty_rules(self):
        context = build_biological_context("Kv1.3")

        self.assertIn("Mechanism Scope Boundary", context)
        self.assertIn("Known/Established Indications", context)
        self.assertIn("Extended Mechanistic Bridges", context)

    def test_production_scope_keywords_are_enabled(self):
        self.assertIn(
            "myasthenia",
            get_out_of_scope_keywords("Kv1.3"),
        )


if __name__ == "__main__":
    unittest.main()
