import json
import tempfile
import unittest
from pathlib import Path

from dp_indicator.benchmarks.evidence_integrity import (
    build_variants,
    extract_source_units,
    split_source_units,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def make_repeat_fixture(root: Path) -> Path:
    repeat_root = root / "repeat_runs"
    stage_root = repeat_root / "rep_01" / "checkpoints" / "stages"
    evidence_pool = [
        {
            "evidence_id": "PMID:1",
            "title": "Kv1.3 blockade in macrophages",
            "abstract_snippet": (
                "Kv1.3 blockade reduced IL-6. Other outcomes were unchanged."
            ),
        },
        {
            "evidence_id": "PMID:2",
            "title": "Nonexistent record",
            "abstract_snippet": "Kv1.3 blockade reduced IL-6.",
        },
        {
            "evidence_id": "DOI:10.1/example",
            "title": "Wrong identifier type",
            "abstract_snippet": "Kv1.3 blockade reduced IL-6.",
        },
        {
            "evidence_id": "PMID:3",
            "title": "Missing abstract",
            "abstract_snippet": "",
        },
        {
            "evidence_id": "PMID:4",
            "title": "Bridge evidence",
            "abstract_snippet": "Kv1.3 blockade reduced IL-6.",
        },
        {
            "evidence_id": "PMID:5",
            "title": "Whitespace-normalized source",
            "abstract_snippet": "KCa3.1 activation\nincreased T-cell proliferation.",
        },
    ]
    _write_json(
        stage_root / "pre_verification.json",
        {"evidence_pool": evidence_pool},
    )

    def result(evidence_id, quote, **extra):
        return {
            "evidence_id": evidence_id,
            "quote": quote,
            "verdict": "unsupported",
            "quote_verified": False,
            **extra,
        }

    claims = [
        {
            "claim_id": "claim-1",
            "hypothesis_id": "Inflammation",
            "text": "A historical model claim that must not become the gold text.",
            "evidence_results": [
                result("PMID:1", "Kv1.3 blockade reduced IL-6."),
                result("PMID:2", "Kv1.3 blockade reduced IL-6."),
                result("DOI:10.1/example", "Kv1.3 blockade reduced IL-6."),
                result("PMID:3", "Kv1.3 blockade reduced IL-6."),
                result(
                    "PMID:4",
                    "Kv1.3 blockade reduced IL-6.",
                    evidence_role="bridge_evidence",
                ),
            ],
        },
        {
            "claim_id": "claim-duplicate",
            "hypothesis_id": "Inflammation",
            "text": "Duplicate provenance",
            "evidence_results": [
                result("PMID:1", "  kv1.3   blockade reduced il-6.  "),
            ],
        },
        {
            "claim_id": "claim-2",
            "hypothesis_id": "Immunology",
            "text": "Another historical model claim.",
            "evidence_results": [
                result(
                    "PMID:5",
                    "KCa3.1 activation increased T-cell proliferation.",
                ),
            ],
        },
    ]
    verification = {
        "raw_verification_report": {
            "v1_id_existence": {
                "PMID:1": {"exists": True},
                "PMID:2": {"exists": False},
                "DOI:10.1/example": {"exists": True},
                "PMID:3": {"exists": True},
                "PMID:4": {"exists": True},
                "PMID:5": {"exists": True},
            },
            "claim_grounding": {"claims": claims},
        }
    }
    _write_json(stage_root / "verification.json", verification)
    return repeat_root


def eligible_units():
    return [
        {
            "source_id": "source-inflammation",
            "repeat": "rep_01",
            "hypothesis_id": "Inflammation",
            "claim_id": "claim-1",
            "original_claim": "Kv1.3 blockade may alter cytokines.",
            "evidence_id": "PMID:1",
            "title": "Kv1.3 blockade in macrophages",
            "abstract": (
                "Kv1.3 blockade reduced IL-6. Other outcomes were unchanged."
            ),
            "statement": "Kv1.3 blockade reduced IL-6.",
            "gold": "supported",
            "source_hashes": {"abstract_sha256": "a" * 64},
        },
        {
            "source_id": "source-immunology",
            "repeat": "rep_02",
            "hypothesis_id": "Immunology",
            "claim_id": "claim-2",
            "original_claim": "KCa3.1 may alter T-cell behavior.",
            "evidence_id": "PMID:5",
            "title": "KCa3.1 activation in T cells",
            "abstract": "KCa3.1 activation increased T-cell proliferation.",
            "statement": "KCa3.1 activation increased T-cell proliferation.",
            "gold": "supported",
            "source_hashes": {"abstract_sha256": "b" * 64},
        },
    ]


class EvidenceIntegrityDatasetTests(unittest.TestCase):
    def test_clean_unit_uses_exact_quote_not_historical_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            units = extract_source_units(
                make_repeat_fixture(Path(directory))
            )

        clean = next(
            unit for unit in units if unit["evidence_id"] == "PMID:1"
        )
        self.assertEqual(clean["statement"], "Kv1.3 blockade reduced IL-6.")
        self.assertIn(clean["statement"], clean["abstract"])
        self.assertEqual(clean["gold"], "supported")
        self.assertIs(clean["exists"], True)
        self.assertEqual(
            clean["original_claim"],
            "A historical model claim that must not become the gold text.",
        )

    def test_extraction_enforces_eligibility_and_normalized_quote_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            units = extract_source_units(
                make_repeat_fixture(Path(directory))
            )

        self.assertEqual(
            [unit["evidence_id"] for unit in units],
            ["PMID:1", "PMID:5"],
        )
        self.assertTrue(all(unit["exists"] is True for unit in units))
        normalized = next(
            unit for unit in units if unit["evidence_id"] == "PMID:5"
        )
        self.assertEqual(
            normalized["statement"],
            "KCa3.1 activation increased T-cell proliferation.",
        )
        self.assertEqual(
            set(normalized["source_hashes"]),
            {
                "verification_sha256",
                "pre_verification_sha256",
                "abstract_sha256",
                "quote_sha256",
            },
        )

    def test_every_corruption_has_known_label_and_provenance(self):
        variants = build_variants(eligible_units(), seed=20260728)

        self.assertEqual(len(variants), 10)
        self.assertEqual(
            {item["variant_type"] for item in variants},
            {
                "clean_quote",
                "invalid_id",
                "citation_swap",
                "causal_overclaim",
                "entity_swap",
            },
        )
        self.assertEqual(
            sum(item["gold"] == "supported" for item in variants),
            2,
        )
        self.assertTrue(
            all(
                item["exists"] is (
                    item["variant_type"] != "invalid_id"
                )
                for item in variants
            )
        )
        self.assertTrue(
            all(
                item["mutation_audit"]
                for item in variants
                if item["variant_type"] != "clean_quote"
            )
        )
        for item in variants:
            if item["variant_type"] == "causal_overclaim":
                self.assertTrue(
                    item["mutation_audit"]["validation"][
                        "added_proposition_absent_from_source"
                    ]
                )
            if item["variant_type"] == "entity_swap":
                replacement = item["mutation_audit"]["replacement"]
                frozen_source = f"{item['title']} {item['abstract']}"
                self.assertNotIn(
                    replacement.casefold(),
                    frozen_source.casefold(),
                )

    def test_mutations_are_seeded_and_citation_swaps_cross_topics(self):
        first = build_variants(eligible_units(), seed=17)
        second = build_variants(eligible_units(), seed=17)

        self.assertEqual(first, second)
        swaps = [
            item for item in first
            if item["variant_type"] == "citation_swap"
        ]
        self.assertTrue(
            all(
                item["hypothesis_id"]
                != item["mutation_audit"]["donor_hypothesis_id"]
                for item in swaps
            )
        )

    def test_unit_without_safe_transformations_emits_only_baselines(self):
        units = eligible_units() + [
            {
                "source_id": "source-no-safe-mutation",
                "repeat": "rep_03",
                "hypothesis_id": " ",
                "claim_id": "claim-3",
                "original_claim": "A descriptive claim.",
                "evidence_id": "PMID:9",
                "title": "Descriptive source",
                "abstract": "A descriptive biomedical observation.",
                "statement": "A descriptive biomedical observation.",
                "gold": "supported",
                "source_hashes": {"abstract_sha256": "c" * 64},
            }
        ]

        variants = build_variants(units, seed=17)

        self.assertEqual(
            [
                item["variant_type"]
                for item in variants
                if item["source_id"] == "source-no-safe-mutation"
            ],
            ["clean_quote", "invalid_id"],
        )
        self.assertEqual(len(variants), 12)

    def test_explicit_templates_cover_overclaim_and_entity_mutations(self):
        archived_like = {
            "source_id": "source-archived-like",
            "repeat": "rep_03",
            "hypothesis_id": "Kidney disease",
            "claim_id": "claim-4",
            "original_claim": "Kv1.3 may affect inflammation.",
            "evidence_id": "PMID:10",
            "title": "Kv1.3 and KCa3.1 in renal inflammation",
            "abstract": (
                "Kv1.3 inhibition attenuated renal inflammation. "
                "KCa3.1 remained unchanged."
            ),
            "statement": "Kv1.3 inhibition attenuated renal inflammation.",
            "gold": "supported",
            "source_hashes": {"abstract_sha256": "d" * 64},
        }

        variants = build_variants(
            eligible_units() + [archived_like],
            seed=17,
        )
        archived_variants = [
            item
            for item in variants
            if item["source_id"] == "source-archived-like"
        ]

        self.assertEqual(len(archived_variants), 5)
        overclaim = next(
            item
            for item in archived_variants
            if item["variant_type"] == "causal_overclaim"
        )
        entity = next(
            item
            for item in archived_variants
            if item["variant_type"] == "entity_swap"
        )
        self.assertIn("randomized phase III", overclaim["statement"])
        self.assertIn("approved treatment", overclaim["statement"])
        self.assertIn("Nav1.5", entity["statement"])

    def test_pilot_and_confirmation_are_source_disjoint_and_deterministic(self):
        units = [
            {"source_id": f"source-{index:03d}"}
            for index in range(160)
        ]

        first = split_source_units(units, 40, 120, 20260728)
        second = split_source_units(units, 40, 120, 20260728)

        self.assertEqual(first, second)
        self.assertEqual(len(first["pilot_ids"]), 40)
        self.assertEqual(len(first["confirmation_ids"]), 120)
        self.assertTrue(
            set(first["pilot_ids"]).isdisjoint(first["confirmation_ids"])
        )

    def test_split_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            split_source_units(
                [{"source_id": "same"}, {"source_id": "same"}],
                1,
                1,
                7,
            )


if __name__ == "__main__":
    unittest.main()
