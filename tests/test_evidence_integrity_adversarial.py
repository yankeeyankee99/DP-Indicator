import unittest
from unittest.mock import patch

from dp_indicator.benchmarks.evidence_integrity import (
    CONFIRMATION_BUNDLE_COUNT,
    CONFIRMATION_PAIRS_PER_CORRUPTION,
    PILOT_BUNDLE_COUNT,
    PILOT_PAIRS_PER_CORRUPTION,
    RESERVED_BENCHMARK_PMID,
    MutationIneligible,
    _assess_citation_donor,
    _causal_overclaim_variant,
    _choose_cross_topic_donor,
    _coverage_counts,
    _entity_swap_variant,
    _select_coverage_group,
    build_split_variants,
    build_variants,
    sample_paired_variants,
    select_coverage_aware_split,
    split_source_units,
    validate_variant_yields,
)


def unit(
    source_id,
    topic,
    statement,
    *,
    title=None,
    abstract=None,
    evidence_id=None,
):
    return {
        "source_id": source_id,
        "repeat": "rep_01",
        "hypothesis_id": topic,
        "claim_id": f"claim-{source_id}",
        "original_claim": statement,
        "evidence_id": evidence_id or f"PMID:{100 + len(source_id)}",
        "title": title or statement,
        "abstract": abstract or statement,
        "statement": statement,
        "gold": "supported",
        "source_hashes": {"abstract_sha256": source_id * 4},
    }


def safe_pair():
    return [
        unit(
            "source-a",
            "Renal inflammation",
            "Kv1.3 blockade reduced IL-6.",
            title="Kv1.3 blockade in renal macrophages",
            abstract=(
                "Kv1.3 blockade reduced IL-6. Renal macrophages were studied."
            ),
            evidence_id="PMID:101",
        ),
        unit(
            "source-b",
            "T-cell activation",
            "KCa3.1 activation increased T-cell proliferation.",
            title="KCa3.1 activation in T cells",
            abstract=(
                "KCa3.1 activation increased T-cell proliferation. "
                "Adaptive immunity was measured."
            ),
            evidence_id="PMID:102",
        ),
    ]


class CitationAnchorContractTests(unittest.TestCase):
    def test_numeric_anchor_matches_whitespace_and_case_variants(self):
        source = unit(
            "numeric-source",
            "Dose response",
            "A 10mg dose reduced the biomarker.",
        )
        for donor_text in (
            "The administered dose was 10 mg.",
            "The administered dose was 10  mg.",
            "The administered dose was 10 MG.",
        ):
            with self.subTest(donor_text=donor_text):
                donor = unit(
                    "numeric-donor",
                    "Pharmacology",
                    donor_text,
                )

                assessment = _assess_citation_donor(source, donor)
                numeric = next(
                    item
                    for item in assessment["anchor_checks"]
                    if item["kind"] == "numeric"
                )

                self.assertTrue(numeric["present"])
                self.assertFalse(assessment["eligible"])

    def test_donor_evaluations_stop_after_first_safe_candidate(self):
        source = safe_pair()[0]
        donors = [
            unit(
                f"donor-{index}",
                f"Distinct topic {index}",
                "KCa3.1 activation increased T-cell proliferation.",
            )
            for index in range(20)
        ]
        from dp_indicator.benchmarks import evidence_integrity

        original = evidence_integrity._assess_citation_donor
        with patch(
            "dp_indicator.benchmarks.evidence_integrity."
            "_assess_citation_donor",
            wraps=original,
        ) as assessed:
            donor, audit = _choose_cross_topic_donor(
                source,
                [source, *donors],
                seed=17,
            )

        self.assertTrue(audit["eligible"])
        self.assertIn(donor["source_id"], {item["source_id"] for item in donors})
        self.assertEqual(assessed.call_count, 1)

    def test_donor_choice_is_deterministic_under_input_reordering(self):
        source = safe_pair()[0]
        donors = [
            unit(
                f"donor-{index}",
                f"Distinct topic {index}",
                "KCa3.1 activation increased T-cell proliferation.",
            )
            for index in range(8)
        ]

        first, first_audit = _choose_cross_topic_donor(
            source,
            [source, *donors],
            seed=20260728,
        )
        second, second_audit = _choose_cross_topic_donor(
            source,
            list(reversed([source, *donors])),
            seed=20260728,
        )

        self.assertEqual(first["source_id"], second["source_id"])
        self.assertEqual(first_audit, second_audit)

    def test_no_safe_donor_diagnostics_are_bounded_and_aggregated(self):
        source = safe_pair()[0]
        donors = [
            unit(
                f"same-topic-{index}",
                "renal-inflammation",
                "KCa3.1 activation increased T-cell proliferation.",
            )
            for index in range(50)
        ]

        with self.assertRaises(MutationIneligible) as caught:
            _choose_cross_topic_donor(
                source,
                [source, *donors],
                seed=17,
            )

        details = caught.exception.details
        self.assertNotIn("donor_assessments", details)
        self.assertEqual(sum(details["reason_counts"].values()), 50)
        self.assertLessEqual(len(details["diagnostic_samples"]), 9)
        self.assertEqual(details["candidate_count"], 50)

    def test_same_normalized_topic_is_not_a_safe_donor(self):
        source = safe_pair()[0]
        donor = unit(
            "same-topic",
            "  renal-inflammation ",
            "KCa3.1 activation increased T-cell proliferation.",
        )

        assessment = _assess_citation_donor(source, donor)

        self.assertFalse(assessment["eligible"])
        self.assertFalse(assessment["checks"]["distinct_normalized_topic"])
        self.assertEqual(
            assessment["normalized_source_topic"],
            assessment["normalized_donor_topic"],
        )

    def test_high_overlap_paraphrase_is_not_a_safe_donor(self):
        source = unit(
            "overlap-source",
            "Renal inflammation",
            (
                "Kv1.3 treatment reduced inflammatory response in cultured "
                "renal cells after toxin exposure."
            ),
        )
        donor = unit(
            "overlap-donor",
            "Cell toxicology",
            (
                "KCa3.1 treatment increased inflammatory response in cultured "
                "renal cells after toxin exposure."
            ),
        )

        assessment = _assess_citation_donor(source, donor)

        self.assertFalse(assessment["eligible"])
        self.assertFalse(assessment["checks"]["low_content_token_overlap"])
        self.assertGreater(
            assessment["content_token_overlap"],
            assessment["content_token_overlap_threshold"],
        )

    def test_safe_citation_swap_persists_anchor_and_overlap_checks(self):
        variants, report = build_variants(
            safe_pair(),
            seed=20260728,
            return_report=True,
        )

        swaps = [
            item
            for item in variants
            if item["variant_type"] == "citation_swap"
        ]
        self.assertEqual(len(swaps), 2)
        self.assertEqual(report["ineligible_source_count"], 0)
        for swap in swaps:
            audit = swap["mutation_audit"]
            self.assertTrue(audit["essential_anchors"])
            self.assertTrue(
                all(not check["present"] for check in audit["anchor_checks"])
            )
            self.assertTrue(audit["validation"]["all_anchors_absent"])
            self.assertTrue(
                audit["validation"]["low_content_token_overlap"]
            )
            self.assertTrue(
                audit["validation"]["distinct_normalized_topic"]
            )


class CausalOverclaimContractTests(unittest.TestCase):
    def test_missing_indication_is_ineligible(self):
        source = unit(
            "missing-indication",
            "  ",
            "Kv1.3 blockade reduced IL-6.",
        )

        with self.assertRaises(MutationIneligible) as caught:
            _causal_overclaim_variant(source)

        self.assertEqual(caught.exception.operator, "causal_overclaim")
        self.assertEqual(caught.exception.reason, "missing_indication")

    def test_preexisting_clinical_efficacy_claim_is_ineligible(self):
        source = unit(
            "clinical-source",
            "Renal inflammation",
            "Kv1.3 inhibition improved clinical remission.",
            title="Phase III randomized controlled trial of Kv1.3 inhibition",
            abstract=(
                "Kv1.3 inhibition improved clinical remission. The trial "
                "demonstrated clinical efficacy in renal inflammation."
            ),
        )

        with self.assertRaises(MutationIneligible) as caught:
            _causal_overclaim_variant(source)

        self.assertEqual(
            caught.exception.reason,
            "compound_anchor_in_source",
        )

    def test_preexisting_randomized_trial_claim_is_ineligible(self):
        source = unit(
            "randomized-source",
            "Renal inflammation",
            "Kv1.3 blockade reduced IL-6.",
            title="Randomized trial of Kv1.3 inhibition",
            abstract="Kv1.3 blockade reduced IL-6.",
        )

        with self.assertRaises(MutationIneligible) as caught:
            _causal_overclaim_variant(source)

        self.assertEqual(
            caught.exception.reason,
            "compound_anchor_in_source",
        )

    def test_earlier_clinical_language_remains_eligible_for_compound_claim(self):
        cases = (
            ("Phase I study reported tolerability.", "phase I"),
            ("Phase II study suggested clinical benefit.", "phase II"),
            (
                "An observational cohort reported clinical benefit.",
                "observational",
            ),
            ("Complete renal response was observed.", "renal response"),
            ("Improved patient outcomes were reported.", "patient outcomes"),
        )
        for extra_source, label in cases:
            with self.subTest(label=label):
                source = unit(
                    f"earlier-{label}",
                    "Renal inflammation",
                    "Kv1.3 blockade reduced IL-6.",
                    abstract=(
                        "Kv1.3 blockade reduced IL-6. " + extra_source
                    ),
                )

                variant = _causal_overclaim_variant(source)

                self.assertIn("randomized phase III", variant["statement"])
                self.assertIn("approved treatment", variant["statement"])
                self.assertTrue(
                    variant["mutation_audit"]["validation"][
                        "clearly_missing_conjunct"
                    ]
                )

    def test_regulatory_approval_anchor_is_ineligible(self):
        source = unit(
            "approved-source",
            "Renal inflammation",
            "Kv1.3 blockade reduced IL-6.",
            abstract=(
                "Kv1.3 blockade reduced IL-6. Regulatory approval was "
                "granted for this treatment."
            ),
        )

        with self.assertRaises(MutationIneligible) as caught:
            _causal_overclaim_variant(source)

        self.assertEqual(caught.exception.reason, "compound_anchor_in_source")

    def test_randomized_phase_three_anchor_is_ineligible(self):
        source = unit(
            "phase-three-source",
            "Renal inflammation",
            "Kv1.3 blockade reduced IL-6.",
            title="Randomized phase III trial of Kv1.3 blockade",
            abstract="Kv1.3 blockade reduced IL-6.",
        )

        with self.assertRaises(MutationIneligible) as caught:
            _causal_overclaim_variant(source)

        self.assertEqual(caught.exception.reason, "compound_anchor_in_source")

    def test_exact_quote_is_preserved_before_separate_overclaim(self):
        source = safe_pair()[0]

        variant = _causal_overclaim_variant(source)

        self.assertTrue(
            variant["statement"].startswith(source["statement"] + " ")
        )
        self.assertEqual(
            variant["mutation_audit"]["clean_quote"],
            source["statement"],
        )
        self.assertNotEqual(
            variant["mutation_audit"]["overclaim_sentence"],
            source["statement"],
        )

    def test_overclaim_audit_proves_added_proposition_is_unsupported(self):
        variant = _causal_overclaim_variant(safe_pair()[0])
        audit = variant["mutation_audit"]
        checks = audit["validation"]

        self.assertEqual(audit["normalized_indication"], "inflammation renal")
        self.assertTrue(audit["overclaim_anchors"])
        self.assertEqual(
            {item["canonical"] for item in audit["overclaim_anchors"]},
            {"randomized phase III result", "approved treatment status"},
        )
        self.assertEqual(
            set(audit["missing_conjuncts"]),
            {item["canonical"] for item in audit["overclaim_anchors"]},
        )
        for hidden_marker in (
            "benchmark",
            "sentinel",
            "unsupported",
            RESERVED_BENCHMARK_PMID,
        ):
            self.assertNotIn(
                hidden_marker.casefold(),
                audit["overclaim_sentence"].casefold(),
            )
        self.assertTrue(
            all(not item["present"] for item in audit["anchor_checks"])
        )
        self.assertTrue(
            all(
                not item["present"]
                for item in audit["disease_specific_conclusion_checks"]
            )
        )
        self.assertTrue(checks["clean_quote_preserved_exactly"])
        self.assertTrue(checks["clean_quote_exact_in_source"])
        self.assertTrue(checks["overclaim_is_separate_sentence"])
        self.assertTrue(checks["overclaim_anchors_absent_from_source"])
        self.assertTrue(
            checks["disease_specific_conclusions_absent_from_source"]
        )
        self.assertTrue(checks["no_randomized_phase_iii_anchor"])
        self.assertTrue(checks["no_regulatory_approval_anchor"])
        self.assertTrue(checks["added_proposition_absent_from_source"])
        self.assertTrue(checks["clearly_missing_conjunct"])


class EntityContractTests(unittest.TestCase):
    def test_entity_does_not_match_token_substring(self):
        source = unit(
            "entity-substring",
            "Assay",
            "antiKv1.3like reagent reduced cytokines.",
        )

        with self.assertRaises(MutationIneligible) as caught:
            _entity_swap_variant(source)

        self.assertEqual(caught.exception.reason, "no_central_entity")

    def test_replacement_alias_in_source_makes_entity_swap_ineligible(self):
        source = unit(
            "alias",
            "Cardiac biology",
            "Kv1.3 blockade reduced IL-6.",
            title="SCN5A and inflammatory signaling",
            abstract=(
                "Kv1.3 blockade reduced IL-6. SCN5A expression was measured."
            ),
        )

        with self.assertRaises(MutationIneligible) as caught:
            _entity_swap_variant(source)

        self.assertEqual(caught.exception.reason, "replacement_alias_in_source")

    def test_entity_audit_records_recognition_boundaries_and_aliases(self):
        variant = _entity_swap_variant(safe_pair()[0])
        audit = variant["mutation_audit"]

        self.assertEqual(audit["recognized_entity"], "Kv1.3")
        self.assertTrue(audit["validation"]["token_boundary_match"])
        self.assertTrue(audit["validation"]["central_entity_in_exact_quote"])
        self.assertTrue(
            audit["validation"]["replacement_aliases_absent_from_source"]
        )
        self.assertIn("SCN5A", audit["replacement_aliases"])


class AttritionAndInvalidIdTests(unittest.TestCase):
    def test_real_source_variants_explicitly_exist(self):
        variants = build_variants(safe_pair(), seed=17)

        by_type = {
            variant_type: [
                item["exists"]
                for item in variants
                if item["variant_type"] == variant_type
            ]
            for variant_type in (
                "clean_quote",
                "citation_swap",
                "causal_overclaim",
                "entity_swap",
            )
        }

        self.assertTrue(all(by_type.values()))
        self.assertTrue(
            all(exists is True for values in by_type.values() for exists in values)
        )

    def test_partial_eligibility_emits_baselines_and_safe_operators(self):
        no_semantic = unit(
            "no-semantic",
            " ",
            "A descriptive biomedical observation.",
        )

        variants, report = build_variants(
            safe_pair() + [no_semantic],
            seed=17,
            return_report=True,
        )

        emitted = [
            item["variant_type"]
            for item in variants
            if item["source_id"] == "no-semantic"
        ]
        self.assertEqual(emitted, ["clean_quote", "invalid_id"])
        self.assertEqual(
            report["emitted_counts"],
            {
                "clean_quote": 3,
                "invalid_id": 3,
                "citation_swap": 2,
                "causal_overclaim": 2,
                "entity_swap": 2,
            },
        )
        self.assertEqual(report["complete_source_count"], 2)
        self.assertEqual(report["partial_source_count"], 1)

    def test_attrition_report_records_each_expected_reason(self):
        no_mutation = unit(
            "no-mutation",
            " ",
            "A descriptive biomedical observation.",
        )

        variants, report = build_variants(
            safe_pair() + [no_mutation],
            seed=17,
            return_report=True,
        )

        self.assertEqual(
            [
                item["variant_type"]
                for item in variants
                if item["source_id"] == "no-mutation"
            ],
            ["clean_quote", "invalid_id"],
        )
        self.assertEqual(report["input_source_count"], 3)
        self.assertEqual(report["emitted_source_count"], 3)
        self.assertEqual(report["complete_source_count"], 2)
        self.assertEqual(report["partial_source_count"], 1)
        records = [
            item
            for item in report["records"]
            if item["source_id"] == "no-mutation"
        ]
        self.assertEqual(
            {item["operator"] for item in records},
            {"citation_swap", "causal_overclaim", "entity_swap"},
        )
        self.assertTrue(all(item["reason"] for item in records))

    def test_unexpected_value_error_is_not_swallowed(self):
        with patch(
            "dp_indicator.benchmarks.evidence_integrity."
            "_causal_overclaim_variant",
            side_effect=ValueError("implementation defect"),
        ):
            with self.assertRaisesRegex(ValueError, "implementation defect"):
                build_variants(safe_pair(), seed=17, return_report=True)

    def test_malformed_source_is_not_reported_as_expected_attrition(self):
        malformed = safe_pair()[0]
        del malformed["statement"]

        with self.assertRaisesRegex(ValueError, "missing fields"):
            build_variants([malformed], seed=17, return_report=True)

    def test_invalid_id_uses_reserved_frozen_universe_definition(self):
        variants = build_variants(safe_pair(), seed=17)
        invalid = next(
            item
            for item in variants
            if item["variant_type"] == "invalid_id"
        )
        audit = invalid["mutation_audit"]

        self.assertEqual(invalid["evidence_id"], RESERVED_BENCHMARK_PMID)
        self.assertIs(invalid["exists"], False)
        self.assertEqual(
            audit["operational_gold"],
            "source_unavailable_in_frozen_v1_universe",
        )
        self.assertEqual(
            audit["validation_basis"],
            "reserved_benchmark_id_absent_from_frozen_v1",
        )
        self.assertTrue(
            audit["validation"]["not_a_real_world_nonexistence_claim"]
        )


class SourceShortfallTests(unittest.TestCase):
    @staticmethod
    def units(count):
        return [
            {"source_id": f"source-{index:03d}"}
            for index in range(count)
        ]

    def test_shortfall_preserves_pilot_and_uses_all_remaining_sources(self):
        split = split_source_units(
            self.units(141),
            pilot_n=40,
            confirmation_n=120,
            seed=20260728,
        )

        self.assertEqual(len(split["pilot_ids"]), 40)
        self.assertEqual(len(split["confirmation_ids"]), 101)
        self.assertEqual(
            len(split["pilot_ids"]) + len(split["confirmation_ids"]),
            141,
        )
        self.assertEqual(
            split["shortfall"],
            {
                "requested_total": 160,
                "available_total": 141,
                "total_shortfall": 19,
                "pilot_shortfall": 0,
                "confirmation_shortfall": 19,
                "used_all_available": True,
            },
        )

    def test_shortfall_uses_smaller_meaningful_pilot(self):
        split = split_source_units(
            self.units(25),
            pilot_n=40,
            confirmation_n=120,
            seed=9,
        )

        self.assertEqual(len(split["pilot_ids"]), 25)
        self.assertEqual(split["confirmation_ids"], [])
        self.assertEqual(split["shortfall"]["pilot_shortfall"], 15)

    def test_fewer_than_twenty_sources_is_not_a_meaningful_pilot(self):
        with self.assertRaisesRegex(ValueError, "meaningful pilot"):
            split_source_units(
                self.units(19),
                pilot_n=40,
                confirmation_n=120,
                seed=9,
            )


class CoverageAwareSplitTests(unittest.TestCase):
    def test_citation_count_requires_each_sources_own_donor(self):
        eligibility = {
            "source-a": {"invalid_id", "citation_swap"},
            "source-b": {"invalid_id", "citation_swap"},
            "source-c": {"invalid_id"},
        }
        citation_donors = {
            "source-a": "source-b",
            "source-b": "source-c",
        }

        counts = _coverage_counts(
            {"source-a", "source-b"},
            eligibility,
            citation_donors,
        )

        self.assertEqual(counts["citation_swap"], 1)

    def test_reserve_path_preserves_directed_citation_package(self):
        ids = ("source-a", "source-b", "source-c", "source-d", "source-e")
        units_by_id = {
            source_id: unit(
                source_id,
                source_id,
                "Treatment reduced the biomarker.",
            )
            for source_id in ids
        }
        group_eligibility = {
            source_id: {"invalid_id"} for source_id in ids
        }
        group_eligibility["source-a"].add("citation_swap")
        group_donors = {"source-a": "source-b"}
        reserve_eligibility = {
            source_id: {
                "invalid_id",
                "causal_overclaim",
                "entity_swap",
            }
            for source_id in ids
        }
        reserve_eligibility["source-b"].add("citation_swap")
        reserve_eligibility["source-c"].add("citation_swap")
        reserve_donors = {
            "source-b": "source-c",
            "source-c": "source-a",
        }

        with patch(
            "dp_indicator.benchmarks.evidence_integrity."
            "_constructibility",
            return_value=(group_eligibility, group_donors),
        ):
            selected, _ = _select_coverage_group(
                units_by_id,
                set(ids),
                size=2,
                required_per_type=1,
                seed=17,
                split_name="reserve-regression",
                reserve_eligibility=reserve_eligibility,
                reserve_citation_donors=reserve_donors,
                reserve_required_per_type=1,
            )

        remaining = set(ids).difference(selected)
        self.assertGreaterEqual(
            _coverage_counts(
                remaining,
                reserve_eligibility,
                reserve_donors,
            )["citation_swap"],
            1,
        )

    @staticmethod
    def coverage_pool(clinical_eligible=50):
        units = []
        for index in range(160):
            if index % 2:
                statement = (
                    "KCa3.1 activation increased T-cell proliferation."
                )
                title = "KCa3.1 activation in T cells"
            else:
                statement = "Kv1.3 blockade reduced IL-6."
                title = "Kv1.3 blockade in macrophages"
            indication = (
                f"Indication {index:03d}"
                if index < clinical_eligible
                else " "
            )
            units.append(
                unit(
                    f"coverage-{index:03d}",
                    indication,
                    statement,
                    title=title,
                    abstract=statement,
                    evidence_id=f"PMID:{1000 + index}",
                )
            )
        return units

    @staticmethod
    def constrained_short_pool():
        units = []
        statements = (
            ("Kv1.3 blockade reduced IL-6.", "Kv1.3 macrophage study"),
            (
                "KCa3.1 activation increased T-cell proliferation.",
                "KCa3.1 T-cell study",
            ),
            (
                "Kinase activity reduced marker abundance.",
                "Kinase marker study",
            ),
            (
                "Membrane voltage increased conductance.",
                "Membrane conductance study",
            ),
        )
        for index in range(100):
            if index < 40:
                statement, title = statements[index % 2]
            else:
                statement, title = statements[2 + index % 2]
            units.append(
                unit(
                    f"short-{index:03d}",
                    f"Indication {index:03d}",
                    statement,
                    title=title,
                    abstract=statement,
                    evidence_id=f"PMID:{3000 + index}",
                )
            )
        return units

    def test_coverage_split_is_deterministic_disjoint_and_meets_minima(self):
        units = self.coverage_pool()

        first = select_coverage_aware_split(
            units,
            pilot_n=40,
            confirmation_n=120,
            seed=20260728,
        )
        second = select_coverage_aware_split(
            list(reversed(units)),
            pilot_n=40,
            confirmation_n=120,
            seed=20260728,
        )
        built = build_split_variants(units, first, seed=20260728)

        self.assertEqual(first, second)
        self.assertEqual(len(first["pilot_ids"]), 40)
        self.assertEqual(len(first["confirmation_ids"]), 120)
        self.assertTrue(
            set(first["pilot_ids"]).isdisjoint(
                first["confirmation_ids"]
            )
        )
        self.assertTrue(built["pilot"]["yield_report"]["all_ready"])
        self.assertTrue(
            built["confirmation"]["yield_report"]["all_ready"]
        )
        self.assertEqual(first["selection_method"], "constructibility")
        self.assertTrue(first["selection_strata"])

    def test_coverage_split_records_unavoidable_operator_shortfall(self):
        units = self.coverage_pool(clinical_eligible=5)

        split = select_coverage_aware_split(
            units,
            pilot_n=40,
            confirmation_n=120,
            seed=17,
        )

        self.assertGreater(
            split["coverage_shortfall"]["causal_overclaim"],
            0,
        )
        self.assertIn(
            "causal_overclaim",
            split["blocked_types"],
        )

    def test_short_pool_stays_coverage_aware_when_random_split_fails(self):
        units = self.constrained_short_pool()
        random_split = split_source_units(
            units,
            pilot_n=40,
            confirmation_n=120,
            seed=17,
        )
        random_built = build_split_variants(
            units,
            random_split,
            seed=17,
        )

        selected = select_coverage_aware_split(
            units,
            pilot_n=40,
            confirmation_n=120,
            seed=17,
        )
        selected_built = build_split_variants(
            units,
            selected,
            seed=17,
        )

        self.assertFalse(
            random_built["confirmation"]["yield_report"]["all_ready"]
        )
        self.assertEqual(len(selected["pilot_ids"]), 40)
        self.assertEqual(len(selected["confirmation_ids"]), 60)
        self.assertTrue(
            selected_built["pilot"]["yield_report"]["all_ready"]
        )
        self.assertTrue(
            selected_built["confirmation"]["yield_report"]["all_ready"]
        )
        self.assertEqual(
            selected["selection_method"],
            "constructibility",
        )
        self.assertEqual(
            selected,
            select_coverage_aware_split(
                list(reversed(units)),
                pilot_n=40,
                confirmation_n=120,
                seed=17,
            ),
        )
        for split_name in ("pilot", "confirmation"):
            self.assertEqual(
                selected["selection_strata"][split_name][
                    "constructible_counts"
                ],
                selected_built[split_name]["yield_report"]["counts"],
            )
            self.assertTrue(
                selected["selection_strata"][split_name][
                    "post_selection_yield_verified"
                ]
            )


class SplitYieldAndSamplingTests(unittest.TestCase):
    @staticmethod
    def paired_manifest(per_type):
        variants = []
        for variant_type in (
            "invalid_id",
            "citation_swap",
            "causal_overclaim",
            "entity_swap",
        ):
            for index in range(per_type):
                source_id = f"{variant_type}-{index:03d}"
                variants.extend(
                    [
                        {
                            "source_id": source_id,
                            "variant_id": f"{source_id}-clean",
                            "variant_type": "clean_quote",
                            "gold": "supported",
                        },
                        {
                            "source_id": source_id,
                            "variant_id": f"{source_id}-{variant_type}",
                            "variant_type": variant_type,
                            "gold": "unsupported",
                        },
                    ]
                )
        return variants

    def test_yield_validation_blocks_only_short_corruption_types(self):
        variants = self.paired_manifest(10)
        variants = [
            item
            for item in variants
            if not (
                item["variant_type"] == "entity_swap"
                and item["source_id"].endswith(("008", "009"))
            )
        ]

        result = validate_variant_yields(variants, minimum_per_type=10)

        self.assertEqual(result["counts"]["entity_swap"], 8)
        self.assertEqual(result["shortfall"]["entity_swap"], 2)
        self.assertEqual(result["blocked_types"], ["entity_swap"])
        self.assertIn("citation_swap", result["ready_types"])
        self.assertFalse(result["all_ready"])

    def test_fixed_pair_samples_produce_revised_item_counts(self):
        variants = self.paired_manifest(30)

        pilot = sample_paired_variants(
            variants,
            pairs_per_type=PILOT_PAIRS_PER_CORRUPTION,
            seed=17,
        )
        confirmation = sample_paired_variants(
            list(reversed(variants)),
            pairs_per_type=CONFIRMATION_PAIRS_PER_CORRUPTION,
            seed=17,
        )

        self.assertEqual(PILOT_PAIRS_PER_CORRUPTION, 10)
        self.assertEqual(CONFIRMATION_PAIRS_PER_CORRUPTION, 30)
        self.assertEqual(len(pilot), 80)
        self.assertEqual(len(confirmation), 240)
        self.assertEqual(
            confirmation,
            sample_paired_variants(
                variants,
                pairs_per_type=30,
                seed=17,
            ),
        )
        self.assertEqual(PILOT_BUNDLE_COUNT, 9)
        self.assertEqual(CONFIRMATION_BUNDLE_COUNT, 30)

    def test_split_variants_are_built_independently_and_source_disjoint(self):
        first = safe_pair()
        second = [
            {
                **item,
                "source_id": item["source_id"].replace("source", "confirm"),
                "evidence_id": item["evidence_id"].replace("10", "20"),
                "hypothesis_id": f"Confirmation {item['hypothesis_id']}",
            }
            for item in safe_pair()
        ]
        units = first + second
        split = {
            "pilot_ids": [item["source_id"] for item in first],
            "confirmation_ids": [item["source_id"] for item in second],
        }

        result = build_split_variants(units, split, seed=17)

        pilot_ids = {
            item["source_id"] for item in result["pilot"]["variants"]
        }
        confirmation_ids = {
            item["source_id"]
            for item in result["confirmation"]["variants"]
        }
        self.assertEqual(pilot_ids, set(split["pilot_ids"]))
        self.assertEqual(confirmation_ids, set(split["confirmation_ids"]))
        self.assertTrue(pilot_ids.isdisjoint(confirmation_ids))


if __name__ == "__main__":
    unittest.main()
