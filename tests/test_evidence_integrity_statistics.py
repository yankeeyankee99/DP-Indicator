import copy
import unittest
from unittest.mock import patch

from dp_indicator.benchmarks.evidence_integrity_stats import (
    binary_metrics,
    build_bundles,
    bundle_bootstrap_difference,
    bundle_macro_f1_difference,
    bundle_metrics,
    bundle_permutation_test,
    cluster_bootstrap_difference,
    cluster_permutation_test,
    exact_mcnemar,
    holm_adjust,
    compute_bundle_hash,
    compute_split_manifest_hash,
    integrity_decision,
    macro_f1_difference,
    majority_vote_predictions,
)


SEMANTIC_TYPES = ("citation_swap", "causal_overclaim", "entity_swap")


def binary_record(
    variant_id,
    source_id,
    gold,
    prediction,
    variant_type,
):
    return {
        "variant_id": variant_id,
        "source_id": source_id,
        "gold": gold,
        "prediction": prediction,
        "variant_type": variant_type,
    }


def paired_conditions():
    gold = ("supported", "unsupported") * 4
    full_predictions = (
        "supported",
        "unsupported",
        "supported",
        "unsupported",
        "unsupported",
        "unsupported",
        "supported",
        "supported",
    )
    control_predictions = (
        "supported",
        "supported",
        "unsupported",
        "supported",
        "unsupported",
        "unsupported",
        "supported",
        "unsupported",
    )
    types = (
        "clean_quote",
        "invalid_id",
        "clean_quote",
        "citation_swap",
        "clean_quote",
        "causal_overclaim",
        "clean_quote",
        "entity_swap",
    )
    first = []
    second = []
    for index, (label, full, control, variant_type) in enumerate(
        zip(gold, full_predictions, control_predictions, types)
    ):
        variant_id = f"v-{index}"
        source_id = f"source-{index // 2}"
        first.append(
            binary_record(
                variant_id, source_id, label, full, variant_type
            )
        )
        second.append(
            binary_record(
                variant_id, source_id, label, control, variant_type
            )
        )
    return first, second


def variant_manifest(source_count):
    variants = []
    for index in range(source_count):
        source_id = f"source-{index:03d}"
        common = {
            "source_id": source_id,
            "repeat": f"rep_{index % 6 + 1:02d}",
            "hypothesis_id": f"hypothesis-{index % 7}",
            "evidence_id": f"PMID:{1000 + index}",
            "exists": True,
            "abstract": f"Frozen abstract for source {index}.",
        }
        variants.append(
            {
                **common,
                "variant_id": f"{source_id}-clean",
                "variant_type": "clean_quote",
                "gold": "supported",
            }
        )
        for variant_type in SEMANTIC_TYPES:
            variants.append(
                {
                    **common,
                    "variant_id": f"{source_id}-{variant_type}",
                    "variant_type": variant_type,
                    "gold": "unsupported",
                    "mutation_audit": {
                        "operator": variant_type,
                        "validation": {"safe": True},
                    },
                }
            )
        variants.append(
            {
                **common,
                "variant_id": f"{source_id}-invalid_id",
                "variant_type": "invalid_id",
                "gold": "unsupported",
                "evidence_id": "PMID:00000000",
                "exists": False,
            }
        )
    return variants


def split_manifest(split_name, source_count):
    if split_name == "pilot":
        pilot_ids = [f"source-{index:03d}" for index in range(source_count)]
        confirmation_ids = [
            f"confirmation-{index:03d}" for index in range(120)
        ]
    else:
        pilot_ids = [f"pilot-{index:03d}" for index in range(40)]
        confirmation_ids = [
            f"source-{index:03d}" for index in range(source_count)
        ]
    manifest = {
        "name": split_name,
        "pilot_ids": pilot_ids,
        "confirmation_ids": confirmation_ids,
    }
    manifest["split_sha256"] = compute_split_manifest_hash(manifest)
    return manifest


class BinaryMetricTests(unittest.TestCase):
    def test_holm_adjustment_is_monotone_and_preserves_raw_values(self):
        adjusted = holm_adjust({
            "small": 0.01,
            "middle": 0.03,
            "large": 0.2,
        })
        self.assertEqual(adjusted["small"], {
            "raw_p": 0.01,
            "holm_adjusted_p": 0.03,
        })
        self.assertEqual(adjusted["middle"]["holm_adjusted_p"], 0.06)
        self.assertEqual(adjusted["large"]["holm_adjusted_p"], 0.2)

    def test_repetitions_are_majority_voted_before_scoring(self):
        first, _ = paired_conditions()
        second = [dict(item) for item in first]
        third = [dict(item) for item in first]
        second[0]["prediction"] = "unsupported"

        result = majority_vote_predictions({1: first, 2: second, 3: third})

        self.assertEqual(result["records"], first)
        self.assertEqual(result["repetition_ids"], [1, 2, 3])
        self.assertEqual(result["disagreement_count"], 1)
        self.assertEqual(result["disagreement_rate"], 0.125)

    def test_majority_vote_requires_exact_judged_repetition_ids(self):
        first, second = paired_conditions()
        for repetitions in (
            {1: first},
            {1: first, 2: second},
            {1: first, 2: second, 3: first, 4: first},
            {0: first, 1: second, 2: first},
        ):
            with self.subTest(ids=set(repetitions)):
                with self.assertRaisesRegex(ValueError, r"\{1, 2, 3\}"):
                    majority_vote_predictions(repetitions)

    def test_majority_vote_rejects_duplicate_or_misaligned_items(self):
        first, second = paired_conditions()
        with self.assertRaisesRegex(ValueError, "alignment"):
            majority_vote_predictions({1: first, 2: second[:-1], 3: first})
        with self.assertRaisesRegex(ValueError, "duplicate variant_id"):
            majority_vote_predictions({
                1: first + [dict(first[0])],
                2: second,
                3: first,
            })

    def test_exact_binary_metrics_clean_fpr_and_per_type_recall(self):
        records, _ = paired_conditions()

        result = binary_metrics(records)

        self.assertEqual(result["confusion_matrix"], {
            "supported": {"supported": 3, "unsupported": 1},
            "unsupported": {"supported": 1, "unsupported": 3},
        })
        self.assertEqual(result["macro_f1"], 0.75)
        self.assertEqual(result["balanced_accuracy"], 0.75)
        self.assertEqual(result["unsupported_recall"], 0.75)
        self.assertEqual(result["clean_false_positive_rate"], 0.25)
        self.assertEqual(
            result["recall_by_corruption_type"],
            {
                "invalid_id": 1.0,
                "citation_swap": 1.0,
                "causal_overclaim": 1.0,
                "entity_swap": 0.0,
            },
        )

    def test_binary_metrics_fail_closed_on_missing_class_or_duplicate_id(self):
        one_class = [
            binary_record(
                "v-1", "source-1", "supported", "supported", "clean_quote"
            )
        ]
        with self.assertRaisesRegex(ValueError, "both binary classes"):
            binary_metrics(one_class)
        duplicate = paired_conditions()[0]
        with self.assertRaisesRegex(ValueError, "duplicate variant_id"):
            binary_metrics(duplicate + [dict(duplicate[0])])

    def test_pair_ids_disambiguate_reused_clean_variant_ids(self):
        records = []
        for index, variant_type in enumerate((
            "invalid_id",
            "citation_swap",
            "causal_overclaim",
            "entity_swap",
        )):
            clean = binary_record(
                "shared-clean",
                "source-shared",
                "supported",
                "supported",
                "clean_quote",
            )
            clean.update({
                "pair_id": f"pair-{index}",
                "pair_role": "clean_control",
            })
            corruption = binary_record(
                f"corruption-{index}",
                "source-shared",
                "unsupported",
                "unsupported",
                variant_type,
            )
            corruption.update({
                "pair_id": f"pair-{index}",
                "pair_role": "corruption",
            })
            records.extend((clean, corruption))

        result = binary_metrics(records)

        self.assertEqual(result["record_count"], 8)
        self.assertEqual(result["macro_f1"], 1.0)

    def test_macro_difference_requires_exact_id_gold_and_source_alignment(self):
        first, second = paired_conditions()
        self.assertEqual(macro_f1_difference(first, second), 0.25)
        second[0]["source_id"] = "wrong-source"
        with self.assertRaisesRegex(ValueError, "alignment"):
            macro_f1_difference(first, second)

    def test_exact_mcnemar_uses_majority_voted_pairs(self):
        first, second = paired_conditions()

        result = exact_mcnemar(first, second)

        self.assertEqual(result["a_correct_b_wrong"], 3)
        self.assertEqual(result["a_wrong_b_correct"], 1)
        self.assertEqual(result["discordant"], 4)
        self.assertEqual(result["p_value"], 0.625)


class ClusterInferenceTests(unittest.TestCase):
    def test_bootstrap_resamples_source_clusters_deterministically(self):
        first, second = paired_conditions()

        result = cluster_bootstrap_difference(
            first, second, seed=17, draws=200
        )
        repeated = cluster_bootstrap_difference(
            list(reversed(first)),
            list(reversed(second)),
            seed=17,
            draws=200,
        )

        self.assertEqual(result, repeated)
        self.assertEqual(result["cluster_count"], 4)
        self.assertEqual(result["draws"], 200)
        self.assertEqual(result["observed_difference"], 0.25)
        self.assertLessEqual(result["ci_low"], result["ci_high"])

    def test_cluster_permutation_enumerates_small_cluster_set(self):
        first, second = paired_conditions()

        result = cluster_permutation_test(
            first, second, seed=17, draws=999
        )

        self.assertEqual(result["cluster_count"], 4)
        self.assertEqual(result["method"], "exact_enumeration")
        self.assertEqual(result["draws"], 16)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)

    def test_cluster_inference_rejects_nonpositive_or_excessive_draws(self):
        first, second = paired_conditions()
        with self.assertRaisesRegex(ValueError, "draws"):
            cluster_bootstrap_difference(first, second, seed=1, draws=0)
        with self.assertRaisesRegex(ValueError, "bounded"):
            cluster_permutation_test(
                first, second, seed=1, draws=100001
            )

    def test_large_cluster_permutation_requires_preregistered_draw_count(self):
        first = []
        second = []
        for index in range(21):
            variant_type = (
                "invalid_id",
                "citation_swap",
                "causal_overclaim",
                "entity_swap",
            )[index % 4]
            gold = "unsupported"
            first.append(
                binary_record(
                    f"v-{index}", f"source-{index}", gold,
                    "unsupported", variant_type,
                )
            )
            second.append(
                binary_record(
                    f"v-{index}", f"source-{index}", gold,
                    "supported", variant_type,
                )
            )

        with self.assertRaisesRegex(ValueError, "100000"):
            cluster_permutation_test(first, second, seed=17, draws=10)

    def test_monte_carlo_zero_extremes_uses_plus_one_correction(self):
        first = []
        second = []
        for index in range(21):
            first.append(binary_record(
                f"v-{index}", f"source-{index}", "unsupported",
                "unsupported", "entity_swap",
            ))
            second.append(binary_record(
                f"v-{index}", f"source-{index}", "unsupported",
                "supported", "entity_swap",
            ))
        non_extreme_mask = (1 << 10) - 1
        with patch.object(
            __import__("random").Random,
            "getrandbits",
            return_value=non_extreme_mask,
        ):
            result = cluster_permutation_test(first, second, seed=17)

        self.assertEqual(result["method"], "seeded_monte_carlo")
        self.assertEqual(result["extreme_draws"], 0)
        self.assertEqual(result["p_value"], round(1 / 100001, 6))


class BundleTests(unittest.TestCase):
    @staticmethod
    def predictions_for(bundles):
        return [
            {
                "variant_id": member["variant_id"],
                "source_id": member["source_id"],
                "prediction": "supported",
            }
            for bundle in bundles
            for member in bundle["members"]
        ]

    def test_integrity_decision_mapping(self):
        self.assertEqual(integrity_decision(0), "advance")
        self.assertEqual(integrity_decision(1), "hold")
        self.assertEqual(integrity_decision(2), "reject")
        self.assertEqual(integrity_decision(4), "reject")
        with self.assertRaises(ValueError):
            integrity_decision(-1)

    def test_pilot_bundles_are_deterministic_balanced_and_source_unique(self):
        variants = variant_manifest(40)
        manifest = split_manifest("pilot", 40)

        first = build_bundles(variants, manifest, seed=20260728)
        second = build_bundles(
            list(reversed(variants)), manifest, seed=20260728
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 9)
        self.assertEqual(
            {label: sum(b["gold"] == label for b in first) for label in (
                "advance", "hold", "reject"
            )},
            {"advance": 3, "hold": 3, "reject": 3},
        )
        all_sources = [
            member["source_id"]
            for bundle in first
            for member in bundle["members"]
        ]
        self.assertEqual(len(all_sources), 36)
        self.assertEqual(len(set(all_sources)), 36)
        for bundle in first:
            self.assertEqual(len(bundle["members"]), 4)
            self.assertEqual(len(bundle["bundle_sha256"]), 64)
            corruptions = [
                member for member in bundle["members"]
                if member["variant_type"] != "clean_quote"
            ]
            self.assertEqual(
                len(corruptions),
                {"advance": 0, "hold": 1, "reject": 2}[bundle["gold"]],
            )
            self.assertTrue(
                all(
                    member["variant_type"] in SEMANTIC_TYPES
                    and member["evidence_id"] != "PMID:00000000"
                    for member in corruptions
                )
            )
            for member in bundle["members"]:
                self.assertTrue({
                    "variant_id",
                    "source_id",
                    "statement",
                    "evidence_id",
                    "title",
                    "abstract",
                    "exists",
                    "variant_type",
                    "gold",
                }.issubset(member))

    def test_confirmation_uses_all_120_sources_once(self):
        bundles = build_bundles(
            variant_manifest(120),
            split_manifest("confirmation", 120),
            seed=17,
        )
        self.assertEqual(len(bundles), 30)
        self.assertEqual(
            len({
                member["source_id"]
                for bundle in bundles
                for member in bundle["members"]
            }),
            120,
        )

    def test_bundle_construction_fails_on_shortfall_or_cross_split_source(self):
        with self.assertRaisesRegex(ValueError, "36 eligible"):
            build_bundles(
                variant_manifest(35),
                split_manifest("pilot", 35),
                seed=17,
            )
        variants = variant_manifest(40)
        with self.assertRaisesRegex(ValueError, "outside pilot"):
            manifest = split_manifest("pilot", 39)
            build_bundles(variants, manifest, seed=17)

    def test_bundles_exclude_semantic_corruptions_without_valid_ids(self):
        variants = variant_manifest(36)
        for item in variants:
            if (
                item["source_id"] == "source-000"
                and item["variant_type"] in SEMANTIC_TYPES
            ):
                item["exists"] = False

        with self.assertRaisesRegex(ValueError, "36 eligible"):
            build_bundles(
                variants, split_manifest("pilot", 36), seed=17
            )

    def test_candidate_pool_enforces_complete_semantic_contract(self):
        corruptions = {
            "exists": lambda item: item.update(exists=False),
            "pmid": lambda item: item.update(evidence_id="DOI:10/example"),
            "abstract": lambda item: item.update(abstract=" "),
            "gold": lambda item: item.update(gold="supported"),
            "audit": lambda item: item.update(mutation_audit={}),
            "validation": lambda item: item["mutation_audit"].update(
                validation={"safe": False}
            ),
        }
        for label, mutate in corruptions.items():
            with self.subTest(label=label):
                variants = variant_manifest(36)
                for item in variants:
                    if (
                        item["source_id"] == "source-000"
                        and item["variant_type"] in SEMANTIC_TYPES
                    ):
                        mutate(item)
                with self.assertRaisesRegex(ValueError, "36 eligible"):
                    build_bundles(
                        variants,
                        split_manifest("pilot", 36),
                        seed=17,
                    )

    def test_final_contaminated_choice_reuses_only_validated_candidates(self):
        variants = variant_manifest(36)
        for item in variants:
            if item["variant_type"] == "causal_overclaim":
                item["mutation_audit"]["validation"]["safe"] = False
            elif item["variant_type"] == "entity_swap":
                item["abstract"] = ""

        bundles = build_bundles(
            variants, split_manifest("pilot", 36), seed=17
        )

        contaminated = [
            member
            for bundle in bundles
            for member in bundle["members"]
            if member["variant_type"] != "clean_quote"
        ]
        self.assertTrue(contaminated)
        self.assertEqual(
            {member["variant_type"] for member in contaminated},
            {"citation_swap"},
        )

    def test_clean_candidate_requires_complete_valid_source_contract(self):
        corruptions = {
            "exists": lambda item: item.pop("exists"),
            "reserved PMID": lambda item: item.update(
                evidence_id="PMID:00000000"
            ),
            "PMID syntax": lambda item: item.update(
                evidence_id="DOI:10/example"
            ),
            "non-empty abstract": lambda item: item.update(abstract=" "),
        }
        for message, mutate in corruptions.items():
            with self.subTest(message=message):
                variants = variant_manifest(36)
                clean = next(
                    item for item in variants
                    if item["source_id"] == "source-000"
                    and item["variant_type"] == "clean_quote"
                )
                mutate(clean)
                with self.assertRaisesRegex(ValueError, "36 eligible"):
                    build_bundles(
                        variants,
                        split_manifest("pilot", 36),
                        seed=17,
                    )

    def test_split_manifest_is_mandatory_hashed_and_disjoint(self):
        variants = variant_manifest(40)
        with self.assertRaisesRegex(ValueError, "manifest"):
            build_bundles(variants, "pilot", seed=17)
        missing_hash = split_manifest("pilot", 40)
        del missing_hash["split_sha256"]
        with self.assertRaisesRegex(ValueError, "hash"):
            build_bundles(variants, missing_hash, seed=17)
        bad_hash = split_manifest("pilot", 40)
        bad_hash["split_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash"):
            build_bundles(variants, bad_hash, seed=17)
        overlap = split_manifest("pilot", 40)
        overlap["confirmation_ids"][0] = overlap["pilot_ids"][0]
        overlap["split_sha256"] = compute_split_manifest_hash(overlap)
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_bundles(variants, overlap, seed=17)

    def test_selection_maximizes_repeat_and_hypothesis_representation(self):
        variants = variant_manifest(40)
        for item in variants:
            item["repeat"] = "rep_01"
            item["hypothesis_id"] = "common"
        special_sources = []
        for index in range(4):
            source_id = f"source-{36 + index:03d}"
            special_sources.append(source_id)
            for item in variants:
                if item["source_id"] == source_id:
                    item["repeat"] = f"rep_{index + 2:02d}"
                    item["hypothesis_id"] = f"special-{index}"

        bundles = build_bundles(
            variants, split_manifest("pilot", 40), seed=17
        )

        selected = {
            member["source_id"]
            for bundle in bundles
            for member in bundle["members"]
        }
        self.assertTrue(set(special_sources).issubset(selected))
        summary = bundles[0]["stratum_summary"]
        self.assertEqual(summary["selected_repeat_count"], 5)
        self.assertEqual(summary["selected_hypothesis_count"], 5)
        self.assertEqual(
            summary["method"],
            "deterministic_greedy_coverage_and_allocation",
        )

    def test_rare_stratum_is_distributed_across_classes_and_bundles(self):
        variants = variant_manifest(36)
        rare_sources = {f"source-{index:03d}" for index in range(30, 36)}
        for item in variants:
            if item["source_id"] in rare_sources:
                item["repeat"] = "rep_rare"
                item["hypothesis_id"] = "hypothesis-rare"
            else:
                item["repeat"] = "rep_common"
                item["hypothesis_id"] = "hypothesis-common"

        bundles = build_bundles(
            variants, split_manifest("pilot", 36), seed=17
        )

        rare_by_class = {
            label: sum(
                member["source_id"] in rare_sources
                for bundle in bundles if bundle["gold"] == label
                for member in bundle["members"]
            )
            for label in ("advance", "hold", "reject")
        }
        rare_bundle_counts = [
            sum(
                member["source_id"] in rare_sources
                for member in bundle["members"]
            )
            for bundle in bundles
        ]
        self.assertEqual(rare_by_class, {
            "advance": 2,
            "hold": 2,
            "reject": 2,
        })
        self.assertEqual(sorted(rare_bundle_counts), [0, 0, 0, 1, 1, 1, 1, 1, 1])
        summary = bundles[0]["stratum_summary"]
        self.assertEqual(
            {
                label: values["repeat_counts"]["rep_rare"]
                for label, values in summary["by_class"].items()
            },
            rare_by_class,
        )
        self.assertEqual(
            {
                label: values["hypothesis_counts"]["hypothesis-rare"]
                for label, values in summary["by_class"].items()
            },
            rare_by_class,
        )
        self.assertEqual(len(summary["by_bundle"]), 9)

    def test_bundle_metrics_include_all_prespecified_endpoints(self):
        manifest = split_manifest("pilot", 40)
        bundles = build_bundles(variant_manifest(40), manifest, seed=17)
        predictions = []
        for bundle in bundles:
            for member in bundle["members"]:
                prediction = (
                    "unsupported"
                    if member["variant_type"] != "clean_quote"
                    else "supported"
                )
                predictions.append({
                    "variant_id": member["variant_id"],
                    "source_id": member["source_id"],
                    "prediction": prediction,
                })

        result = bundle_metrics(predictions, bundles, manifest)

        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["exact_match_accuracy"], 1.0)
        self.assertEqual(result["erroneous_advance_rate"], 0.0)
        self.assertEqual(result["reject_recall"], 1.0)
        self.assertEqual(result["weighted_ordinal_distance"], 0.0)
        self.assertEqual(
            result["transition_matrix"],
            {
                "advance": {"advance": 3, "hold": 0, "reject": 0},
                "hold": {"advance": 0, "hold": 3, "reject": 0},
                "reject": {"advance": 0, "hold": 0, "reject": 3},
            },
        )

    def test_paired_bundle_inference_uses_unique_bundles_as_clusters(self):
        manifest = split_manifest("pilot", 40)
        bundles = build_bundles(variant_manifest(40), manifest, seed=17)
        full = []
        id_only = []
        for bundle in bundles:
            for member in bundle["members"]:
                base = {
                    "variant_id": member["variant_id"],
                    "source_id": member["source_id"],
                }
                full.append({
                    **base,
                    "prediction": (
                        "supported"
                        if member["variant_type"] == "clean_quote"
                        else "unsupported"
                    ),
                })
                id_only.append({**base, "prediction": "supported"})

        effect = bundle_macro_f1_difference(
            full, id_only, bundles, manifest
        )
        bootstrap = bundle_bootstrap_difference(
            full, id_only, bundles, manifest, seed=5, draws=200
        )
        permutation = bundle_permutation_test(
            full, id_only, bundles, manifest, seed=5, draws=999
        )

        self.assertGreater(effect, 0)
        self.assertEqual(bootstrap["cluster_count"], 9)
        self.assertEqual(bootstrap["draws"], 200)
        self.assertEqual(bootstrap["observed_difference"], effect)
        self.assertEqual(permutation["cluster_count"], 9)
        self.assertEqual(permutation["method"], "exact_enumeration")
        self.assertEqual(permutation["draws"], 512)
        self.assertGreaterEqual(permutation["p_value"], 0)
        self.assertLessEqual(permutation["p_value"], 1)

    def test_bundle_metrics_require_exact_unique_id_alignment(self):
        manifest = split_manifest("pilot", 40)
        bundles = build_bundles(variant_manifest(40), manifest, seed=17)
        predictions = [
            {
                "variant_id": member["variant_id"],
                "source_id": member["source_id"],
                "prediction": "supported",
            }
            for bundle in bundles
            for member in bundle["members"]
        ]
        with self.assertRaisesRegex(ValueError, "alignment"):
            bundle_metrics(predictions[:-1], bundles, manifest)
        with self.assertRaisesRegex(ValueError, "duplicate variant_id"):
            bundle_metrics(
                predictions + [dict(predictions[0])], bundles, manifest
            )

    def test_bundle_metrics_reject_tampered_bundle_hash(self):
        manifest = split_manifest("pilot", 40)
        bundles = build_bundles(variant_manifest(40), manifest, seed=17)
        predictions = [
            {
                "variant_id": member["variant_id"],
                "source_id": member["source_id"],
                "prediction": "supported",
            }
            for bundle in bundles
            for member in bundle["members"]
        ]
        bundles[0]["gold"] = "reject"

        with self.assertRaisesRegex(ValueError, "hash"):
            bundle_metrics(predictions, bundles, manifest)

    def test_bundle_metrics_requires_valid_split_manifest(self):
        manifest = split_manifest("pilot", 40)
        bundles = build_bundles(variant_manifest(40), manifest, seed=17)
        predictions = self.predictions_for(bundles)
        with self.assertRaisesRegex(ValueError, "manifest"):
            bundle_metrics(predictions, bundles)
        overlap = copy.deepcopy(manifest)
        overlap["confirmation_ids"][0] = overlap["pilot_ids"][0]
        overlap["split_sha256"] = compute_split_manifest_hash(overlap)
        with self.assertRaisesRegex(ValueError, "overlap"):
            bundle_metrics(predictions, bundles, overlap)

    def test_bundle_metrics_rejects_rehashed_structural_corruption(self):
        manifest = split_manifest("pilot", 40)
        original = build_bundles(variant_manifest(40), manifest, seed=17)
        cases = {}

        wrong_class = copy.deepcopy(original)
        wrong_class[0]["gold"] = "reject"
        wrong_class[0]["bundle_sha256"] = compute_bundle_hash(wrong_class[0])
        cases["injected corruption count"] = wrong_class

        reused = copy.deepcopy(original)
        reused[1]["members"][0]["source_id"] = reused[0]["members"][0][
            "source_id"
        ]
        reused[1]["bundle_sha256"] = compute_bundle_hash(reused[1])
        cases["source reused"] = reused

        duplicate_within = copy.deepcopy(original)
        duplicate_within[0]["members"][1]["source_id"] = (
            duplicate_within[0]["members"][0]["source_id"]
        )
        duplicate_within[0]["bundle_sha256"] = compute_bundle_hash(
            duplicate_within[0]
        )
        cases["four unique sources"] = duplicate_within

        invalid_contamination = copy.deepcopy(original)
        contaminated_bundle = next(
            bundle for bundle in invalid_contamination
            if bundle["gold"] != "advance"
        )
        contaminated = next(
            member for member in contaminated_bundle["members"]
            if member["variant_type"] != "clean_quote"
        )
        contaminated.update({
            "variant_type": "invalid_id",
            "exists": False,
            "evidence_id": "PMID:00000000",
        })
        contaminated_bundle["bundle_sha256"] = compute_bundle_hash(
            contaminated_bundle
        )
        cases["semantic corruption"] = invalid_contamination

        invalid_clean = copy.deepcopy(original)
        clean_member = next(
            member for member in invalid_clean[0]["members"]
            if member["variant_type"] == "clean_quote"
        )
        clean_member["gold"] = "unsupported"
        invalid_clean[0]["bundle_sha256"] = compute_bundle_hash(
            invalid_clean[0]
        )
        cases["clean member"] = invalid_clean

        for message, bundles in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    bundle_metrics(
                        self.predictions_for(bundles), bundles, manifest
                    )

    def test_bundle_metrics_rejects_source_outside_requested_split(self):
        manifest = split_manifest("pilot", 40)
        bundles = build_bundles(variant_manifest(40), manifest, seed=17)
        bundles[0]["members"][0]["source_id"] = "confirmation-000"
        bundles[0]["bundle_sha256"] = compute_bundle_hash(bundles[0])
        with self.assertRaisesRegex(ValueError, "outside pilot"):
            bundle_metrics(
                self.predictions_for(bundles), bundles, manifest
            )


if __name__ == "__main__":
    unittest.main()
