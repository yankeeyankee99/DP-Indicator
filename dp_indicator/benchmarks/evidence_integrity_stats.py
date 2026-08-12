"""Dependency-free statistics for the evidence-integrity benchmark."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict


BINARY_LABELS = ("supported", "unsupported")
CORRUPTION_TYPES = (
    "invalid_id",
    "citation_swap",
    "causal_overclaim",
    "entity_swap",
)
SEMANTIC_TYPES = ("citation_swap", "causal_overclaim", "entity_swap")
BUNDLE_LABELS = ("advance", "hold", "reject")
_BUNDLE_COUNTS = {"pilot": 9, "confirmation": 30}
_MAX_BOOTSTRAP_DRAWS = 1_000_000
_MAX_PERMUTATION_DRAWS = 100_000


def _rounded(value: float) -> float:
    return round(value, 6)


def holm_adjust(p_values: dict[str, float]) -> dict[str, dict[str, float]]:
    """Return monotone Holm-adjusted p-values with raw values retained."""
    if not isinstance(p_values, dict) or not p_values:
        raise ValueError("p_values must be a non-empty mapping")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < 0
        or value > 1
        for _, value in ordered
    ):
        raise ValueError("p-values must be numeric values from 0 to 1")
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for index, (name, raw) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * float(raw)))
        adjusted[name] = {
            "raw_p": _rounded(float(raw)),
            "holm_adjusted_p": _rounded(running),
        }
    return {name: adjusted[name] for name in sorted(adjusted)}


def _require_nonempty_records(records: list[dict], name: str) -> None:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{name} must be a non-empty list")


def _indexed_records(records: list[dict], name: str) -> dict[str, dict]:
    _require_nonempty_records(records, name)
    indexed: dict[str, dict] = {}
    variant_pair_flags: dict[str, list[bool]] = defaultdict(list)
    required = {
        "variant_id",
        "source_id",
        "gold",
        "prediction",
        "variant_type",
    }
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{name} contains a non-object record")
        missing = required.difference(record)
        if missing:
            raise ValueError(f"{name} record missing fields: {sorted(missing)}")
        variant_id = record["variant_id"]
        source_id = record["source_id"]
        if not isinstance(variant_id, str) or not variant_id:
            raise ValueError(f"{name} has invalid variant_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"{name} has invalid source_id")
        if record["gold"] not in BINARY_LABELS:
            raise ValueError(f"{name} has invalid gold label")
        if record["prediction"] not in BINARY_LABELS:
            raise ValueError(f"{name} has invalid prediction label")
        if not isinstance(record["variant_type"], str):
            raise ValueError(f"{name} has invalid variant_type")
        has_pair_id = "pair_id" in record
        has_pair_role = "pair_role" in record
        if has_pair_id != has_pair_role:
            raise ValueError(
                f"{name} record must provide pair_id and pair_role together"
            )
        if has_pair_id:
            pair_id = record["pair_id"]
            pair_role = record["pair_role"]
            if not isinstance(pair_id, str) or not pair_id:
                raise ValueError(f"{name} has invalid pair_id")
            if pair_role not in ("clean_control", "corruption"):
                raise ValueError(f"{name} has invalid pair_role")
            observation_id = f"pair:{pair_id}:{pair_role}"
        else:
            observation_id = f"variant:{variant_id}"
        if observation_id in indexed:
            if not has_pair_id:
                raise ValueError(
                    f"{name} has duplicate variant_id: {variant_id}"
                )
            raise ValueError(
                f"{name} has duplicate pair observation: {observation_id}"
            )
        indexed[observation_id] = record
        variant_pair_flags[variant_id].append(has_pair_id)
    for variant_id, flags in variant_pair_flags.items():
        if len(flags) > 1 and not all(flags):
            raise ValueError(
                f"{name} has ambiguous duplicate variant_id: {variant_id}"
            )
    return indexed


def _aligned_pair(
    a: list[dict],
    b: list[dict],
) -> tuple[list[dict], list[dict]]:
    a_by_id = _indexed_records(a, "condition a")
    b_by_id = _indexed_records(b, "condition b")
    if set(a_by_id) != set(b_by_id):
        raise ValueError("condition ID alignment mismatch")
    first = []
    second = []
    for observation_id in sorted(a_by_id):
        left = a_by_id[observation_id]
        right = b_by_id[observation_id]
        for field in ("variant_id", "source_id", "gold", "variant_type"):
            if left[field] != right[field]:
                raise ValueError(
                    "condition alignment mismatch for "
                    f"{observation_id}: {field}"
                )
        first.append(left)
        second.append(right)
    return first, second


def majority_vote_predictions(repetitions: dict[int, list[dict]]) -> dict:
    """Aggregate repeated model calls before any inferential analysis."""
    if not isinstance(repetitions, dict) or set(repetitions) != {1, 2, 3}:
        raise ValueError(
            "judged conditions require repetition IDs {1, 2, 3}"
        )
    indexed = [
        _indexed_records(repetitions[repetition_id], (
            f"repetition {repetition_id}"
        ))
        for repetition_id in (1, 2, 3)
    ]
    expected_ids = set(indexed[0])
    for repetition in indexed[1:]:
        if set(repetition) != expected_ids:
            raise ValueError("repetition ID alignment mismatch")
    output = []
    disagreement_count = 0
    for observation_id in sorted(expected_ids):
        reference = indexed[0][observation_id]
        votes = []
        for repetition in indexed:
            current = repetition[observation_id]
            for field in ("variant_id", "source_id", "gold", "variant_type"):
                if current[field] != reference[field]:
                    raise ValueError(
                        "repetition alignment mismatch for "
                        f"{observation_id}: {field}"
                    )
            votes.append(current["prediction"])
        counts = Counter(votes)
        prediction, count = counts.most_common(1)[0]
        if count <= len(votes) // 2:
            raise ValueError("majority vote did not produce a strict majority")
        if len(counts) > 1:
            disagreement_count += 1
        output.append({**reference, "prediction": prediction})
    return {
        "records": output,
        "repetition_ids": [1, 2, 3],
        "disagreement_count": disagreement_count,
        "disagreement_rate": _rounded(disagreement_count / len(output)),
    }


def _binary_values(records: list[dict]) -> dict:
    counts = {
        gold: {prediction: 0 for prediction in BINARY_LABELS}
        for gold in BINARY_LABELS
    }
    for record in records:
        counts[record["gold"]][record["prediction"]] += 1
    tp = counts["supported"]["supported"]
    fn = counts["supported"]["unsupported"]
    fp = counts["unsupported"]["supported"]
    tn = counts["unsupported"]["unsupported"]

    def recall(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def f1(correct: int, false_a: int, false_b: int) -> float:
        denominator = 2 * correct + false_a + false_b
        return 2 * correct / denominator if denominator else 0.0

    supported_recall = recall(tp, tp + fn)
    unsupported_recall = recall(tn, tn + fp)
    return {
        "confusion_matrix": counts,
        "macro_f1": (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2,
        "balanced_accuracy": (
            supported_recall + unsupported_recall
        ) / 2,
        "unsupported_recall": unsupported_recall,
    }


def binary_metrics(records: list[dict]) -> dict:
    """Compute exact binary endpoints from majority-voted item records."""
    indexed = _indexed_records(records, "records")
    ordered = [indexed[key] for key in sorted(indexed)]
    gold_counts = Counter(item["gold"] for item in ordered)
    if any(gold_counts[label] == 0 for label in BINARY_LABELS):
        raise ValueError("records must contain both binary classes")
    clean = [
        item for item in ordered if item["variant_type"] == "clean_quote"
    ]
    if not clean:
        raise ValueError("records are incomplete: no clean_quote controls")
    by_type = {}
    for variant_type in CORRUPTION_TYPES:
        items = [
            item for item in ordered
            if item["variant_type"] == variant_type
        ]
        if not items:
            raise ValueError(
                f"records are incomplete: no {variant_type} corruptions"
            )
        if any(item["gold"] != "unsupported" for item in items):
            raise ValueError(f"{variant_type} has invalid gold labels")
        by_type[variant_type] = sum(
            item["prediction"] == "unsupported" for item in items
        ) / len(items)
    if any(item["gold"] != "supported" for item in clean):
        raise ValueError("clean_quote has invalid gold labels")

    values = _binary_values(ordered)
    return {
        "record_count": len(ordered),
        "confusion_matrix": values["confusion_matrix"],
        "macro_f1": _rounded(values["macro_f1"]),
        "balanced_accuracy": _rounded(values["balanced_accuracy"]),
        "unsupported_recall": _rounded(values["unsupported_recall"]),
        "clean_false_positive_rate": _rounded(
            sum(item["prediction"] == "unsupported" for item in clean)
            / len(clean)
        ),
        "recall_by_corruption_type": {
            key: _rounded(value) for key, value in by_type.items()
        },
    }


def macro_f1_difference(a: list[dict], b: list[dict]) -> float:
    first, second = _aligned_pair(a, b)
    return _rounded(
        _binary_values(first)["macro_f1"]
        - _binary_values(second)["macro_f1"]
    )


def exact_mcnemar(a: list[dict], b: list[dict]) -> dict:
    """Return the two-sided exact paired McNemar test."""
    first, second = _aligned_pair(a, b)
    a_correct_b_wrong = 0
    a_wrong_b_correct = 0
    for left, right in zip(first, second):
        left_correct = left["prediction"] == left["gold"]
        right_correct = right["prediction"] == right["gold"]
        if left_correct and not right_correct:
            a_correct_b_wrong += 1
        elif right_correct and not left_correct:
            a_wrong_b_correct += 1
    discordant = a_correct_b_wrong + a_wrong_b_correct
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(a_correct_b_wrong, a_wrong_b_correct)
        tail_numerator = sum(
            math.comb(discordant, value)
            for value in range(smaller + 1)
        )
        p_value = min(1.0, 2.0 * tail_numerator / (2 ** discordant))
    return {
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant": discordant,
        "p_value": _rounded(p_value),
    }


def _cluster_pair(
    a: list[dict],
    b: list[dict],
) -> tuple[list[str], dict[str, list[dict]], dict[str, list[dict]]]:
    first, second = _aligned_pair(a, b)
    a_clusters: dict[str, list[dict]] = defaultdict(list)
    b_clusters: dict[str, list[dict]] = defaultdict(list)
    for left, right in zip(first, second):
        a_clusters[left["source_id"]].append(left)
        b_clusters[right["source_id"]].append(right)
    clusters = sorted(a_clusters)
    if not clusters:
        raise ValueError("cluster inference requires source clusters")
    return clusters, a_clusters, b_clusters


def _raw_macro_difference(a: list[dict], b: list[dict]) -> float:
    return (
        _binary_values(a)["macro_f1"]
        - _binary_values(b)["macro_f1"]
    )


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1 - fraction)
        + sorted_values[upper] * fraction
    )


def cluster_bootstrap_difference(
    a: list[dict],
    b: list[dict],
    seed: int,
    draws: int = 10_000,
) -> dict:
    if not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if draws > _MAX_BOOTSTRAP_DRAWS:
        raise ValueError("bootstrap draws exceed bounded maximum")
    clusters, a_clusters, b_clusters = _cluster_pair(a, b)
    rng = random.Random(seed)
    differences = []
    for _ in range(draws):
        selected = [rng.choice(clusters) for _ in clusters]
        first = [
            record
            for cluster in selected
            for record in a_clusters[cluster]
        ]
        second = [
            record
            for cluster in selected
            for record in b_clusters[cluster]
        ]
        differences.append(_raw_macro_difference(first, second))
    differences.sort()
    return {
        "observed_difference": macro_f1_difference(a, b),
        "cluster_count": len(clusters),
        "draws": draws,
        "seed": seed,
        "ci_low": _rounded(_percentile(differences, 0.025)),
        "ci_high": _rounded(_percentile(differences, 0.975)),
    }


def cluster_permutation_test(
    a: list[dict],
    b: list[dict],
    seed: int,
    draws: int = 100_000,
) -> dict:
    if not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if draws > _MAX_PERMUTATION_DRAWS:
        raise ValueError("permutation draws exceed bounded maximum")
    clusters, a_clusters, b_clusters = _cluster_pair(a, b)
    observed = _raw_macro_difference(
        [item for cluster in clusters for item in a_clusters[cluster]],
        [item for cluster in clusters for item in b_clusters[cluster]],
    )
    if len(clusters) > 20 and draws != _MAX_PERMUTATION_DRAWS:
        raise ValueError(
            "more than 20 clusters require exactly 100000 permutation draws"
        )
    if len(clusters) <= 20:
        masks = range(2 ** len(clusters))
        actual_draws = 2 ** len(clusters)
        method = "exact_enumeration"
    else:
        rng = random.Random(seed)
        masks = [
            rng.getrandbits(len(clusters))
            for _ in range(draws)
        ]
        actual_draws = draws
        method = "seeded_monte_carlo"
    extreme = 0
    tolerance = 1e-12
    for mask in masks:
        first = []
        second = []
        for index, cluster in enumerate(clusters):
            if mask & (1 << index):
                first.extend(b_clusters[cluster])
                second.extend(a_clusters[cluster])
            else:
                first.extend(a_clusters[cluster])
                second.extend(b_clusters[cluster])
        permuted = _raw_macro_difference(first, second)
        if abs(permuted) + tolerance >= abs(observed):
            extreme += 1
    if method == "exact_enumeration":
        p_value = extreme / actual_draws
    else:
        p_value = (extreme + 1) / (actual_draws + 1)
    return {
        "observed_difference": _rounded(observed),
        "cluster_count": len(clusters),
        "draws": actual_draws,
        "seed": seed,
        "method": method,
        "extreme_draws": extreme,
        "p_value": _rounded(p_value),
    }


def integrity_decision(unsupported_count: int) -> str:
    if (
        not isinstance(unsupported_count, int)
        or isinstance(unsupported_count, bool)
        or unsupported_count < 0
        or unsupported_count > 4
    ):
        raise ValueError("unsupported_count must be an integer from 0 to 4")
    if unsupported_count == 0:
        return "advance"
    if unsupported_count == 1:
        return "hold"
    return "reject"


def _canonical_hash(value: dict, excluded_key: str) -> str:
    payload = {
        key: item for key, item in value.items() if key != excluded_key
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def compute_split_manifest_hash(manifest: dict) -> str:
    if not isinstance(manifest, dict):
        raise ValueError("split manifest must be an object")
    return _canonical_hash(manifest, "split_sha256")


def _split_contract(
    items: list[dict],
    manifest: dict | None,
) -> tuple[str, str, set[str]]:
    if not isinstance(manifest, dict):
        raise ValueError("a full split manifest is required")
    split_name = manifest.get("name")
    if split_name not in _BUNDLE_COUNTS:
        raise ValueError("split manifest requires name pilot or confirmation")
    pilot_ids = manifest.get("pilot_ids")
    confirmation_ids = manifest.get("confirmation_ids")
    if not isinstance(pilot_ids, list) or not isinstance(
        confirmation_ids, list
    ):
        raise ValueError(
            "split manifest requires pilot_ids and confirmation_ids"
        )
    if any(not isinstance(value, str) or not value for value in (
        *pilot_ids,
        *confirmation_ids,
    )):
        raise ValueError("split manifest contains an invalid source ID")
    if len(pilot_ids) != len(set(pilot_ids)) or len(
        confirmation_ids
    ) != len(set(confirmation_ids)):
        raise ValueError("split manifest contains duplicate source IDs")
    if set(pilot_ids).intersection(confirmation_ids):
        raise ValueError("pilot and confirmation source IDs overlap")
    manifest_hash = manifest.get("split_sha256")
    if (
        not isinstance(manifest_hash, str)
        or manifest_hash != compute_split_manifest_hash(manifest)
    ):
        raise ValueError("split manifest hash mismatch")
    allowed = set(
        pilot_ids if split_name == "pilot" else confirmation_ids
    )
    observed = {
        item.get("source_id")
        for item in items
        if isinstance(item, dict)
    }
    outside = observed.difference(allowed)
    if outside:
        raise ValueError(
            f"items contain source outside {split_name} split"
        )
    return split_name, manifest_hash, allowed


def _variant_index(variants: list[dict]) -> dict[str, dict[str, dict]]:
    _require_nonempty_records(variants, "variants")
    by_source: dict[str, dict[str, dict]] = defaultdict(dict)
    variant_ids = set()
    required = {
        "variant_id",
        "source_id",
        "variant_type",
        "gold",
        "evidence_id",
    }
    for item in variants:
        if not isinstance(item, dict):
            raise ValueError("variants contain a non-object")
        missing = required.difference(item)
        if missing:
            raise ValueError(f"variant missing fields: {sorted(missing)}")
        variant_id = item["variant_id"]
        source_id = item["source_id"]
        variant_type = item["variant_type"]
        if not all(
            isinstance(value, str) and value
            for value in (variant_id, source_id, variant_type)
        ):
            raise ValueError("variant has invalid identifier fields")
        if variant_id in variant_ids:
            raise ValueError(f"duplicate variant_id: {variant_id}")
        variant_ids.add(variant_id)
        if variant_type in by_source[source_id]:
            raise ValueError(
                f"duplicate {variant_type} variant for source {source_id}"
            )
        by_source[source_id][variant_type] = item
    return by_source


def _valid_pmid(value: object) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"PMID:\d+", value) is not None
        and value != "PMID:00000000"
    )


def _valid_mutation_audit(item: dict) -> bool:
    audit = item.get("mutation_audit")
    if not isinstance(audit, dict) or audit.get("operator") != item.get(
        "variant_type"
    ):
        return False
    validation = audit.get("validation")
    return (
        isinstance(validation, dict)
        and bool(validation)
        and all(value is True for value in validation.values())
    )


def _valid_semantic(item: dict) -> bool:
    return (
        item.get("variant_type") in SEMANTIC_TYPES
        and item.get("exists") is True
        and _valid_pmid(item.get("evidence_id"))
        and isinstance(item.get("abstract"), str)
        and bool(item["abstract"].strip())
        and item.get("gold") == "unsupported"
        and _valid_mutation_audit(item)
    )


def _valid_clean(item: dict) -> bool:
    return (
        item.get("variant_type") == "clean_quote"
        and item.get("gold") == "supported"
        and item.get("exists") is True
        and _valid_pmid(item.get("evidence_id"))
        and isinstance(item.get("abstract"), str)
        and bool(item["abstract"].strip())
    )


def _bundle_member(item: dict) -> dict:
    keys = (
        "variant_id",
        "source_id",
        "statement",
        "variant_type",
        "evidence_id",
        "exists",
        "title",
        "abstract",
        "gold",
        "repeat",
        "hypothesis_id",
        "mutation_audit",
        "source_hashes",
        "source_artifact_hashes",
        "source_quote",
        "original_claim",
    )
    return copy.deepcopy({
        key: item.get(key)
        for key in keys
        if key in item
        or key in {
            "statement",
            "title",
            "mutation_audit",
        }
    })


def compute_bundle_hash(bundle: dict) -> str:
    if not isinstance(bundle, dict):
        raise ValueError("bundle must be an object")
    return _canonical_hash(bundle, "bundle_sha256")


def _select_stratified_sources(
    eligible: list[str],
    by_source: dict[str, dict[str, dict]],
    count: int,
    seed: int,
) -> tuple[list[str], dict]:
    metadata = {}
    for source_id in eligible:
        clean = by_source[source_id]["clean_quote"]
        metadata[source_id] = (
            str(clean.get("repeat", "")),
            str(clean.get("hypothesis_id", "")),
        )
    remaining = set(eligible)
    selected = []
    repeat_counts: Counter = Counter()
    hypothesis_counts: Counter = Counter()
    while len(selected) < count:
        def score(source_id: str) -> tuple:
            repeat, hypothesis = metadata[source_id]
            tie = hashlib.sha256(
                f"{seed}|{source_id}|stratum".encode("utf-8")
            ).hexdigest()
            return (
                int(repeat not in repeat_counts)
                + int(hypothesis not in hypothesis_counts),
                int(repeat not in repeat_counts),
                int(hypothesis not in hypothesis_counts),
                -repeat_counts[repeat],
                -hypothesis_counts[hypothesis],
                tie,
            )

        chosen = max(sorted(remaining), key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        repeat, hypothesis = metadata[chosen]
        repeat_counts[repeat] += 1
        hypothesis_counts[hypothesis] += 1
    available_repeats = {metadata[item][0] for item in eligible}
    available_hypotheses = {metadata[item][1] for item in eligible}
    summary = {
        "method": "deterministic_greedy_coverage_and_allocation",
        "eligible_source_count": len(eligible),
        "selected_source_count": len(selected),
        "available_repeat_count": len(available_repeats),
        "available_hypothesis_count": len(available_hypotheses),
        "selected_repeat_count": len(repeat_counts),
        "selected_hypothesis_count": len(hypothesis_counts),
        "selected_repeat_counts": dict(sorted(repeat_counts.items())),
        "selected_hypothesis_counts": dict(
            sorted(hypothesis_counts.items())
        ),
    }
    return selected, summary


def _stratum_counts(
    source_ids: list[str],
    metadata: dict[str, tuple[str, str]],
) -> dict:
    repeat_counts = Counter(metadata[source_id][0] for source_id in source_ids)
    hypothesis_counts = Counter(
        metadata[source_id][1] for source_id in source_ids
    )
    return {
        "source_count": len(source_ids),
        "repeat_counts": dict(sorted(repeat_counts.items())),
        "hypothesis_counts": dict(sorted(hypothesis_counts.items())),
    }


def _allocate_stratified_bundles(
    selected: list[str],
    by_source: dict[str, dict[str, dict]],
    labels: list[str],
    seed: int,
    summary: dict,
    split_name: str,
) -> tuple[list[dict], dict]:
    metadata = {
        source_id: (
            str(by_source[source_id]["clean_quote"].get("repeat", "")),
            str(
                by_source[source_id]["clean_quote"].get(
                    "hypothesis_id", ""
                )
            ),
        )
        for source_id in selected
    }
    slots = [
        {
            "bundle_id": f"{split_name}-bundle-{index + 1:03d}",
            "label": label,
            "source_ids": [],
        }
        for index, label in enumerate(labels)
    ]
    class_repeat_counts = {
        label: Counter() for label in BUNDLE_LABELS
    }
    class_hypothesis_counts = {
        label: Counter() for label in BUNDLE_LABELS
    }
    bundle_repeat_counts = [Counter() for _ in slots]
    bundle_hypothesis_counts = [Counter() for _ in slots]
    for source_id in selected:
        repeat, hypothesis = metadata[source_id]
        candidates = [
            index
            for index, slot in enumerate(slots)
            if len(slot["source_ids"]) < 4
        ]

        def allocation_score(index: int) -> tuple:
            label = slots[index]["label"]
            class_repeat = class_repeat_counts[label][repeat]
            class_hypothesis = class_hypothesis_counts[label][hypothesis]
            bundle_repeat = bundle_repeat_counts[index][repeat]
            bundle_hypothesis = bundle_hypothesis_counts[index][hypothesis]
            tie = hashlib.sha256(
                f"{seed}|{source_id}|{index}|allocation".encode("utf-8")
            ).hexdigest()
            return (
                class_repeat + class_hypothesis,
                max(class_repeat, class_hypothesis),
                bundle_repeat + bundle_hypothesis,
                max(bundle_repeat, bundle_hypothesis),
                class_repeat,
                class_hypothesis,
                bundle_repeat,
                bundle_hypothesis,
                len(slots[index]["source_ids"]),
                tie,
            )

        chosen_index = min(candidates, key=allocation_score)
        chosen = slots[chosen_index]
        chosen["source_ids"].append(source_id)
        label = chosen["label"]
        class_repeat_counts[label][repeat] += 1
        class_hypothesis_counts[label][hypothesis] += 1
        bundle_repeat_counts[chosen_index][repeat] += 1
        bundle_hypothesis_counts[chosen_index][hypothesis] += 1
    if any(len(slot["source_ids"]) != 4 for slot in slots):
        raise ValueError("stratified allocation did not fill every bundle")
    summary = copy.deepcopy(summary)
    summary["by_class"] = {
        label: _stratum_counts(
            [
                source_id
                for slot in slots
                if slot["label"] == label
                for source_id in slot["source_ids"]
            ],
            metadata,
        )
        for label in BUNDLE_LABELS
    }
    summary["by_bundle"] = {
        slot["bundle_id"]: _stratum_counts(slot["source_ids"], metadata)
        for slot in slots
    }
    return slots, summary


def build_bundles(
    variants: list[dict],
    split: dict,
    seed: int,
) -> list[dict]:
    """Build balanced, source-disjoint four-item evidence bundles."""
    split_name, split_hash, _ = _split_contract(variants, split)
    by_source = _variant_index(variants)
    eligible = []
    valid_semantics: dict[str, list[dict]] = {}
    for source_id, choices in by_source.items():
        clean = choices.get("clean_quote")
        semantics = [
            choices[kind]
            for kind in SEMANTIC_TYPES
            if kind in choices
            and _valid_semantic(choices[kind])
        ]
        if clean and _valid_clean(clean) and semantics:
            eligible.append(source_id)
            valid_semantics[source_id] = semantics
    bundle_count = _BUNDLE_COUNTS[split_name]
    required_sources = bundle_count * 4
    if len(eligible) < required_sources:
        raise ValueError(
            f"{split_name} requires {required_sources} eligible sources; "
            f"found {len(eligible)}"
        )
    selected, stratum_summary = _select_stratified_sources(
        eligible, by_source, required_sources, seed
    )
    rng = random.Random(seed)
    per_class = bundle_count // 3
    labels = [
        label
        for _ in range(per_class)
        for label in BUNDLE_LABELS
    ]
    slots, stratum_summary = _allocate_stratified_bundles(
        selected,
        by_source,
        labels,
        seed,
        stratum_summary,
        split_name,
    )
    bundles = []
    for slot in slots:
        label = slot["label"]
        source_ids = slot["source_ids"]
        corruption_count = {"advance": 0, "hold": 1, "reject": 2}[label]
        corrupt_positions = set(rng.sample(range(4), corruption_count))
        members = []
        for position, source_id in enumerate(source_ids):
            choices = by_source[source_id]
            if position in corrupt_positions:
                candidates = sorted(
                    valid_semantics[source_id],
                    key=lambda item: item["variant_type"],
                )
                chosen = rng.choice(candidates)
            else:
                chosen = choices["clean_quote"]
            members.append(_bundle_member(chosen))
        members.sort(key=lambda item: item["source_id"])
        bundle = {
            "bundle_id": slot["bundle_id"],
            "split": split_name,
            "split_manifest_sha256": split_hash,
            "gold": label,
            "injected_corruption_count": corruption_count,
            "stratum_summary": copy.deepcopy(stratum_summary),
            "members": members,
        }
        bundle["bundle_sha256"] = compute_bundle_hash(bundle)
        bundles.append(bundle)
    bundles.sort(key=lambda item: item["bundle_id"])
    return bundles


def bundle_metrics(
    predictions: list[dict],
    bundles: list[dict],
    split_manifest: dict | None = None,
) -> dict:
    """Compute exact multiclass endpoints with strict member-ID alignment."""
    _require_nonempty_records(predictions, "predictions")
    _require_nonempty_records(bundles, "bundles")
    manifest_items = [
        member
        for bundle in bundles
        if isinstance(bundle, dict)
        for member in bundle.get("members", [])
        if isinstance(member, dict)
    ]
    split_name, split_hash, _ = _split_contract(
        manifest_items, split_manifest
    )
    by_id = {}
    for item in predictions:
        if not isinstance(item, dict):
            raise ValueError("predictions contain a non-object")
        required = {"variant_id", "source_id", "prediction"}
        missing = required.difference(item)
        if missing:
            raise ValueError(f"prediction missing fields: {sorted(missing)}")
        variant_id = item["variant_id"]
        if variant_id in by_id:
            raise ValueError(f"duplicate variant_id: {variant_id}")
        if item["prediction"] not in BINARY_LABELS:
            raise ValueError("prediction has invalid binary label")
        by_id[variant_id] = item

    expected = {}
    bundle_ids = set()
    used_sources = set()
    gold_counts: Counter = Counter()
    for bundle in bundles:
        if not isinstance(bundle, dict):
            raise ValueError("bundles contain a non-object")
        required = {
            "bundle_id",
            "split",
            "split_manifest_sha256",
            "gold",
            "injected_corruption_count",
            "stratum_summary",
            "members",
            "bundle_sha256",
        }
        missing = required.difference(bundle)
        if missing:
            raise ValueError(f"bundle missing fields: {sorted(missing)}")
        if bundle["bundle_id"] in bundle_ids:
            raise ValueError(f"duplicate bundle_id: {bundle['bundle_id']}")
        bundle_ids.add(bundle["bundle_id"])
        if bundle["bundle_sha256"] != compute_bundle_hash(bundle):
            raise ValueError(
                f"bundle hash mismatch: {bundle['bundle_id']}"
            )
        if (
            bundle["split"] != split_name
            or bundle["split_manifest_sha256"] != split_hash
        ):
            raise ValueError("bundle split manifest alignment mismatch")
        if bundle["gold"] not in BUNDLE_LABELS:
            raise ValueError("bundle has invalid gold label")
        gold_counts[bundle["gold"]] += 1
        summary = bundle["stratum_summary"]
        if (
            not isinstance(summary, dict)
            or summary.get("method")
            != "deterministic_greedy_coverage_and_allocation"
        ):
            raise ValueError("bundle has invalid stratum summary")
        if not isinstance(bundle["members"], list) or len(
            bundle["members"]
        ) != 4:
            raise ValueError("each bundle must contain exactly four members")
        source_ids = [member.get("source_id") for member in bundle["members"]]
        if len(set(source_ids)) != 4:
            raise ValueError("bundle must contain four unique sources")
        reused_sources = set(source_ids).intersection(used_sources)
        if reused_sources:
            raise ValueError("source reused across bundles in split")
        used_sources.update(source_ids)
        contaminated_count = 0
        for member in bundle["members"]:
            variant_id = member.get("variant_id")
            if variant_id in expected:
                raise ValueError(
                    f"bundle member variant reused: {variant_id}"
                )
            if member.get("variant_type") == "clean_quote":
                if not _valid_clean(member):
                    raise ValueError("invalid clean member")
            else:
                if not _valid_semantic(member):
                    raise ValueError("invalid semantic corruption member")
                contaminated_count += 1
            expected[variant_id] = member
        if bundle["injected_corruption_count"] != contaminated_count:
            raise ValueError("injected corruption count mismatch")
        expected_gold = integrity_decision(contaminated_count)
        if bundle["gold"] != expected_gold:
            raise ValueError(
                "bundle class does not match injected corruption count"
            )
    expected_per_class = len(bundles) // 3
    if (
        len(bundles) != _BUNDLE_COUNTS[split_name]
        or any(
            gold_counts[label] != expected_per_class
            for label in BUNDLE_LABELS
        )
    ):
        raise ValueError("bundle classes are not exactly balanced")
    if set(by_id) != set(expected):
        raise ValueError("prediction and bundle ID alignment mismatch")
    for variant_id, member in expected.items():
        if by_id[variant_id]["source_id"] != member["source_id"]:
            raise ValueError(
                f"prediction source alignment mismatch for {variant_id}"
            )

    transition = {
        gold: {prediction: 0 for prediction in BUNDLE_LABELS}
        for gold in BUNDLE_LABELS
    }
    decisions = []
    for bundle in sorted(bundles, key=lambda item: item["bundle_id"]):
        unsupported = sum(
            by_id[member["variant_id"]]["prediction"] == "unsupported"
            for member in bundle["members"]
        )
        decision = integrity_decision(unsupported)
        gold = bundle["gold"]
        transition[gold][decision] += 1
        decisions.append((gold, decision))
    if any(sum(transition[label].values()) == 0 for label in BUNDLE_LABELS):
        raise ValueError("bundles are incomplete: all three classes required")

    f1_values = []
    for label in BUNDLE_LABELS:
        tp = transition[label][label]
        fp = sum(
            transition[other][label]
            for other in BUNDLE_LABELS
            if other != label
        )
        fn = sum(
            transition[label][other]
            for other in BUNDLE_LABELS
            if other != label
        )
        denominator = 2 * tp + fp + fn
        f1_values.append(2 * tp / denominator if denominator else 0.0)
    contaminated = [
        decision for gold, decision in decisions if gold != "advance"
    ]
    reject = [
        decision for gold, decision in decisions if gold == "reject"
    ]
    positions = {label: index for index, label in enumerate(BUNDLE_LABELS)}
    return {
        "bundle_count": len(bundles),
        "macro_f1": _rounded(sum(f1_values) / len(f1_values)),
        "transition_matrix": transition,
        "erroneous_advance_rate": _rounded(
            sum(value == "advance" for value in contaminated)
            / len(contaminated)
        ),
        "reject_recall": _rounded(
            sum(value == "reject" for value in reject) / len(reject)
        ),
        "exact_match_accuracy": _rounded(
            sum(gold == prediction for gold, prediction in decisions)
            / len(decisions)
        ),
        "weighted_ordinal_distance": _rounded(
            sum(
                abs(positions[gold] - positions[prediction])
                for gold, prediction in decisions
            )
            / len(decisions)
        ),
    }


def _bundle_decision_records(
    predictions: list[dict],
    bundles: list[dict],
    split_manifest: dict,
) -> list[dict]:
    bundle_metrics(predictions, bundles, split_manifest)
    by_id = {item["variant_id"]: item for item in predictions}
    return [
        {
            "bundle_id": bundle["bundle_id"],
            "gold": bundle["gold"],
            "prediction": integrity_decision(sum(
                by_id[member["variant_id"]]["prediction"] == "unsupported"
                for member in bundle["members"]
            )),
        }
        for bundle in sorted(bundles, key=lambda item: item["bundle_id"])
    ]


def _bundle_macro_f1(records: list[dict]) -> float:
    scores = []
    for label in BUNDLE_LABELS:
        tp = sum(
            item["gold"] == label and item["prediction"] == label
            for item in records
        )
        fp = sum(
            item["gold"] != label and item["prediction"] == label
            for item in records
        )
        fn = sum(
            item["gold"] == label and item["prediction"] != label
            for item in records
        )
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def _aligned_bundle_records(
    a: list[dict],
    b: list[dict],
    bundles: list[dict],
    split_manifest: dict,
) -> tuple[list[dict], list[dict]]:
    first = _bundle_decision_records(a, bundles, split_manifest)
    second = _bundle_decision_records(b, bundles, split_manifest)
    if [
        (item["bundle_id"], item["gold"]) for item in first
    ] != [
        (item["bundle_id"], item["gold"]) for item in second
    ]:
        raise ValueError("bundle condition alignment mismatch")
    return first, second


def bundle_macro_f1_difference(
    a: list[dict],
    b: list[dict],
    bundles: list[dict],
    split_manifest: dict,
) -> float:
    first, second = _aligned_bundle_records(
        a, b, bundles, split_manifest
    )
    return _rounded(_bundle_macro_f1(first) - _bundle_macro_f1(second))


def bundle_bootstrap_difference(
    a: list[dict],
    b: list[dict],
    bundles: list[dict],
    split_manifest: dict,
    *,
    seed: int,
    draws: int = 10_000,
) -> dict:
    if not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if draws > _MAX_BOOTSTRAP_DRAWS:
        raise ValueError("bootstrap draws exceed bounded maximum")
    first, second = _aligned_bundle_records(
        a, b, bundles, split_manifest
    )
    rng = random.Random(seed)
    differences = []
    for _ in range(draws):
        indices = [rng.randrange(len(first)) for _ in first]
        differences.append(
            _bundle_macro_f1([first[index] for index in indices])
            - _bundle_macro_f1([second[index] for index in indices])
        )
    differences.sort()
    return {
        "observed_difference": bundle_macro_f1_difference(
            a, b, bundles, split_manifest
        ),
        "cluster_count": len(first),
        "draws": draws,
        "seed": seed,
        "ci_low": _rounded(_percentile(differences, 0.025)),
        "ci_high": _rounded(_percentile(differences, 0.975)),
    }


def bundle_permutation_test(
    a: list[dict],
    b: list[dict],
    bundles: list[dict],
    split_manifest: dict,
    *,
    seed: int,
    draws: int = 100_000,
) -> dict:
    if not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    if draws > _MAX_PERMUTATION_DRAWS:
        raise ValueError("permutation draws exceed bounded maximum")
    first, second = _aligned_bundle_records(
        a, b, bundles, split_manifest
    )
    cluster_count = len(first)
    if cluster_count > 20 and draws != _MAX_PERMUTATION_DRAWS:
        raise ValueError(
            "more than 20 bundles require exactly 100000 permutation draws"
        )
    observed = _bundle_macro_f1(first) - _bundle_macro_f1(second)
    if cluster_count <= 20:
        masks = range(2 ** cluster_count)
        actual_draws = 2 ** cluster_count
        method = "exact_enumeration"
    else:
        rng = random.Random(seed)
        masks = [
            rng.getrandbits(cluster_count)
            for _ in range(draws)
        ]
        actual_draws = draws
        method = "seeded_monte_carlo"
    extreme = 0
    for mask in masks:
        permuted_first = []
        permuted_second = []
        for index, (left, right) in enumerate(zip(first, second)):
            if mask & (1 << index):
                permuted_first.append({**left, "prediction": right["prediction"]})
                permuted_second.append({**right, "prediction": left["prediction"]})
            else:
                permuted_first.append(left)
                permuted_second.append(right)
        difference = (
            _bundle_macro_f1(permuted_first)
            - _bundle_macro_f1(permuted_second)
        )
        if abs(difference) + 1e-12 >= abs(observed):
            extreme += 1
    p_value = (
        extreme / actual_draws
        if method == "exact_enumeration"
        else (extreme + 1) / (actual_draws + 1)
    )
    return {
        "observed_difference": _rounded(observed),
        "cluster_count": cluster_count,
        "draws": actual_draws,
        "seed": seed,
        "method": method,
        "extreme_draws": extreme,
        "p_value": _rounded(p_value),
    }
