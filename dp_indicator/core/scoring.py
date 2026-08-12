"""Scoring weights and validation utilities.

Single source of truth for G1-G4 weights.  The ``compute_overall`` function
guarantees that ``overall == Σ(weight_i × score_i)`` — the value is never
taken from LLM output directly.
"""
from __future__ import annotations

DEFAULT_WEIGHTS: dict[str, float] = {
    "G1": 0.30,  # rationality
    "G2": 0.35,  # evidence landscape
    "G3": 0.20,  # falsifiability
    "G4": 0.15,  # feasibility
}


def compute_overall(scores: dict, weights: dict[str, float] = None) -> float:
    """Compute weighted overall score from individual dimension scores.

    If the caller passes a pre-existing ``overall``, it is compared to the
    computed value; a mismatch > 0.01 is recorded in ``_overall_corrected``
    so downstream code can detect it.
    """
    w = weights or DEFAULT_WEIGHTS
    computed = round(sum(scores.get(k, 0) * w[k] for k in w), 3)
    if "overall" in scores and abs(scores["overall"] - computed) > 0.01:
        scores["_overall_original"] = scores["overall"]
        scores["_overall_corrected"] = True
    return computed
