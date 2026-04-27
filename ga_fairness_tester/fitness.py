"""Dual fitness function for GA-FT.

    fitness(x) = alpha * delta_conf(x) + beta * diversity_bonus(x, found_IDIs)

  - delta_conf      : max |prob(M, variant_i) - prob(M, variant_j)| across
                      sensitive flips (exploitation: drives search toward inputs
                      where the model is most sensitive to the protected
                      attribute).
  - diversity_bonus : min normalised Euclidean distance from x to any
                      already-found IDI (exploration: rewards candidates that
                      are far from existing discoveries, so the GA does not
                      collapse onto a single IDI cluster).

Note: this exploration term is a per-candidate bonus used inside the GA's
fitness. It is distinct from the per-run *idi_diversity* metric reported in
the results table, which is the average pairwise distance among all unique
IDIs found in that run (computed in random_search._avg_pairwise_dist).
"""

from __future__ import annotations

import numpy as np

from .data_loader import FeatureMeta
from .oracle import evaluate_individual


ALPHA_DEFAULT = 0.7
BETA_DEFAULT = 0.3


def normalise(x: np.ndarray, feature_meta: list[FeatureMeta]) -> np.ndarray:
    ranges = np.array([max(m.high - m.low, 1) for m in feature_meta], dtype="float32")
    lows = np.array([m.low for m in feature_meta], dtype="float32")
    return (x - lows) / ranges


def min_distance_to_found(
    x: np.ndarray, found_norm: np.ndarray | None, n_features: int
) -> float:
    if found_norm is None or found_norm.shape[0] == 0:
        return 1.0  # no IDIs yet -> max diversity bonus
    diffs = found_norm - x
    dists = np.linalg.norm(diffs, axis=1) / np.sqrt(n_features)
    return float(dists.min())


def evaluate_fitness(
    x: np.ndarray,
    sensitive_idx: int,
    meta: FeatureMeta,
    adapter,
    feature_meta: list[FeatureMeta],
    found_norm: np.ndarray | None,
    alpha: float = ALPHA_DEFAULT,
    beta: float = BETA_DEFAULT,
) -> tuple[float, bool, float]:
    """Return (fitness_score, is_discriminatory, conf_delta)."""
    is_disc, conf_delta, _, _ = evaluate_individual(x, sensitive_idx, meta, adapter)
    x_norm = normalise(x, feature_meta)
    div = min_distance_to_found(x_norm, found_norm, len(feature_meta))
    score = alpha * conf_delta + beta * div
    return score, is_disc, conf_delta
