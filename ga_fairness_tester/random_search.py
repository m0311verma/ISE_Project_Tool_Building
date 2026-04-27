"""Random Search baseline.

Samples random rows from the dataset, evaluates whether each is an IDI under
all flips of the sensitive attribute, and tracks unique IDIs found within the
budget. No fitness signal — every sample is independent and uniformly random.
"""

from __future__ import annotations

import time

import numpy as np

from .data_loader import LoadedDataset
from .model_adapter import ModelAdapter
from .oracle import canonical_idi_key, evaluate_individual


def random_search(
    data: LoadedDataset,
    adapter: ModelAdapter,
    sensitive_name: str,
    budget: int = 1000,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    sensitive_idx = data.sensitive_idx(sensitive_name)
    meta = data.feature_meta[sensitive_idx]

    found_keys: set[tuple] = set()
    found_xs: list[np.ndarray] = []
    convergence: list[int] = []

    t0 = time.time()
    for step in range(budget):
        idx = rng.integers(0, data.X.shape[0])
        x = data.X[idx]
        is_disc, _, _, _ = evaluate_individual(x, sensitive_idx, meta, adapter)
        if is_disc:
            key = canonical_idi_key(x, sensitive_idx)
            if key not in found_keys:
                found_keys.add(key)
                found_xs.append(x.copy())
        convergence.append(len(found_keys))
    runtime = time.time() - t0

    return {
        "algorithm": "RS",
        "n_idis": len(found_keys),
        "idi_ratio": len(found_keys) / budget,
        "idi_diversity": _avg_pairwise_dist(found_xs, data.feature_meta),
        "convergence": convergence,
        "runtime_seconds": runtime,
    }


def _avg_pairwise_dist(xs: list[np.ndarray], feature_meta) -> float:
    """Average pairwise normalised Euclidean distance among found IDI seeds."""
    if len(xs) < 2:
        return 0.0
    arr = np.asarray(xs, dtype="float32")
    ranges = np.array(
        [max(m.high - m.low, 1) for m in feature_meta], dtype="float32"
    )
    arr_norm = arr / ranges
    n = arr_norm.shape[0]
    total = 0.0
    count = 0
    for i in range(n):
        diffs = arr_norm[i + 1 :] - arr_norm[i]
        total += np.linalg.norm(diffs, axis=1).sum()
        count += n - i - 1
    return float(total / count) if count else 0.0
