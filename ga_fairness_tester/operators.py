"""Selection, crossover and mutation operators for GA-FT."""

from __future__ import annotations

import numpy as np

from .data_loader import FeatureMeta


def tournament_select(
    population: np.ndarray, fitness: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """One tournament-of-k: returns one winner."""
    idx = rng.integers(0, len(population), size=k)
    winner = idx[np.argmax(fitness[idx])]
    return population[winner].copy()


def uniform_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    sensitive_idx: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-feature random pick from each parent, but never swap the sensitive feature
    (its value is preserved from parent_a so the child stays a valid 'seed'; the
    oracle will explore all sensitive flips at evaluation time)."""
    mask = rng.random(len(parent_a)) < 0.5
    child = np.where(mask, parent_a, parent_b).astype("float32")
    child[sensitive_idx] = parent_a[sensitive_idx]
    return child


def mutate(
    individual: np.ndarray,
    feature_meta: list[FeatureMeta],
    sensitive_idx: int,
    mutation_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-feature mutation: random valid integer in [low, high]. Sensitive feature
    is left untouched — the oracle iterates the sensitive domain itself."""
    out = individual.copy()
    for i, meta in enumerate(feature_meta):
        if i == sensitive_idx:
            continue
        if rng.random() < mutation_rate:
            out[i] = rng.integers(meta.low, meta.high + 1)
    return out
