"""GA-FT: Genetic Algorithm Fairness Tester.

Population-based search guided by a dual fitness function (model-confidence
delta + diversity penalty) to find Individual Discriminatory Instances (IDIs)
faster than the Random Search baseline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .data_loader import LoadedDataset
from .fitness import (
    ALPHA_DEFAULT,
    BETA_DEFAULT,
    evaluate_fitness,
    normalise,
)
from .model_adapter import ModelAdapter
from .operators import mutate, tournament_select, uniform_crossover
from .oracle import canonical_idi_key
from .random_search import _avg_pairwise_dist


@dataclass
class GAParams:
    pop_size: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1   # per feature
    tournament_k: int = 5
    elitism_frac: float = 0.10
    alpha: float = ALPHA_DEFAULT
    beta: float = BETA_DEFAULT


def _init_population(
    data: LoadedDataset, pop_size: int, rng: np.random.Generator
) -> np.ndarray:
    idx = rng.integers(0, data.X.shape[0], size=pop_size)
    return data.X[idx].copy()


def ga_ft(
    data: LoadedDataset,
    adapter: ModelAdapter,
    sensitive_name: str,
    budget: int = 1000,
    seed: int = 0,
    params: GAParams | None = None,
) -> dict:
    params = params or GAParams()
    rng = np.random.default_rng(seed)
    sensitive_idx = data.sensitive_idx(sensitive_name)
    meta = data.feature_meta[sensitive_idx]
    n_features = data.n_features

    population = _init_population(data, params.pop_size, rng)
    fitness = np.zeros(params.pop_size, dtype="float32")

    found_keys: set[tuple] = set()
    found_xs: list[np.ndarray] = []
    found_norm: np.ndarray = np.zeros((0, n_features), dtype="float32")

    convergence: list[int] = []
    budget_used = 0
    t0 = time.time()

    def evaluate_and_record(x: np.ndarray) -> float:
        nonlocal budget_used, found_norm
        score, is_disc, _ = evaluate_fitness(
            x, sensitive_idx, meta, adapter, data.feature_meta,
            found_norm if found_norm.shape[0] else None,
            params.alpha, params.beta,
        )
        budget_used += 1
        if is_disc:
            key = canonical_idi_key(x, sensitive_idx)
            if key not in found_keys:
                found_keys.add(key)
                found_xs.append(x.copy())
                found_norm = np.vstack([found_norm, normalise(x, data.feature_meta)])
        convergence.append(len(found_keys))
        return score

    # Initial population evaluation
    for i in range(params.pop_size):
        if budget_used >= budget:
            break
        fitness[i] = evaluate_and_record(population[i])

    n_elite = max(1, int(params.elitism_frac * params.pop_size))

    while budget_used < budget:
        # Elitism: keep top n_elite individuals
        elite_idx = np.argsort(fitness)[-n_elite:]
        elites = population[elite_idx].copy()
        elite_fit = fitness[elite_idx].copy()

        # Generate offspring to fill the rest of the population
        new_pop = list(elites)
        new_fit = list(elite_fit)
        while len(new_pop) < params.pop_size and budget_used < budget:
            parent_a = tournament_select(population, fitness, params.tournament_k, rng)
            parent_b = tournament_select(population, fitness, params.tournament_k, rng)
            if rng.random() < params.crossover_rate:
                child = uniform_crossover(parent_a, parent_b, sensitive_idx, rng)
            else:
                child = parent_a.copy()
            child = mutate(child, data.feature_meta, sensitive_idx,
                           params.mutation_rate, rng)
            new_pop.append(child)
            new_fit.append(evaluate_and_record(child))

        population = np.asarray(new_pop[: params.pop_size], dtype="float32")
        fitness = np.asarray(new_fit[: params.pop_size], dtype="float32")

    runtime = time.time() - t0

    return {
        "algorithm": "GA-FT",
        "n_idis": len(found_keys),
        "idi_ratio": len(found_keys) / budget_used,
        "idi_diversity": _avg_pairwise_dist(found_xs, data.feature_meta),
        "convergence": convergence,
        "runtime_seconds": runtime,
        "budget_used": budget_used,
    }
