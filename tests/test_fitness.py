"""Unit tests for the dual fitness function and its components."""

from __future__ import annotations

import numpy as np
import pytest

from ga_fairness_tester.data_loader import FeatureMeta
from ga_fairness_tester.fitness import (
    evaluate_fitness,
    min_distance_to_found,
    normalise,
)


class _StubAdapter:
    def __init__(self, prob_by_sensitive: dict[int, float], sensitive_idx: int):
        self.prob_by_sensitive = prob_by_sensitive
        self.sensitive_idx = sensitive_idx

    def predict_proba_pos(self, X):
        X = np.atleast_2d(X)
        return np.array([self.prob_by_sensitive[int(row[self.sensitive_idx])] for row in X])

    def predict_class(self, X):
        return (self.predict_proba_pos(X) >= 0.5).astype(int)


def _two_feature_meta() -> list[FeatureMeta]:
    return [
        FeatureMeta(name="sens", col_idx=0, low=0,  high=1,  domain=np.array([0, 1])),
        FeatureMeta(name="x1",   col_idx=1, low=0,  high=10, domain=np.arange(11)),
    ]


def test_normalise_maps_low_to_zero_and_high_to_one():
    meta = _two_feature_meta()
    x = np.array([0, 0], dtype="float32")
    np.testing.assert_allclose(normalise(x, meta), [0.0, 0.0])
    x = np.array([1, 10], dtype="float32")
    np.testing.assert_allclose(normalise(x, meta), [1.0, 1.0])


def test_min_distance_to_found_returns_one_when_no_idis_yet():
    x_norm = np.array([0.5, 0.5])
    assert min_distance_to_found(x_norm, None, n_features=2) == 1.0


def test_min_distance_to_found_picks_nearest():
    x_norm = np.array([0.5, 0.5])
    found = np.array([[0.0, 0.0], [0.6, 0.6]])  # second row is closer
    d = min_distance_to_found(x_norm, found, n_features=2)
    expected = np.linalg.norm([0.5 - 0.6, 0.5 - 0.6]) / np.sqrt(2)
    assert d == pytest.approx(expected, abs=1e-6)


def test_evaluate_fitness_combines_terms_with_alpha_beta():
    """fitness = a*conf_delta + b*diversity. With known stubs, check both arithmetic and direction."""
    meta = _two_feature_meta()
    adapter = _StubAdapter({0: 0.2, 1: 0.9}, sensitive_idx=0)  # conf_delta = 0.7
    x = np.array([0, 5], dtype="float32")
    score, is_disc, conf = evaluate_fitness(
        x, sensitive_idx=0, meta=meta[0], adapter=adapter, feature_meta=meta,
        found_norm=None, alpha=0.7, beta=0.3,
    )
    # No IDIs found yet -> diversity = 1.0; expected = 0.7*0.7 + 0.3*1.0 = 0.79
    assert is_disc is True
    assert conf == pytest.approx(0.7, abs=1e-6)
    assert score == pytest.approx(0.7 * 0.7 + 0.3 * 1.0, abs=1e-6)


def test_evaluate_fitness_diversity_drops_when_x_is_close_to_existing_idi():
    meta = _two_feature_meta()
    adapter = _StubAdapter({0: 0.2, 1: 0.9}, sensitive_idx=0)
    x = np.array([0, 5], dtype="float32")  # normalises to [0, 0.5]
    found = np.array([[0.0, 0.5]], dtype="float32")  # exact match in normalised space
    score, _, _ = evaluate_fitness(
        x, 0, meta[0], adapter, meta, found_norm=found, alpha=0.7, beta=0.3,
    )
    # diversity = 0; fitness = 0.7*0.7 + 0.3*0 = 0.49
    assert score == pytest.approx(0.49, abs=1e-6)
