"""Unit tests for the oracle: pair generation, IDI detection, dedup keying."""

from __future__ import annotations

import numpy as np
import pytest

from ga_fairness_tester.data_loader import FeatureMeta
from ga_fairness_tester.oracle import (
    canonical_idi_key,
    evaluate_individual,
    is_idi,
    make_pair,
)


class _StubAdapter:
    """Adapter whose probability is a programmable function of the sensitive value."""

    def __init__(self, prob_by_sensitive: dict[int, float], sensitive_idx: int):
        self.prob_by_sensitive = prob_by_sensitive
        self.sensitive_idx = sensitive_idx

    def predict_proba_pos(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.array([self.prob_by_sensitive[int(row[self.sensitive_idx])] for row in X])

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba_pos(X) >= 0.5).astype(int)


def _binary_meta(sensitive_idx: int = 0) -> FeatureMeta:
    return FeatureMeta(
        name="sens",
        col_idx=sensitive_idx,
        low=0,
        high=1,
        domain=np.array([0, 1]),
    )


def test_make_pair_returns_one_row_per_sensitive_value():
    x = np.array([0, 5, 10], dtype="float32")
    meta = FeatureMeta(name="s", col_idx=0, low=0, high=2, domain=np.array([0, 1, 2]))
    variants = make_pair(x, sensitive_idx=0, meta=meta)
    assert variants.shape == (3, 3)
    assert list(variants[:, 0]) == [0, 1, 2]
    # non-sensitive columns are unchanged
    assert (variants[:, 1] == 5).all()
    assert (variants[:, 2] == 10).all()


def test_make_pair_handles_six_valued_sensitive_attribute():
    """COMPAS Race has six categories — test the multi-valued generalisation."""
    x = np.array([0, 7, 7], dtype="float32")
    meta = FeatureMeta(name="race", col_idx=0, low=0, high=5,
                       domain=np.array([0, 1, 2, 3, 4, 5]))
    variants = make_pair(x, sensitive_idx=0, meta=meta)
    assert variants.shape == (6, 3)


def test_is_idi_fires_when_predictions_disagree():
    adapter = _StubAdapter({0: 0.2, 1: 0.8}, sensitive_idx=0)  # flips class
    variants = make_pair(np.array([0, 1], dtype="float32"), 0, _binary_meta())
    assert is_idi(variants, adapter) is True


def test_is_idi_does_not_fire_when_predictions_agree():
    adapter = _StubAdapter({0: 0.2, 1: 0.3}, sensitive_idx=0)  # both negative
    variants = make_pair(np.array([0, 1], dtype="float32"), 0, _binary_meta())
    assert is_idi(variants, adapter) is False


def test_is_idi_does_not_fire_when_both_positive():
    adapter = _StubAdapter({0: 0.7, 1: 0.9}, sensitive_idx=0)  # both positive
    variants = make_pair(np.array([0, 1], dtype="float32"), 0, _binary_meta())
    assert is_idi(variants, adapter) is False


def test_evaluate_individual_returns_correct_conf_delta():
    adapter = _StubAdapter({0: 0.2, 1: 0.85}, sensitive_idx=0)
    is_disc, conf_delta, _, _ = evaluate_individual(
        np.array([0, 1], dtype="float32"), 0, _binary_meta(), adapter
    )
    assert is_disc is True
    assert conf_delta == pytest.approx(0.65, abs=1e-6)


def test_canonical_idi_key_ignores_sensitive_value():
    x_a = np.array([0, 5, 10])
    x_b = np.array([1, 5, 10])  # differs only in sensitive
    assert canonical_idi_key(x_a, sensitive_idx=0) == canonical_idi_key(x_b, sensitive_idx=0)


def test_canonical_idi_key_distinguishes_genuinely_different_inputs():
    x_a = np.array([0, 5, 10])
    x_b = np.array([0, 6, 10])  # differs in non-sensitive
    assert canonical_idi_key(x_a, sensitive_idx=0) != canonical_idi_key(x_b, sensitive_idx=0)
