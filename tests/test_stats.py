"""Unit tests for Vargha-Delaney A12 and magnitude labelling."""

from __future__ import annotations

import numpy as np
import pytest

from ga_fairness_tester.stats import magnitude_label, vargha_delaney_a12


def test_a12_equals_one_when_x_strictly_dominates_y():
    x = np.array([10, 11, 12])
    y = np.array([1, 2, 3])
    assert vargha_delaney_a12(x, y) == 1.0


def test_a12_equals_zero_when_y_strictly_dominates_x():
    x = np.array([1, 2, 3])
    y = np.array([10, 11, 12])
    assert vargha_delaney_a12(x, y) == 0.0


def test_a12_equals_half_when_samples_are_identical():
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 2, 3, 4])
    assert vargha_delaney_a12(x, y) == pytest.approx(0.5, abs=1e-6)


def test_a12_handles_ties_with_half_weight():
    """Two-vs-two with one tie: x=[1,3], y=[1,2]. Pairs: (1,1)=tie, (1,2)<, (3,1)>, (3,2)>.
       A12 = (greater + 0.5*equal) / (nx*ny) = (2 + 0.5*1) / 4 = 0.625."""
    x = np.array([1, 3])
    y = np.array([1, 2])
    assert vargha_delaney_a12(x, y) == pytest.approx(0.625, abs=1e-6)


def test_magnitude_label_thresholds():
    # |A12 - 0.5| < 0.06 -> negligible
    assert magnitude_label(0.50)  == "negligible"
    assert magnitude_label(0.55)  == "negligible"
    # 0.06 <= delta < 0.14 -> small
    assert magnitude_label(0.57)  == "small"
    assert magnitude_label(0.63)  == "small"
    # 0.14 <= delta < 0.21 -> medium
    assert magnitude_label(0.65)  == "medium"
    assert magnitude_label(0.70)  == "medium"
    # delta >= 0.21 -> large
    assert magnitude_label(0.72)  == "large"
    assert magnitude_label(1.00)  == "large"
    # symmetric below 0.5 (RS dominates)
    assert magnitude_label(0.30)  == "medium"
    assert magnitude_label(0.00)  == "large"
