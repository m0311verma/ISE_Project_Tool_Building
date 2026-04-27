"""IDI detection and pair generation.

An Individual Discriminatory Instance (IDI) is a pair (xa, xb) that differs
ONLY in the sensitive attribute and produces different model predictions.
"""

from __future__ import annotations

import numpy as np

from .data_loader import FeatureMeta
from .model_adapter import ModelAdapter


def make_pair(x: np.ndarray, sensitive_idx: int, meta: FeatureMeta) -> np.ndarray:
    """Return an array shape (k+1, n_features) of variants of x, one per sensitive value.

    The first row equals x. Subsequent rows are x with sensitive feature replaced by
    each other value in the feature's domain. We test every distinct flip so the
    oracle can detect any discrimination, not just one direction.
    """
    x = np.asarray(x, dtype="float32").reshape(-1)
    domain = meta.domain
    variants = np.tile(x, (len(domain), 1))
    variants[:, sensitive_idx] = domain
    return variants


def is_idi(variants: np.ndarray, adapter: ModelAdapter) -> bool:
    """True if the model produces at least two distinct class predictions across variants."""
    preds = adapter.predict_class(variants)
    return bool(np.unique(preds).size > 1)


def evaluate_individual(
    x: np.ndarray, sensitive_idx: int, meta: FeatureMeta, adapter: ModelAdapter
) -> tuple[bool, float, np.ndarray, np.ndarray]:
    """Return (is_discriminatory, max_conf_delta, variants, probs).

    max_conf_delta = max over all pairs (i, j) of |prob_i - prob_j|
    where probs are positive-class probabilities for each variant.
    This is the multi-valued generalisation of |prob(xa) - prob(xb)|.
    """
    variants = make_pair(x, sensitive_idx, meta)
    probs = adapter.predict_proba_pos(variants)
    preds = (probs >= 0.5).astype(int)
    discriminatory = bool(np.unique(preds).size > 1)
    conf_delta = float(probs.max() - probs.min())
    return discriminatory, conf_delta, variants, probs


def canonical_idi_key(x: np.ndarray, sensitive_idx: int) -> tuple:
    """Hashable key that ignores the sensitive feature, so two pairs that differ
    only in which sensitive value is xa vs xb count as the same IDI."""
    arr = np.asarray(x, dtype="int64").copy()
    arr[sensitive_idx] = -1
    return tuple(arr.tolist())
