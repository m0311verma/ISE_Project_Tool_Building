"""Unified black-box model interface across DNN (Keras) and sklearn classifiers.

GA-FT and Random Search query models only via predict_proba_pos() and
predict_class(), making the algorithm fully model-agnostic.
"""

from __future__ import annotations

import os
import warnings

import joblib
import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from .dataset_config import SKLEARN_DIR, get_dataset


class ModelAdapter:
    """Wraps DNN/LR/RF/GBM behind a single predict_proba_pos / predict_class API."""

    def __init__(self, model, model_type: str, dataset_name: str):
        self.model = model
        self.model_type = model_type  # "dnn" | "lr" | "rf" | "gbm"
        self.dataset_name = dataset_name

    @property
    def name(self) -> str:
        return f"{self.dataset_name}/{self.model_type}"

    def predict_proba_pos(self, X: np.ndarray) -> np.ndarray:
        """Probability of positive class (shape: n,)."""
        X = np.atleast_2d(X).astype("float32")
        if self.model_type == "dnn":
            return self.model.predict(X, verbose=0).flatten()
        return self.model.predict_proba(X)[:, 1]

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Hard 0/1 prediction (shape: n,)."""
        X = np.atleast_2d(X).astype("float32")
        if self.model_type == "dnn":
            return (self.model.predict(X, verbose=0).flatten() >= 0.5).astype(int)
        return self.model.predict(X).astype(int)


def load_adapter(dataset_name: str, model_type: str) -> ModelAdapter:
    """Load the requested model from disk and wrap in ModelAdapter."""
    spec = get_dataset(dataset_name)
    if model_type == "dnn":
        from tensorflow.keras.models import load_model
        model = load_model(spec.dnn_path, compile=False)
    elif model_type in ("lr", "rf", "gbm"):
        path = os.path.join(SKLEARN_DIR, f"{dataset_name}_{model_type}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Sklearn model {path!r} missing. Run train_sklearn_models.py first."
            )
        model = joblib.load(path)
    else:
        raise ValueError(f"Unknown model_type {model_type!r}")
    return ModelAdapter(model, model_type, dataset_name)
