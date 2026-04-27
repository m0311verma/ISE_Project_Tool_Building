"""Load CSV datasets and derive per-feature metadata (min/max, integer domain)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dataset_config import DatasetSpec, get_dataset


@dataclass
class FeatureMeta:
    name: str
    col_idx: int
    low: int
    high: int  # inclusive
    domain: np.ndarray  # all observed integer values for this column


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    X: np.ndarray            # shape (n_rows, n_features), float32
    y: np.ndarray            # shape (n_rows,), int
    feature_names: list[str]
    feature_meta: list[FeatureMeta]
    n_features: int

    def sensitive_idx(self, sensitive_name: str) -> int:
        return self.spec.sensitive[sensitive_name].col_idx


def load_dataset(name: str) -> LoadedDataset:
    spec = get_dataset(name)
    df = pd.read_csv(spec.csv)
    label_col = spec.label_col
    feature_names = [c for c in df.columns if c != label_col]
    X = df[feature_names].values.astype("float32")
    y = df[label_col].values.astype("int64")

    meta = []
    for i, col in enumerate(feature_names):
        vals = df[col].values
        meta.append(
            FeatureMeta(
                name=col,
                col_idx=i,
                low=int(vals.min()),
                high=int(vals.max()),
                domain=np.array(sorted(np.unique(vals).astype(int))),
            )
        )

    return LoadedDataset(
        spec=spec,
        X=X,
        y=y,
        feature_names=feature_names,
        feature_meta=meta,
        n_features=len(feature_names),
    )
