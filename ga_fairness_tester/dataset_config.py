"""Dataset configuration: sensitive feature column indices and value domains.

Column indices were verified by inspecting CSV headers from the ideas-labo/ISE
lab4 repository. Sensitive attributes follow the README of lab4.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
DNN_DIR = os.path.join(os.path.dirname(__file__), "..", "DNN")
SKLEARN_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@dataclass
class SensitiveFeature:
    name: str
    col_idx: int
    values: list  # discrete domain to flip across; for "continuous" use bin endpoints


@dataclass
class DatasetSpec:
    name: str
    csv: str
    dnn_path: str
    label_col: str
    sensitive: dict[str, SensitiveFeature] = field(default_factory=dict)


DATASET_CONFIG: dict[str, DatasetSpec] = {
    "adult": DatasetSpec(
        name="adult",
        csv=os.path.join(DATA_DIR, "processed_adult.csv"),
        dnn_path=os.path.join(DNN_DIR, "model_processed_adult.h5"),
        label_col="Class-label",
        sensitive={
            # Header: race,gender,relationship,age,hours-per-week,workclass,
            #         occupation,education,native-country,marital-status,Class-label
            "race":   SensitiveFeature("race",   col_idx=0, values=[0, 1, 2, 3, 4]),
            "gender": SensitiveFeature("gender", col_idx=1, values=[0, 1]),
            "age":    SensitiveFeature("age",    col_idx=3, values=list(range(17, 91))),
        },
    ),
    "compas": DatasetSpec(
        name="compas",
        csv=os.path.join(DATA_DIR, "processed_compas.csv"),
        dnn_path=os.path.join(DNN_DIR, "model_processed_compas.h5"),
        label_col="Recidivism",
        sensitive={
            # Header: Sex,Age,Race,Juvenile Felonies,Juvenile Misdemeanors,
            #         Juvenile Others,Previous Convictions,Degree of Conviction,
            #         Days in Jail,COMPAS Score,Recidivism
            "sex":  SensitiveFeature("sex",  col_idx=0, values=[0, 1]),
            "race": SensitiveFeature("race", col_idx=2, values=[0, 1, 2, 3, 4, 5]),
        },
    ),
}


def list_datasets() -> list[str]:
    return list(DATASET_CONFIG.keys())


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASET_CONFIG:
        raise KeyError(f"Unknown dataset {name!r}. Known: {list_datasets()}")
    return DATASET_CONFIG[name]
