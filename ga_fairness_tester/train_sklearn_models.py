"""Train Logistic Regression, Random Forest and Gradient Boosting per dataset.

Models are saved with joblib to ../models/ for reuse by ModelAdapter.
"""

from __future__ import annotations

import os

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .data_loader import load_dataset
from .dataset_config import SKLEARN_DIR, list_datasets


SKLEARN_FAMILIES = {
    "lr":  lambda: Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=2000, random_state=0))]),
    "rf":  lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
    "gbm": lambda: GradientBoostingClassifier(n_estimators=200, random_state=0),
}


def train_one(dataset_name: str, family: str) -> tuple[float, str]:
    data = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=0.2, random_state=0, stratify=data.y
    )
    model = SKLEARN_FAMILIES[family]()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    os.makedirs(SKLEARN_DIR, exist_ok=True)
    path = os.path.join(SKLEARN_DIR, f"{dataset_name}_{family}.pkl")
    joblib.dump(model, path)
    return acc, path


def main(datasets: list[str] | None = None, families: list[str] | None = None) -> None:
    datasets = datasets or list_datasets()
    families = families or list(SKLEARN_FAMILIES.keys())
    for ds in datasets:
        for fam in families:
            acc, path = train_one(ds, fam)
            print(f"  {ds:>8} | {fam:>3} | test_acc={acc:.4f} | saved -> {os.path.basename(path)}")


if __name__ == "__main__":
    main()
