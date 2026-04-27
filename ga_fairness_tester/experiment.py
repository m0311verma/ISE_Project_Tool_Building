"""Run the full experimental matrix: 30 independent runs of RS and GA-FT
across (dataset x model x sensitive_feature) and write results to CSV.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time

from .data_loader import load_dataset
from .dataset_config import DATASET_CONFIG
from .ga_ft import ga_ft
from .model_adapter import load_adapter
from .random_search import random_search


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DEFAULT_DATASETS = ["adult", "compas"]
DEFAULT_MODELS = ["dnn", "lr", "rf", "gbm"]
DEFAULT_RUNS = 30
DEFAULT_BUDGET = 1000


def run_matrix(
    datasets: list[str],
    models: list[str],
    runs: int,
    budget: int,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    cache_data = {}
    cache_adapter = {}

    configs = []
    for ds in datasets:
        for mt in models:
            for sf in DATASET_CONFIG[ds].sensitive.keys():
                configs.append((ds, mt, sf))

    print(f"=> {len(configs)} configurations x {runs} seeds x 2 algorithms "
          f"= {len(configs) * runs * 2} runs")
    t_start = time.time()

    for ds, mt, sf in configs:
        if ds not in cache_data:
            cache_data[ds] = load_dataset(ds)
        data = cache_data[ds]

        key = (ds, mt)
        if key not in cache_adapter:
            cache_adapter[key] = load_adapter(ds, mt)
        adapter = cache_adapter[key]

        cfg_t0 = time.time()
        for seed in range(runs):
            rs = random_search(data, adapter, sf, budget=budget, seed=seed)
            ga = ga_ft(data, adapter, sf, budget=budget, seed=seed)
            for r, alg in [(rs, "RS"), (ga, "GA-FT")]:
                rows.append({
                    "algorithm": alg,
                    "dataset": ds,
                    "model": mt,
                    "sensitive_feature": sf,
                    "run_id": seed,
                    "budget": budget,
                    "n_idis": r["n_idis"],
                    "idi_ratio": r["idi_ratio"],
                    "idi_diversity": r["idi_diversity"],
                    "runtime_seconds": r["runtime_seconds"],
                })
        elapsed = time.time() - cfg_t0
        print(f"  done: {ds:>6} | {mt:>3} | {sf:>6} | {elapsed:6.1f}s "
              f"| total elapsed {(time.time() - t_start)/60:5.1f} min")
        # incremental save in case of interruption
        _write(out_path, rows)

    _write(out_path, rows)
    total = time.time() - t_start
    print(f"\nTotal: {len(rows)} rows | {total/60:.1f} min | -> {out_path}")


def _write(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "results.csv"))
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_matrix(args.datasets, args.models, args.runs, args.budget, args.out)
