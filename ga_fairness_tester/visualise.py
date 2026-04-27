"""Generate box plots and convergence curves for the report."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_loader import load_dataset
from .ga_ft import ga_ft
from .model_adapter import load_adapter
from .random_search import random_search


FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def boxplot_by_model(results_csv: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(results_csv)
    for ds in df["dataset"].unique():
        for sf in df[df.dataset == ds]["sensitive_feature"].unique():
            sub = df[(df.dataset == ds) & (df.sensitive_feature == sf)]
            models = sorted(sub["model"].unique())
            fig, ax = plt.subplots(figsize=(7, 4))
            positions = []
            data_arrays = []
            labels = []
            for i, mt in enumerate(models):
                rs = sub[(sub.model == mt) & (sub.algorithm == "RS")]["idi_ratio"].values
                ga = sub[(sub.model == mt) & (sub.algorithm == "GA-FT")]["idi_ratio"].values
                positions.extend([i * 3 + 1, i * 3 + 2])
                data_arrays.extend([rs, ga])
                labels.extend([f"{mt}\nRS", f"{mt}\nGA-FT"])
            bp = ax.boxplot(data_arrays, positions=positions, widths=0.7,
                            patch_artist=True)
            for patch, lbl in zip(bp["boxes"], labels):
                patch.set_facecolor("#a6cee3" if "RS" in lbl else "#fb9a99")
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("IDI Ratio")
            ax.set_title(f"{ds.upper()} — sensitive: {sf}")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            path = os.path.join(out_dir, f"box_{ds}_{sf}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"  -> {path}")


def _aggregate_trajectories(
    runner, data, adapter, sensitive: str, budget: int, n_seeds: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run `runner` n_seeds times and return (median, p25, p75) across seeds."""
    trajectories = np.zeros((n_seeds, budget), dtype="float32")
    for seed in range(n_seeds):
        result = runner(data, adapter, sensitive, budget=budget, seed=seed)
        traj = np.asarray(result["convergence"], dtype="float32")
        # Pad/truncate to exactly `budget` points so all rows align.
        if traj.shape[0] < budget:
            pad = np.full(budget - traj.shape[0], traj[-1] if traj.size else 0,
                          dtype="float32")
            traj = np.concatenate([traj, pad])
        trajectories[seed] = traj[:budget]
    return (
        np.median(trajectories, axis=0),
        np.percentile(trajectories, 25, axis=0),
        np.percentile(trajectories, 75, axis=0),
    )


def convergence_curves(out_dir: str, dataset: str, model: str,
                       sensitive: str, budget: int = 1000,
                       n_seeds: int = 30) -> None:
    """Plot median + IQR convergence trajectories across n_seeds independent runs."""
    os.makedirs(out_dir, exist_ok=True)
    data = load_dataset(dataset)
    adapter = load_adapter(dataset, model)

    rs_med, rs_lo, rs_hi = _aggregate_trajectories(
        random_search, data, adapter, sensitive, budget, n_seeds
    )
    ga_med, ga_lo, ga_hi = _aggregate_trajectories(
        ga_ft, data, adapter, sensitive, budget, n_seeds
    )

    x = np.arange(budget)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(x, rs_lo, rs_hi, alpha=0.2, color="#1f78b4")
    ax.plot(x, rs_med, label=f"Random Search (median, n={n_seeds})",
            color="#1f78b4", linewidth=2)
    ax.fill_between(x, ga_lo, ga_hi, alpha=0.2, color="#e31a1c")
    ax.plot(x, ga_med, label=f"GA-FT (median, n={n_seeds})",
            color="#e31a1c", linewidth=2)
    ax.set_xlabel("Evaluations (budget consumed)")
    ax.set_ylabel("Unique IDIs found")
    ax.set_title(f"Convergence — {dataset.upper()} / {model.upper()} / {sensitive} "
                 f"(median ± IQR, {n_seeds} seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"conv_{dataset}_{model}_{sensitive}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> {path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/results.csv")
    p.add_argument("--out",     default=FIG_DIR)
    p.add_argument("--conv-dataset", default="adult")
    p.add_argument("--conv-model",   default="dnn")
    p.add_argument("--conv-sensitive", default="gender")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print("Box plots:")
    boxplot_by_model(args.results, args.out)
    print("Convergence curves:")
    for sf in ["gender", "race", "age"]:
        try:
            convergence_curves(args.out, args.conv_dataset, args.conv_model, sf)
        except Exception as e:
            print(f"  skipped {sf}: {e}")
    for sf in ["sex", "race"]:
        try:
            convergence_curves(args.out, "compas", "dnn", sf)
        except Exception as e:
            print(f"  skipped compas/{sf}: {e}")
