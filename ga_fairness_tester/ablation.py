"""Alpha/Beta sensitivity sweep for the dual fitness function.

Sweeps the exploitation weight alpha across {0.0, 0.3, 0.5, 0.7, 0.9, 1.0}
with beta = 1 - alpha. Endpoints test the degenerate cases:

  - alpha = 0.0 -> pure diversity (no confidence signal); the GA only spreads
                   away from already-found IDIs.
  - alpha = 1.0 -> pure exploitation (no diversity); the GA chases the
                   confidence delta and may collapse onto one IDI cluster.

Runs on one representative configuration (adult / gbm / gender) where GA-FT
is known to dominate RS, so any drop in IDI ratio is attributable to the
fitness weighting rather than search-space saturation. Results are saved to
`results/ablation.csv` and a summary plot to `figures/ablation_alpha.png`.
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from .data_loader import load_dataset
from .ga_ft import GAParams, ga_ft
from .model_adapter import load_adapter


ALPHAS_DEFAULT = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def run_sweep(
    dataset: str,
    model: str,
    sensitive: str,
    alphas: tuple[float, ...],
    n_seeds: int,
    budget: int,
    out_csv: str,
) -> list[dict]:
    data = load_dataset(dataset)
    adapter = load_adapter(dataset, model)
    rows: list[dict] = []
    for alpha in alphas:
        beta = 1.0 - alpha
        for seed in range(n_seeds):
            params = GAParams(alpha=alpha, beta=beta)
            r = ga_ft(data, adapter, sensitive, budget=budget,
                      seed=seed, params=params)
            rows.append({
                "alpha": alpha, "beta": beta, "seed": seed,
                "idi_ratio": r["idi_ratio"], "idi_diversity": r["idi_diversity"],
            })
        sub = [row["idi_ratio"] for row in rows if row["alpha"] == alpha]
        print(f"  alpha={alpha:.2f}  beta={beta:.2f}  median IDI ratio = "
              f"{np.median(sub):.3f}  IQR = "
              f"{np.percentile(sub, 75) - np.percentile(sub, 25):.3f}")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten {len(rows)} rows -> {out_csv}")
    return rows


def plot_sweep(rows: list[dict], out_path: str,
               title_suffix: str = "") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    alphas = sorted({row["alpha"] for row in rows})
    medians, p25s, p75s = [], [], []
    for a in alphas:
        ratios = [r["idi_ratio"] for r in rows if r["alpha"] == a]
        medians.append(np.median(ratios))
        p25s.append(np.percentile(ratios, 25))
        p75s.append(np.percentile(ratios, 75))
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.fill_between(alphas, p25s, p75s, alpha=0.2, color="#e31a1c",
                    label="IQR (25-75 percentile)")
    ax.plot(alphas, medians, marker="o", color="#e31a1c", linewidth=2,
            label="median IDI ratio")
    ax.axvline(0.7, color="#555", linestyle="--", linewidth=1,
               label="default (alpha=0.7)")
    ax.set_xlabel("alpha (beta = 1 - alpha)")
    ax.set_ylabel("IDI ratio")
    ax.set_title(f"alpha/beta sensitivity{title_suffix}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  -> {out_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="adult")
    p.add_argument("--model", default="gbm")
    p.add_argument("--sensitive", default="gender")
    p.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS_DEFAULT))
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--out-csv", default=os.path.join(RESULTS_DIR, "ablation.csv"))
    p.add_argument("--out-fig", default=os.path.join(FIG_DIR, "ablation_alpha.png"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f"Sweep: alphas={args.alphas} seeds={args.seeds} budget={args.budget} "
          f"on {args.dataset}/{args.model}/{args.sensitive}")
    rows = run_sweep(
        args.dataset, args.model, args.sensitive,
        tuple(args.alphas), args.seeds, args.budget, args.out_csv,
    )
    plot_sweep(rows, args.out_fig,
               title_suffix=f" ({args.dataset.upper()} / {args.model.upper()} / "
                            f"{args.sensitive}, {args.seeds} seeds)")


if __name__ == "__main__":
    main()
