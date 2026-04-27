"""Wilcoxon rank-sum + Vargha-Delaney A12 effect size analysis."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import ranksums


def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    """Probability that a random sample from x is greater than a random sample from y.
       A12 = 0.5 -> equal. A12 > 0.5 -> x stochastically dominates y."""
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    greater = np.sum(x[:, None] > y[None, :])
    equal = np.sum(x[:, None] == y[None, :])
    return float((greater + 0.5 * equal) / (nx * ny))


def magnitude_label(a12: float) -> str:
    """Vargha-Delaney effect-size magnitude conventions (A>0.5 favours GA-FT)."""
    delta = abs(a12 - 0.5)
    if delta < 0.06:  return "negligible"
    if delta < 0.14:  return "small"
    if delta < 0.21:  return "medium"
    return "large"


def analyse(results_csv: str, out_csv: str) -> pd.DataFrame:
    df = pd.read_csv(results_csv)
    summary = []
    group_cols = ["dataset", "model", "sensitive_feature"]
    for keys, sub in df.groupby(group_cols):
        rs = sub[sub.algorithm == "RS"]["idi_ratio"].values
        ga = sub[sub.algorithm == "GA-FT"]["idi_ratio"].values
        if len(rs) == 0 or len(ga) == 0:
            continue
        stat, p = ranksums(ga, rs)
        a12 = vargha_delaney_a12(ga, rs)
        rs_div = sub[sub.algorithm == "RS"]["idi_diversity"].values
        ga_div = sub[sub.algorithm == "GA-FT"]["idi_diversity"].values
        verdict = (
            f"GA-FT {magnitude_label(a12)}-better"
            if (p <= 0.05 and a12 > 0.5)
            else f"RS {magnitude_label(a12)}-better"
            if (p <= 0.05 and a12 < 0.5)
            else "no significant difference"
        )
        summary.append({
            "dataset": keys[0], "model": keys[1], "sensitive_feature": keys[2],
            "rs_median":   float(np.median(rs)),
            "rs_iqr":      float(np.percentile(rs, 75) - np.percentile(rs, 25)),
            "ga_median":   float(np.median(ga)),
            "ga_iqr":      float(np.percentile(ga, 75) - np.percentile(ga, 25)),
            "wilcoxon_p":  float(p),
            "a12":         float(a12),
            "a12_label":   magnitude_label(a12),
            "verdict":     verdict,
            "rs_div_median": float(np.median(rs_div)) if len(rs_div) else float("nan"),
            "ga_div_median": float(np.median(ga_div)) if len(ga_div) else float("nan"),
        })
    out = pd.DataFrame(summary)
    out.to_csv(out_csv, index=False)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/results.csv")
    p.add_argument("--out",     default="results/stats_summary.csv")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    out = analyse(args.results, args.out)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(out.to_string(index=False))
    print(f"\nWritten to {args.out}")
