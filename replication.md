# Replication Guide

This document lists the exact commands to reproduce **every table and figure**
in the project report from a fresh clone of this repository.

## 0. Environment

```bash
git clone <this-repo-url>
cd <repo-dir>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13.7, TensorFlow 2.20.0, Keras 3.13.2 on macOS Darwin
25.4.0. Pre-trained DNNs and processed CSVs are committed under `DNN/` and
`dataset/`; no external download is required.

## 1. Train sklearn baselines

```bash
python -m ga_fairness_tester.train_sklearn_models
```

Produces `models/{adult,compas}_{lr,rf,gbm}.pkl`. Each model is trained on an
80/20 stratified split with `random_state=0`. Reported test accuracies (for
sanity checking, not used in the experiment):

```
adult |  lr | 0.754
adult |  rf | 0.812
adult | gbm | 0.840
compas|  lr | 0.706
compas|  rf | 0.682
compas| gbm | 0.709
```

## 2. Run the full experimental matrix

```bash
python -m ga_fairness_tester.experiment \
    --datasets adult compas \
    --models dnn lr rf gbm \
    --runs 30 \
    --budget 1000 \
    --out results/results.csv
```

Configurations: `2 datasets × 4 models × |sensitive_features| × 30 seeds × 2
algorithms`. Total runs: **1200** (adult has 3 sensitive features, compas
has 2, so 5 sensitive features in total).

Output: `results/results.csv` with one row per run, schema:

```
algorithm, dataset, model, sensitive_feature, run_id, budget,
n_idis, idi_ratio, idi_diversity, runtime_seconds
```

Wall-clock runtime: ~1.5 hours on a 2024 MacBook Pro (M-series CPU). DNN runs
dominate runtime; the three sklearn families together complete in under a
minute per configuration.

## 3. Statistical analysis

```bash
python -m ga_fairness_tester.stats \
    --results results/results.csv \
    --out     results/stats_summary.csv
```

Produces `results/stats_summary.csv` with per-configuration medians, IQRs,
Wilcoxon rank-sum p-values, Vargha-Delaney A12 effect sizes and verdict
labels. The summary table in the report is a direct subset of these columns.

## 4. Figures

```bash
python -m ga_fairness_tester.visualise \
    --results results/results.csv \
    --out     figures/
```

Produces:
- `figures/box_<dataset>_<sensitive>.png` — IDI ratio box plots, RS vs GA-FT,
  one panel per (dataset, sensitive feature) combination, with all four model
  families side by side.
- `figures/conv_<dataset>_<model>_<sensitive>.png` — convergence curves
  (unique IDIs found vs evaluations consumed) plotted as median ± IQR
  across 30 independent seeds per (dataset, model, sensitive) tuple.

All figures are 150 DPI matplotlib PNGs (no screenshots).

## 5. α/β sensitivity ablation (§5.5 of the report)

```bash
python -m ga_fairness_tester.ablation --seeds 10 --budget 1000
```

Sweeps α ∈ {0.0, 0.3, 0.5, 0.7, 0.9, 1.0} with β = 1 − α on the ADULT /
GBM / gender configuration. Writes `results/ablation.csv` and the
`figures/ablation_alpha.png` plot used in the report.

## 6. Unit tests

```bash
pytest tests/ -q
```

24 tests across `tests/test_oracle.py`, `test_fitness.py`,
`test_operators.py`, `test_stats.py`. All pass on a clean install.

## 7. Determinism

All algorithms use `numpy.random.default_rng(seed)` and identical seeds for
RS and GA-FT within a given run, so the trajectories are bit-exactly
reproducible across machines (modulo TensorFlow non-determinism inherent to
GPU operations — CPU inference is deterministic in our setup).
