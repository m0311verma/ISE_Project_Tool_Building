"""Entry point: single-config run for ad-hoc inspection.

Example:
  python -m ga_fairness_tester.main --dataset adult --model dnn \
      --sensitive gender --algorithm both --budget 1000 --seed 0
"""

from __future__ import annotations

import argparse

from .data_loader import load_dataset
from .ga_ft import ga_ft
from .model_adapter import load_adapter
from .random_search import random_search


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model",   required=True, choices=["dnn", "lr", "rf", "gbm"])
    p.add_argument("--sensitive", required=True)
    p.add_argument("--algorithm", default="both", choices=["rs", "ga", "both"])
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--seed",   type=int, default=0)
    args = p.parse_args(argv)

    data = load_dataset(args.dataset)
    adapter = load_adapter(args.dataset, args.model)

    if args.algorithm in ("rs", "both"):
        r = random_search(data, adapter, args.sensitive, args.budget, args.seed)
        print(f"RS    | IDI_ratio={r['idi_ratio']:.4f} | n_idis={r['n_idis']:>4} "
              f"| diversity={r['idi_diversity']:.4f} | {r['runtime_seconds']:.1f}s")
    if args.algorithm in ("ga", "both"):
        r = ga_ft(data, adapter, args.sensitive, args.budget, args.seed)
        print(f"GA-FT | IDI_ratio={r['idi_ratio']:.4f} | n_idis={r['n_idis']:>4} "
              f"| diversity={r['idi_diversity']:.4f} | {r['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
