# Requirements

## Python

* Python **3.10 or later** (developed and tested on Python 3.13.7).

## Operating systems

Tested on macOS (Darwin 25.4.0). The tool is pure Python and should run on
Linux and Windows without modification.

## Python package dependencies

| Package        | Minimum version | Purpose                                  |
| -------------- | --------------- | ---------------------------------------- |
| numpy          | 1.24            | Numerical arrays                         |
| pandas         | 2.0             | CSV loading                              |
| scikit-learn   | 1.3             | LR / RF / GBM models, train/test split   |
| scipy          | 1.10            | Wilcoxon rank-sum statistical test       |
| matplotlib     | 3.7             | Box plots and convergence curves         |
| joblib         | 1.3             | Persisting trained sklearn models        |
| tensorflow     | 2.15            | Loading the pre-trained DNN .h5 models   |
| keras          | 3.0             | Keras API surface (bundled with TF 2.x)  |
| pytest         | 8.0             | Unit-test runner for `tests/`            |

Install everything in one step:

```bash
pip install -r requirements.txt
```

## Hardware

No GPU required. The full 30-run experiment matrix completes in under
two hours on a 2024 MacBook Pro (M-series CPU). DNN inference dominates
runtime; sklearn families finish in under a minute each.

## Data and models

The `DNN/` and `dataset/` directories contain the pre-trained Keras `.h5`
models and processed CSV datasets sourced from
[ideas-labo/ISE/lab4](https://github.com/ideas-labo/ISE/tree/main/lab4).
No additional downloads are required.
