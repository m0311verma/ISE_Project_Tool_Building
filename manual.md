# GA-FT User Manual

GA-FT is a command-line tool for fairness testing of binary classifiers.
It produces **Individual Discriminatory Instances (IDIs)** — input pairs that
differ only in a sensitive attribute but are classified differently by the
model under test.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Train sklearn models (one-off)

```bash
python -m ga_fairness_tester.train_sklearn_models
```

This trains Logistic Regression, Random Forest and Gradient Boosting on
each dataset and saves them to `models/`. The pre-trained DNNs in `DNN/`
are used directly — they require no training step.

## 3. Run a single fairness test

```bash
python -m ga_fairness_tester.main \
    --dataset adult \
    --model dnn \
    --sensitive gender \
    --algorithm both \
    --budget 1000 \
    --seed 0
```

### Flags

| Flag           | Choices                                | Description                          |
| -------------- | -------------------------------------- | ------------------------------------ |
| `--dataset`    | `adult`, `compas`                      | Dataset under test                   |
| `--model`      | `dnn`, `lr`, `rf`, `gbm`               | Model family under test              |
| `--sensitive`  | dataset-dependent (e.g. `gender`)      | Sensitive attribute to protect       |
| `--algorithm`  | `rs`, `ga`, `both`                     | RS = baseline; GA = GA-FT            |
| `--budget`     | int                                    | Max evaluations per run              |
| `--seed`       | int                                    | Random seed                          |

Sensitive features per dataset:
- **adult**: `gender`, `race`, `age`
- **compas**: `sex`, `race`

## 4. Output

Each run prints the IDI ratio, IDI count, IDI diversity, and runtime:

```
RS    | IDI_ratio=0.0700 | n_idis=  35 | diversity=0.4123 | 0.1s
GA-FT | IDI_ratio=0.1500 | n_idis=  75 | diversity=0.5021 | 0.2s
```

## 5. Running the tests

```bash
pytest tests/ -q
```

24 unit tests cover the oracle (IDI detection, deduplication, multi-valued
sensitive attributes), fitness function (normalisation, distance bonus,
α/β arithmetic), GA operators (tournament selection, crossover preserves
sensitive feature, mutation respects domains), and statistics
(Vargha-Delaney A12 edge cases, magnitude labelling).

## 6. α/β sensitivity ablation

```bash
python -m ga_fairness_tester.ablation \
    --dataset adult --model gbm --sensitive gender \
    --seeds 10 --budget 1000
```

Sweeps α ∈ {0.0, 0.3, 0.5, 0.7, 0.9, 1.0} with β = 1 − α, writing
`results/ablation.csv` and `figures/ablation_alpha.png`.

## 7. Adding a new dataset

1. Drop the pre-processed CSV into `dataset/`.
2. (Optional) Drop a Keras `.h5` model into `DNN/`.
3. Add an entry to `DATASET_CONFIG` in
   [`ga_fairness_tester/dataset_config.py`](ga_fairness_tester/dataset_config.py)
   with the sensitive feature column indices.
4. Re-run `train_sklearn_models` to train sklearn families on the new dataset.
