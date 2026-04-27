# GA-FT — Genetic Algorithm Fairness Tester

ISE Coursework — Tool Building Project (Problem 4: AI Model Fairness Testing)
University of Birmingham, MSc Intelligent Software Engineering.

GA-FT is a model-agnostic, search-based testing tool for finding **Individual
Discriminatory Instances (IDIs)** in classification models. It uses a Genetic
Algorithm with a dual exploitation/exploration fitness function and is
benchmarked against a Random Search baseline on the ADULT and COMPAS datasets
across four model families (DNN, Logistic Regression, Random Forest, Gradient
Boosting).

## Quick start

```bash
pip install -r requirements.txt
python -m ga_fairness_tester.train_sklearn_models                 # train LR/RF/GBM
python -m ga_fairness_tester.main \
    --dataset adult --model dnn --sensitive gender \
    --algorithm both --budget 1000 --seed 0
```

## Reproduce all experiments

```bash
python -m ga_fairness_tester.experiment --runs 30 --budget 1000
python -m ga_fairness_tester.stats
python -m ga_fairness_tester.visualise
python -m ga_fairness_tester.ablation --seeds 10                 # alpha/beta sweep
```

## Run the tests

```bash
pytest tests/ -q                                                 # 24 unit tests
```

See [`replication.md`](replication.md) for full reproduction details and
[`manual.md`](manual.md) for the user-facing manual.

## Repository layout

```
ga_fairness_tester/   tool source code (incl. ablation.py)
tests/                24 pytest unit tests
DNN/                  pre-trained Keras models from ideas-labo/ISE/lab4
dataset/              pre-processed CSVs from ideas-labo/ISE/lab4
models/               sklearn models trained by train_sklearn_models.py
results/              raw experiment CSVs + ablation CSV + stats summary
figures/              matplotlib plots used in the report
requirements.pdf      Python version + dependency list
manual.pdf            user manual
replication.pdf       step-by-step reproduction instructions
```

## License

Code: MIT. Datasets and pre-trained DNNs originate from
[ideas-labo/ISE/lab4](https://github.com/ideas-labo/ISE/tree/main/lab4).
