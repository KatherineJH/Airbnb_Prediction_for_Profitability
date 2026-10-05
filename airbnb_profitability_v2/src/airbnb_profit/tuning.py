"""Hyper-parameter optimisation on TRAIN data only (host-grouped CV).

``tpe`` = Bayesian optimisation (Tree-structured Parzen Estimator).
``ga``  = genetic algorithm (NSGA-II sampler: tournament selection, crossover, mutation).
Both optimise the same objective: mean CV RMSE on the log-revenue target.
"""
from __future__ import annotations

import numpy as np
import optuna
from sklearn.model_selection import GroupKFold, cross_val_score

from .config import derive_seed
from .models import make_pipeline, suggest_params

optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_sampler(name: str, seed: int, population: int):
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "ga":
        return optuna.samplers.NSGAIISampler(seed=seed, population_size=population)
    raise KeyError(name)


def tune(model: str, sampler: str, X, y, groups, cfg: dict, city: str) -> dict:
    seed = derive_seed(cfg["project"]["seed"], city, model, sampler)
    tcfg = cfg["tuning"]
    cv = GroupKFold(n_splits=tcfg["cv_folds"])
    splits = list(cv.split(X, y, groups))  # fixed folds -> every trial sees identical data

    def objective(trial):
        params = suggest_params(model, trial)
        pipe = make_pipeline(model, params, cfg, seed)
        scores = cross_val_score(pipe, X, y, cv=splits, scoring="neg_root_mean_squared_error")
        return float(-np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=make_sampler(sampler, seed, tcfg["ga_population"]))
    study.optimize(objective, n_trials=tcfg["n_trials"], show_progress_bar=False)
    return {
        "city": city, "model": model, "sampler": sampler, "seed": seed,
        "best_params": study.best_params, "best_cv_rmse": study.best_value,
        "trials": [{"number": t.number, "value": t.value, **{f"p_{k}": v for k, v in t.params.items()}}
                   for t in study.trials],
    }
