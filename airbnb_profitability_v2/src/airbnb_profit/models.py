"""Estimator factory and hyper-parameter search spaces."""
from __future__ import annotations

import lightgbm as lgb
import xgboost as xgb
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from .features import make_preprocessor

MODEL_NAMES = ["ridge", "rf", "lgbm", "xgb"]


def suggest_params(model: str, trial) -> dict:
    if model == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-2, 1e3, log=True)}
    if model == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 4, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    if model == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if model == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
    raise KeyError(model)


def make_estimator(model: str, params: dict, seed: int, n_jobs: int):
    if model == "ridge":
        return Ridge(random_state=seed, **params)
    if model == "rf":
        return RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **params)
    if model == "lgbm":
        return lgb.LGBMRegressor(random_state=seed, n_jobs=n_jobs, verbose=-1, deterministic=True,
                                 force_row_wise=True, **params)
    if model == "xgb":
        return xgb.XGBRegressor(random_state=seed, n_jobs=n_jobs, tree_method="hist", **params)
    if model == "dummy":
        return DummyRegressor(strategy="median")
    raise KeyError(model)


def make_pipeline(model: str, params: dict, cfg: dict, seed: int) -> Pipeline:
    """Preprocessor + estimator. Everything learned from data is fitted inside ``fit``."""
    n_jobs = cfg["compute"]["n_jobs"]
    return Pipeline([
        ("prep", make_preprocessor(cfg, seed, scale=(model == "ridge"))),
        ("model", make_estimator(model, params, seed, n_jobs)),
    ])
