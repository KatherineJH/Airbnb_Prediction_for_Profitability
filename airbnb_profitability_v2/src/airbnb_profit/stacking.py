"""Leak-free stacking.

Meta-features are OUT-OF-FOLD predictions of the base learners on the TRAINING set
(host-grouped folds). The meta-learner is trained on those and the whole stack is evaluated once on
the untouched test set. (The original notebook trained and scored the meta-learner on the test set.)
"""
from __future__ import annotations

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from .config import derive_seed
from .models import make_pipeline


def build_stack(best: dict[str, dict], X_train, y_train, groups_train, cfg: dict, city: str) -> StackingRegressor:
    """``best`` maps model name -> tuning result dict (with ``best_params``)."""
    members = cfg["stacking"]["members"]
    folds = list(GroupKFold(n_splits=cfg["stacking"]["cv_folds"]).split(X_train, y_train, groups_train))
    estimators = [
        (m, make_pipeline(m, best[m]["best_params"], cfg, derive_seed(cfg["project"]["seed"], city, m, "final")))
        for m in members
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(positive=True),
        cv=folds,
        n_jobs=1,
        passthrough=False,
    )


def stack_weights(stack: StackingRegressor) -> dict[str, float]:
    names = [n for n, _ in stack.estimators]
    w = dict(zip(names, map(float, stack.final_estimator_.coef_)))
    w["intercept"] = float(stack.final_estimator_.intercept_)
    return w
