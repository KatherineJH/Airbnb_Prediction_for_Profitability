"""Evaluation: honest metrics, host-clustered bootstrap CIs, cross-city tests, importances."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance


def compute_metrics(y_log: np.ndarray, p_log: np.ndarray) -> dict[str, float]:
    """Metrics on the log scale (training scale) and on the euro scale.

    No MAPE: the target contains small values and the old filtering of non-finite ratios silently
    removed the hardest cases. WAPE (sum |err| / sum y) is the robust euro-scale relative error.
    """
    y_log, p_log = np.asarray(y_log, float), np.maximum(np.asarray(p_log, float), 0.0)
    err = p_log - y_log
    y_eur, p_eur = np.expm1(y_log), np.expm1(p_log)
    ss_res, ss_tot = np.sum(err**2), np.sum((y_log - y_log.mean()) ** 2)
    return {
        "r2_log": 1 - ss_res / ss_tot,
        "rmse_log": float(np.sqrt(np.mean(err**2))),
        "mae_log": float(np.mean(np.abs(err))),
        "rmse_eur": float(np.sqrt(np.mean((p_eur - y_eur) ** 2))),
        "mae_eur": float(np.mean(np.abs(p_eur - y_eur))),
        "medae_eur": float(np.median(np.abs(p_eur - y_eur))),
        "wape": float(np.sum(np.abs(p_eur - y_eur)) / np.sum(y_eur)),
        "spearman": float(stats.spearmanr(y_log, p_log)[0]) if np.ptp(p_log) > 0 else float("nan"),
    }


def cluster_bootstrap_ci(y_log, p_log, groups, iterations: int, seed: int,
                         metrics=("r2_log", "rmse_log"), alpha: float = 0.05) -> dict[str, tuple[float, float]]:
    """Resample whole hosts (rows of one host are not independent)."""
    y_log, p_log = np.asarray(y_log), np.asarray(p_log)
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(np.asarray(groups))
    members = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    draws = {m: [] for m in metrics}
    for _ in range(iterations):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[k] for k in pick])
        res = compute_metrics(y_log[idx], p_log[idx])
        for m in metrics:
            draws[m].append(res[m])
    return {m: (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))) for m, v in draws.items()}


def paired_wilcoxon(table: pd.DataFrame, metric: str, reference: str) -> pd.DataFrame:
    """Compare every model with ``reference`` across cities (paired, one value per city)."""
    wide = table.pivot(index="city", columns="model", values=metric)
    rows = []
    for m in wide.columns:
        if m == reference:
            continue
        diff = wide[m] - wide[reference]
        try:
            p = float(stats.wilcoxon(diff).pvalue)
        except ValueError:  # too few cities / all-zero differences
            p = float("nan")
        rows.append({"model": m, "reference": reference, "median_diff": float(diff.median()),
                     "cities_better": int((diff < 0).sum()) if metric.startswith(("rmse", "mae", "wape")) else int((diff > 0).sum()),
                     "n_cities": int(diff.notna().sum()), "wilcoxon_p": p})
    return pd.DataFrame(rows).sort_values("median_diff")


def permutation_table(model, X_test: pd.DataFrame, y_test, repeats: int, seed: int) -> pd.DataFrame:
    """Permutation importance on RAW input columns (an amenity list is one feature)."""
    res = permutation_importance(model, X_test, y_test, scoring="neg_root_mean_squared_error",
                                 n_repeats=repeats, random_state=seed, n_jobs=1)
    return (pd.DataFrame({"feature": X_test.columns, "importance": res.importances_mean, "std": res.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def importance_rank_correlation(importances: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Spearman between cities over the FULL aligned importance vector (not string lists)."""
    wide = pd.DataFrame({c: d.set_index("feature")["importance"] for c, d in importances.items()})
    return wide.corr(method="spearman")
