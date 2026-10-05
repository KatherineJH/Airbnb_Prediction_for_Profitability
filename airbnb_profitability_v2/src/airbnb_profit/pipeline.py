"""Workflow stages. Each stage reads/writes files under ``runs/<run_name>/`` and is cached:

    prepare -> split -> tune -> evaluate -> report

The CLI (``airbnb_profit.cli``) and the notebook call exactly these functions, so a result shown
in the notebook is the result of the scripted pipeline, not of ad-hoc cells.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

from . import evaluate as ev
from . import features as F
from . import target as T
from .config import PROJECT_ROOT, config_hash, derive_seed, resolve
from .data import load_raw, sha256_file
from .manifest import build_manifest
from .models import make_estimator, make_pipeline
from .stacking import build_stack, stack_weights
from .synthetic import make_synthetic_raw
from .tuning import tune

log = logging.getLogger("airbnb_profit")
STAGES = ["prepare", "split", "tune", "evaluate", "report"]


# --------------------------------------------------------------------------- run directory
class Run:
    def __init__(self, cfg: dict, force: bool = False):
        self.cfg, self.force = cfg, force
        self.root = resolve(cfg, "runs_dir") / cfg["project"]["run_name"]
        for name in ["data", "splits", "tuning", "eval", "models", "importance", "report", "figures"]:
            setattr(self, name, self.root / name)
            (self.root / name).mkdir(parents=True, exist_ok=True)
        snap = self.root / "config_snapshot.yaml"
        h = config_hash(cfg)
        if snap.exists():
            old = yaml.safe_load(snap.read_text(encoding="utf8"))
            if config_hash(old) != h and not force:
                raise RuntimeError(
                    f"Run '{cfg['project']['run_name']}' already exists with a different config "
                    f"({config_hash(old)} != {h}). Use a new project.run_name or --force.")
        snap.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf8")

    def cached(self, path: Path) -> bool:
        if path.exists() and not self.force:
            log.info("cached: %s", path.relative_to(self.root))
            return True
        return False

    @property
    def cities(self) -> list[str]:
        return list(self.cfg["data"]["cities"])


def _json_dump(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf8")


# --------------------------------------------------------------------------- 1. prepare
def stage_prepare(run: Run, cities: list[str] | None = None) -> pd.DataFrame:
    cfg, cities = run.cfg, cities or run.cities
    audit_path = run.data / "audit.json"
    if all((run.data / f"{c}.parquet").exists() for c in cities) and audit_path.exists() and not run.force:
        log.info("cached: data/*.parquet")
        return pd.DataFrame(json.loads(audit_path.read_text())).T

    source = cfg["data"]["source"]
    if source == "synthetic":
        raw, fingerprint = make_synthetic_raw(cfg, cities), f"synthetic:seed={cfg['project']['seed']}"
        log.warning("USING SYNTHETIC DATA - numbers are NOT about the real market.")
    else:
        path = resolve(cfg, "raw_csv")
        raw = load_raw(path, cfg["data"]["expected_sha256"])
        fingerprint = sha256_file(path)
    _json_dump(build_manifest(cfg, fingerprint), run.root / "manifest.json")

    n_before = len(raw)
    raw = raw.drop_duplicates(subset=["city", "id"])
    audit: dict[str, dict] = {}
    for city in cities:
        sub = raw[raw[cfg["data"]["city_column"]] == city]
        if sub.empty:
            raise ValueError(f"No rows for city '{city}'")
        kept, tgt, a = T.select_population_and_target(sub, cfg["target"])
        frame = pd.concat([F.build_features(kept, cfg, city), tgt], axis=1)
        F.assert_no_leakage([c for c in frame.columns if c not in (T.REVENUE_COL, T.TARGET_COL)])
        frame.to_parquet(run.data / f"{city}.parquet", index=False)
        audit[city] = a
    _json_dump(audit, audit_path)
    log.info("prepared %d cities (%d raw rows incl. duplicates)", len(cities), n_before)
    return pd.DataFrame(audit).T


def load_city(run: Run, city: str) -> pd.DataFrame:
    return pd.read_parquet(run.data / f"{city}.parquet")


# --------------------------------------------------------------------------- 2. split
def stage_split(run: Run, city: str) -> dict:
    path = run.splits / f"{city}.json"
    if run.cached(path):
        return json.loads(path.read_text())
    cfg = run.cfg
    df = load_city(run, city)
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg["split"]["test_size"],
                            random_state=derive_seed(cfg["project"]["seed"], city, "split"))
    tr, te = next(gss.split(df, groups=df[cfg["split"]["group_column"]]))
    assert set(df.iloc[tr]["host_id"]).isdisjoint(df.iloc[te]["host_id"]), "host leakage across split"
    out = {"train_ids": df.iloc[tr]["id"].astype(int).tolist(), "test_ids": df.iloc[te]["id"].astype(int).tolist()}
    _json_dump(out, path)
    return out


def load_split(run: Run, city: str):
    """(X_train, y_train, g_train, X_test, y_test, g_test, ids_test). Test set: use ONCE, at evaluation."""
    df, sp = load_city(run, city), json.loads((run.splits / f"{city}.json").read_text())
    tr, te = df[df["id"].isin(sp["train_ids"])], df[df["id"].isin(sp["test_ids"])]
    return (tr[F.FEATURE_COLUMNS], tr[T.TARGET_COL].to_numpy(), tr["host_id"].to_numpy(),
            te[F.FEATURE_COLUMNS], te[T.TARGET_COL].to_numpy(), te["host_id"].to_numpy(), te["id"].to_numpy())


# --------------------------------------------------------------------------- 3. tune (train only)
def stage_tune(run: Run, city: str) -> list[dict]:
    cfg = run.cfg
    X_tr, y_tr, g_tr, *_ = load_split(run, city)
    results = []
    for model, samplers in cfg["tuning"]["plan"].items():
        for sampler in samplers:
            path = run.tuning / f"{city}__{model}__{sampler}.json"
            if run.cached(path):
                results.append(json.loads(path.read_text()))
                continue
            log.info("tuning %s / %s / %s", city, model, sampler)
            res = tune(model, sampler, X_tr, y_tr, g_tr, cfg, city)
            _json_dump(res, path)
            results.append(res)
    return results


def load_tuning(run: Run, city: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(run.tuning.glob(f"{city}__*.json"))]


def best_per_family(results: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for r in results:
        if r["model"] not in best or r["best_cv_rmse"] < best[r["model"]]["best_cv_rmse"]:
            best[r["model"]] = r
    return best


# --------------------------------------------------------------------------- 4. evaluate (test, once)
def fit_tuned(run: Run, city: str, res: dict, X_tr, y_tr):
    seed = derive_seed(run.cfg["project"]["seed"], city, res["model"], "final")
    return make_pipeline(res["model"], res["best_params"], run.cfg, seed).fit(X_tr, y_tr)


def stage_evaluate(run: Run, city: str) -> pd.DataFrame:
    path = run.eval / f"{city}_metrics.csv"
    if run.cached(path):
        return pd.read_csv(path)
    cfg, ecfg = run.cfg, run.cfg["evaluation"]
    X_tr, y_tr, g_tr, X_te, y_te, g_te, ids_te = load_split(run, city)
    results = load_tuning(run, city)
    if not results:
        raise RuntimeError(f"No tuning results for {city}; run the 'tune' stage first.")
    best = best_per_family(results)
    selected = min(results, key=lambda r: r["best_cv_rmse"])  # chosen by TRAIN CV, never by test

    preds: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    fitted: dict[str, object] = {}

    dummy = make_estimator("dummy", {}, 0, 1).fit(np.zeros((len(y_tr), 1)), y_tr)
    preds["dummy"], meta["dummy"] = dummy.predict(np.zeros((len(y_te), 1))), {"family": "dummy", "sampler": "-", "cv_rmse": np.nan}
    for r in results:
        label = f"{r['model']}__{r['sampler']}"
        fitted[label] = fit_tuned(run, city, r, X_tr, y_tr)
        preds[label] = fitted[label].predict(X_te)
        meta[label] = {"family": r["model"], "sampler": r["sampler"], "cv_rmse": r["best_cv_rmse"]}
    members_missing = [m for m in cfg["stacking"]["members"] if m not in best]
    if not members_missing:
        stack = build_stack(best, X_tr, y_tr, g_tr, cfg, city).fit(X_tr, y_tr)
        preds["stack"], meta["stack"] = stack.predict(X_te), {"family": "stack", "sampler": "oof", "cv_rmse": np.nan}
        _json_dump(stack_weights(stack), run.eval / f"{city}_stack_weights.json")
        fitted["stack"] = stack

    sel_label = f"{selected['model']}__{selected['sampler']}"
    rows = []
    for label, p in preds.items():
        m = ev.compute_metrics(y_te, p)
        ci = ev.cluster_bootstrap_ci(y_te, p, g_te, ecfg["bootstrap_iterations"],
                                     derive_seed(cfg["project"]["seed"], city, label, "boot"))
        rows.append({"city": city, "model": label, **meta[label], **m,
                     "r2_log_lo": ci["r2_log"][0], "r2_log_hi": ci["r2_log"][1],
                     "rmse_log_lo": ci["rmse_log"][0], "rmse_log_hi": ci["rmse_log"][1],
                     "n_train": len(y_tr), "n_test": len(y_te), "selected_by_cv": label == sel_label})
    table = pd.DataFrame(rows)

    pd.DataFrame({"id": ids_te, "y_true_log": y_te, **preds}).to_parquet(run.eval / f"{city}_predictions.parquet", index=False)
    imp = ev.permutation_table(fitted[sel_label], X_te, y_te, ecfg["permutation_repeats"],
                               derive_seed(cfg["project"]["seed"], city, "perm"))
    imp.to_csv(run.importance / f"{city}.csv", index=False)
    if cfg["compute"]["save_models"]:
        joblib.dump(fitted[sel_label], run.models / f"{city}__selected.joblib", compress=3)
        if "stack" in fitted:
            joblib.dump(fitted["stack"], run.models / f"{city}__stack.joblib", compress=3)
    table.to_csv(path, index=False)
    return table


# --------------------------------------------------------------------------- 5. report
def stage_report(run: Run) -> dict[str, pd.DataFrame]:
    from . import viz  # matplotlib only needed here

    cities = [c for c in run.cities if (run.eval / f"{c}_metrics.csv").exists()]
    metrics = pd.concat([pd.read_csv(run.eval / f"{c}_metrics.csv") for c in cities], ignore_index=True)
    metrics.to_csv(run.report / "metrics_all.csv", index=False)
    summary = (metrics.groupby("model")[["r2_log", "rmse_log", "mae_eur", "medae_eur", "wape", "spearman"]]
               .agg(["mean", "median"]).round(4))
    summary.to_csv(run.report / "summary_by_model.csv")
    tests = pd.concat([
        ev.paired_wilcoxon(metrics, "rmse_log", "dummy"),
        ev.paired_wilcoxon(metrics, "rmse_log", "ridge__tpe"),
    ], ignore_index=True)
    tests.to_csv(run.report / "wilcoxon_tests.csv", index=False)
    imps = {c: pd.read_csv(run.importance / f"{c}.csv") for c in cities}
    rank = ev.importance_rank_correlation(imps) if len(imps) > 1 else pd.DataFrame()
    rank.to_csv(run.report / "importance_rank_spearman.csv")
    out = {"metrics": metrics, "summary": summary, "tests": tests, "rank_corr": rank}
    for name, fig in viz.report_figures(run, metrics, imps, rank).items():
        fig.savefig(run.figures / f"{name}.png", dpi=130, bbox_inches="tight")
        viz.close(fig)
    return out


# --------------------------------------------------------------------------- driver
def run_all(cfg: dict, stages: list[str] | None = None, cities: list[str] | None = None, force: bool = False) -> Run:
    stages = stages or STAGES
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown stages {unknown}; choose from {STAGES}")
    run = Run(cfg, force=force)
    cities = cities or run.cities
    if "prepare" in stages:
        stage_prepare(run, cities)
    for city in cities:
        if "split" in stages:
            stage_split(run, city)
        if "tune" in stages:
            stage_tune(run, city)
        if "evaluate" in stages:
            stage_evaluate(run, city)
    if "report" in stages:
        stage_report(run)
    return run


def clean_run(cfg: dict) -> None:
    root = resolve(cfg, "runs_dir") / cfg["project"]["run_name"]
    if root.exists():
        shutil.rmtree(root)
