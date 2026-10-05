import numpy as np
import pandas as pd

from airbnb_profit import evaluate as ev
from airbnb_profit import features as F
from airbnb_profit import pipeline as P


def test_end_to_end_synthetic(tiny_cfg):
    run = P.run_all(tiny_cfg)  # prepare -> split -> tune -> evaluate -> report
    m = pd.read_csv(run.report / "metrics_all.csv")
    assert set(m["city"]) == {"bergamo", "roma", "firenze"}
    assert {"dummy", "stack", "ridge__tpe", "lgbm__ga"} <= set(m["model"])
    # planted signal must be recoverable, the median baseline must not be
    best = m[m["model"] != "dummy"].groupby("city")["r2_log"].max()
    dummy = m[m["model"] == "dummy"].set_index("city")["r2_log"]
    assert (dummy.abs() < 0.05).all(), dummy
    assert ((best - dummy) > 0.08).all(), (best, dummy)  # tiny data + 3 trials: only demand a clear gap
    # exactly one CV-selected model per city
    assert (m.groupby("city")["selected_by_cv"].sum() == 1).all()


def test_no_host_overlap_and_disjoint_ids(tiny_cfg):
    run = P.run_all(tiny_cfg, stages=["prepare", "split"])
    for city in run.cities:
        X_tr, _, g_tr, X_te, _, g_te, _ = P.load_split(run, city)
        assert set(g_tr).isdisjoint(g_te)
        assert set(X_tr.columns) == set(F.FEATURE_COLUMNS)


def test_cached_stages_and_config_guard(tiny_cfg):
    import pytest
    from airbnb_profit.config import with_overrides
    run = P.run_all(tiny_cfg, stages=["prepare"])
    stamp = (run.data / "bergamo.parquet").stat().st_mtime_ns
    P.run_all(tiny_cfg, stages=["prepare"])  # cached -> untouched
    assert (run.data / "bergamo.parquet").stat().st_mtime_ns == stamp
    with pytest.raises(RuntimeError):
        P.Run(with_overrides(tiny_cfg, **{"tuning.n_trials": 99}))  # same run_name, different config


def test_determinism(tiny_cfg):
    from airbnb_profit.config import with_overrides
    a = P.run_all(with_overrides(tiny_cfg, **{"project.run_name": "det_a"}), cities=["bergamo"], stages=["prepare", "split", "tune", "evaluate"])
    b = P.run_all(with_overrides(tiny_cfg, **{"project.run_name": "det_b"}), cities=["bergamo"], stages=["prepare", "split", "tune", "evaluate"])
    pa = pd.read_parquet(a.eval / "bergamo_predictions.parquet")
    pb = pd.read_parquet(b.eval / "bergamo_predictions.parquet")
    pd.testing.assert_frame_equal(pa, pb)


def test_metrics_and_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.normal(8, 1, 400)
    perfect = ev.compute_metrics(y, y)
    assert perfect["r2_log"] == 1 and perfect["rmse_log"] == 0
    noisy = y + rng.normal(0, 1, 400)
    ci = ev.cluster_bootstrap_ci(y, noisy, rng.integers(0, 40, 400), 50, 1)
    assert ci["rmse_log"][0] < ev.compute_metrics(y, noisy)["rmse_log"] < ci["rmse_log"][1] + 0.3
