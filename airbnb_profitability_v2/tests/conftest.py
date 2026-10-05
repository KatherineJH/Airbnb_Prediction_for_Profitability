import pytest

from airbnb_profit.config import load_config, with_overrides


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory):
    cfg = load_config()
    return with_overrides(
        cfg,
        **{
            "project.run_name": "pytest",
            "paths": {"raw_csv": "data/raw/none.csv", "runs_dir": str(tmp_path_factory.mktemp("runs"))},
            "data.source": "synthetic",
            "data.cities": ["bergamo", "roma", "firenze"],
            "data.synthetic_rows_per_city": 500,
            "tuning.n_trials": 3,
            "tuning.cv_folds": 3,
            "tuning.ga_population": 2,
            "stacking.cv_folds": 3,
            "evaluation.bootstrap_iterations": 20,
            "evaluation.permutation_repeats": 2,
            "compute.n_jobs": 1,
        },
    )
