# Airbnb Italy – Estimated Revenue Prediction (v2, leak-free & reproducible)

Re-development of the original three notebooks (`1.Preprocessing`, `2.Modeling`, `3.Compare_Eval_Visual`)
as **one notebook driving a tested Python package**.

## What changed and why

| # | Problem in v1 | Fix in v2 |
|---|---|---|
| 1 | Stacking / Bayesian / GA meta-learners were **trained and scored on the test set** (adj. R² 1.00, "accuracy" 99 %) | Test set is split off first (by **host**) and touched once. Tuning = CV on train only. Stacking = `StackingRegressor` with **out-of-fold** meta-features, non-negative linear meta-learner |
| 2 | Target was built from inputs (`host_total_listings_count`, price…), min-max scaled on the whole data, clipped logs; "profitability" was vague | Explicit, documented **estimated annual revenue** = price × min(reviews_ltm / review_rate × stay, max occupancy). Features are **ex-ante listing attributes only**; `price`, review, availability columns are structurally excluded (`FORBIDDEN_FEATURES` guard + tests). No dataset-level scaling of the target |
| 3 | Imputation, label-encoding, scaling, amenity top-100 learned on all rows before the split | Stateless parsing only before the split; everything learned (imputer, one-hot, **target encoding**, amenity vocabulary) is **inside the sklearn Pipeline**, refit in each fold |
| – | MAPE with non-finite ratios silently dropped | R², RMSE/MAE (log & €), median AE, WAPE, Spearman; **host-clustered bootstrap CIs** |
| – | `n_estimators` chosen by *training* score | All models tuned by host-grouped CV (Optuna) |
| – | Only the meta-learner tuned; identical "best" params in all cities | Every model family tuned **per city**, with its own seed |
| – | GA search space/direction bugs (n_estimators ∈ {0,1}, accuracy minimised) | GA = Optuna **NSGA-II** sampler on the same objective as Bayesian (TPE) |
| – | Same host in train & test | `GroupShuffleSplit`/`GroupKFold` on `host_id` (asserted in code and tests) |
| – | Kendall/Spearman on lists of strings | Spearman on the **aligned permutation-importance vectors**; permutation importance on raw columns instead of impurity gain |
| – | City "centre" for regions (Sicilia, Puglia, Trentino) | Distance to the **nearest of several anchors** per region |
| – | No baseline, only p=0 ANOVA | Median baseline + tuned Ridge; paired **Wilcoxon** across cities |
| – | Copy-paste ×10 cities, manual `iloc` slicing | One loop over cities; results in tidy CSV |
| – | `eval()` on amenities, Python 3.7 | `json.loads`/`literal_eval`, pinned modern stack |

## Layout

```
airbnb_profitability_v2/
├─ config/default.yaml          single source of truth (hashed into every run)
├─ src/airbnb_profit/           the library: data, target, features, models, tuning, stacking, evaluate, pipeline, viz, cli
├─ notebooks/Airbnb_Profitability_Workflow.ipynb   the one notebook (thin: calls the package, shows results)
├─ scripts/run_pipeline.py      same workflow headless:  python scripts/run_pipeline.py --help
├─ scripts/download_data.py     fetch merged_file.csv
├─ tests/                       pytest: leakage guards, formulas, end-to-end (synthetic), determinism
├─ data/raw/                    put merged_file.csv here (git-ignored)
└─ runs/<run_name>/             all artifacts: data, splits, tuning, eval, models, importance, report, figures, manifest
```

## Workflow

```
prepare → split → tune (train only) → evaluate (test, once) → report
```

Each stage is cached on disk and resumable; the notebook and the CLI call the *same* functions.

```bash
pip install -r requirements.txt && pip install -e .
python scripts/download_data.py                      # or copy merged_file.csv into data/raw/
python scripts/run_pipeline.py                       # full run, all 10 cities
python scripts/run_pipeline.py --cities roma milano --stages tune evaluate report
python scripts/run_pipeline.py --synthetic --set project.run_name=demo --set tuning.n_trials=5   # no real data needed
pytest                                               # 100 % synthetic, a few minutes
jupyter lab notebooks/Airbnb_Profitability_Workflow.ipynb
```

## Reproducibility

* One config file; its hash is stored in `runs/<run>/manifest.json` together with the git commit,
  package versions, Python/OS and the **SHA-256 of the raw data** (pin it with `data.expected_sha256`).
* Re-using a `run_name` with a different config is refused.
* Every random component has a seed derived from `(base seed, city, model, role)` – no global state.
* `tests/test_workflow.py::test_determinism` runs the whole pipeline twice and asserts identical predictions.

## Honest limitations

* **Revenue, not profit**: there is no cost data. The occupancy model (reviews → nights) is a standard
  public heuristic, not ground truth; its parameters live in `config.target` and should be varied as a sensitivity analysis.
* The modelled population is **listings with ≥ 1 review in the last 12 months** (`min_reviews_ltm`); results do not apply to never-booked listings.
* The model is **ex-ante**: it predicts revenue *potential* from listing/host attributes. Because price is part of the target recipe, it is not a feature; a model that also uses price would be a different (and easier) question.
* One snapshot (Sept 2022): no temporal validation is possible.
* Test metrics come from one host-grouped hold-out; the bootstrap CIs quantify test-sample noise, not split noise.
* Synthetic data (`--synthetic`) only exercises the code; its numbers mean nothing about Italy.
