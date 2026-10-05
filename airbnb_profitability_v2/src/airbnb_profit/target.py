"""Target definition: estimated annual revenue from price and review activity.

The target is a deterministic function of the raw columns of a *single row*. No dataset-level
statistic (min-max, quantile, ...) is involved, so it cannot leak across the train/test split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COL = "target_log_revenue"
REVENUE_COL = "est_annual_revenue"


def parse_price(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")


def estimate_revenue(price: pd.Series, reviews_ltm: pd.Series, tcfg: dict) -> pd.Series:
    booked = np.minimum(
        reviews_ltm.astype(float) / tcfg["review_rate"] * tcfg["avg_stay_nights"],
        tcfg["max_occupancy"] * 365,
    )
    return price * booked


def select_population_and_target(raw: pd.DataFrame, tcfg: dict) -> tuple[pd.DataFrame, dict]:
    """Return (rows kept, aligned target frame) plus an audit dict of what was dropped."""
    price = parse_price(raw["price"])
    reviews = pd.to_numeric(raw["number_of_reviews_ltm"], errors="coerce")
    ok_price = price.between(tcfg["price_min"], tcfg["price_max"])
    ok_active = reviews >= tcfg["min_reviews_ltm"]
    keep = ok_price & ok_active
    audit = {
        "n_raw": int(len(raw)),
        "dropped_price_out_of_range_or_missing": int((~ok_price).sum()),
        "dropped_inactive": int((ok_price & ~ok_active).sum()),
        "n_final": int(keep.sum()),
    }
    revenue = estimate_revenue(price[keep], reviews[keep], tcfg)
    tgt = pd.DataFrame({REVENUE_COL: revenue, TARGET_COL: np.log1p(revenue)}, index=raw.index[keep])
    return raw.loc[keep], tgt, audit
