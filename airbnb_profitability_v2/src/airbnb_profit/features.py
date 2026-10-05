"""Feature engineering.

Two layers, deliberately separated:

* ``build_features``  - STATELESS row-wise parsing (string -> number, flags, distance).
  Uses no statistic of the dataset, so running it before the split is leak-free.
* ``make_preprocessor`` - everything that LEARNS from data (imputation, scaling, target
  encoding, amenity vocabulary). It lives inside the sklearn Pipeline, so it is re-fitted on
  the training part of every CV fold and never sees validation/test rows.
"""
from __future__ import annotations

import ast
import inspect
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

NUMERIC = [
    "accommodates", "bedrooms", "beds", "bathroom_num", "bathroom_shared",
    "minimum_nights", "maximum_nights", "latitude", "longitude", "distance_anchor_km",
    "host_tenure_years", "host_total_listings_count", "host_response_rate",
    "host_acceptance_rate", "n_amenities",
    "host_is_superhost", "instant_bookable", "has_license", "has_host_about",
    "has_neighborhood_overview", "host_identity_verified", "host_has_profile_pic",
    "verif_email", "verif_phone", "verif_work_email",
]
CAT_LOW = ["room_type", "host_response_time", "host_location_group"]
CAT_HIGH = ["property_type", "neighbourhood_cleansed"]  # many levels -> target encoding
AMENITY = "amenities"
FEATURE_COLUMNS = NUMERIC + CAT_LOW + CAT_HIGH + [AMENITY]
META_COLUMNS = ["id", "host_id"]

# Anything that is part of the target recipe, or a post-hoc demand/occupancy signal.
FORBIDDEN_FEATURES = {
    "price", "est_annual_revenue", "target_log_revenue",
    "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d", "reviews_per_month",
    "first_review", "last_review", "estimated_occupancy_l365d", "estimated_revenue_l365d",
    "availability_30", "availability_60", "availability_90", "availability_365", "has_availability",
} | {f"review_scores_{k}" for k in
     ["rating", "accuracy", "cleanliness", "checkin", "communication", "location", "value"]}


def assert_no_leakage(columns) -> None:
    bad = sorted(set(columns) & FORBIDDEN_FEATURES)
    if bad:
        raise AssertionError(f"Target-derived / post-hoc columns used as features: {bad}")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def distance_to_nearest_anchor(lat: pd.Series, lon: pd.Series, anchors) -> np.ndarray:
    d = [haversine_km(lat.to_numpy(float), lon.to_numpy(float), a_lat, a_lon) for a_lat, a_lon in anchors]
    return np.min(d, axis=0)


def _tf(s: pd.Series) -> pd.Series:
    return s.map({"t": 1.0, "f": 0.0})  # missing stays NaN (imputer adds an indicator)


def _pct(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.rstrip("%"), errors="coerce")


def parse_amenities(value) -> list[str]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v).strip().lower() for v in value]
    if not isinstance(value, str):
        return []
    for loader in (json.loads, ast.literal_eval):  # never eval()
        try:
            return [str(v).strip().lower() for v in loader(value)]
        except Exception:
            continue
    return []


def parse_bathrooms(text: pd.Series) -> tuple[pd.Series, pd.Series]:
    t = text.astype("string").str.lower()
    num = pd.to_numeric(t.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")
    num = num.where(~t.str.contains("half", na=False), 0.5)  # 'Half-bath' -> 0.5 (was NaN before)
    shared = t.str.contains("shared", na=False).astype(float).where(t.notna())
    return num.astype(float), shared.astype(float)


def build_features(raw: pd.DataFrame, cfg: dict, city: str) -> pd.DataFrame:
    clip = cfg["features"]["clip"]
    out = pd.DataFrame(index=raw.index)
    out["id"], out["host_id"] = raw["id"], raw["host_id"]
    for c in ["accommodates", "bedrooms", "beds", "latitude", "longitude"]:
        out[c] = pd.to_numeric(raw[c], errors="coerce")
    out["bathroom_num"], out["bathroom_shared"] = parse_bathrooms(raw["bathrooms_text"])
    for c in ["minimum_nights", "maximum_nights", "host_total_listings_count"]:
        out[c] = pd.to_numeric(raw[c], errors="coerce").clip(upper=clip[c])
    out["distance_anchor_km"] = distance_to_nearest_anchor(out["latitude"], out["longitude"], cfg["geo"]["anchors"][city])
    since, scraped = pd.to_datetime(raw["host_since"], errors="coerce"), pd.to_datetime(raw["last_scraped"], errors="coerce")
    out["host_tenure_years"] = ((scraped - since).dt.days / 365.25).clip(lower=0)
    out["host_response_rate"], out["host_acceptance_rate"] = _pct(raw["host_response_rate"]), _pct(raw["host_acceptance_rate"])
    for c in ["host_is_superhost", "instant_bookable", "host_identity_verified", "host_has_profile_pic"]:
        out[c] = _tf(raw[c])
    out["has_license"] = raw["license"].notna().astype(float)
    out["has_host_about"] = raw["host_about"].notna().astype(float)
    out["has_neighborhood_overview"] = raw["neighborhood_overview"].notna().astype(float)
    ver = raw["host_verifications"].astype("string").fillna("")
    for k in ["email", "phone", "work_email"]:
        out[f"verif_{k}"] = ver.str.contains(f"'{k}'", regex=False).astype(float)
    out["room_type"] = raw["room_type"].fillna("unknown")
    out["host_response_time"] = raw["host_response_time"].fillna("unknown")
    loc = raw["host_location"]
    out["host_location_group"] = np.where(loc.isna(), "unknown", np.where(loc.astype(str).str.contains("italy", case=False), "italy", "abroad"))
    out["property_type"] = raw["property_type"].fillna("unknown")
    out["neighbourhood_cleansed"] = raw["neighbourhood_cleansed"].fillna("unknown")
    amen = raw["amenities"].map(parse_amenities)
    out[AMENITY] = amen
    out["n_amenities"] = amen.map(len).astype(float)
    cols = META_COLUMNS + FEATURE_COLUMNS
    assert_no_leakage(cols)
    return out[cols]


class AmenityMultiHot(BaseEstimator, TransformerMixin):
    """Multi-hot of the ``top_n`` most frequent amenities. Vocabulary is learned in ``fit`` only."""

    def __init__(self, top_n: int = 40, min_count: int = 20):
        self.top_n = top_n
        self.min_count = min_count

    @staticmethod
    def _lists(X):
        return np.asarray(X, dtype=object)[:, 0]

    def fit(self, X, y=None):
        counts = Counter(a for lst in self._lists(X) for a in set(lst))
        self.vocabulary_ = [a for a, c in counts.most_common(self.top_n) if c >= self.min_count]
        return self

    def transform(self, X):
        index = {a: i for i, a in enumerate(self.vocabulary_)}
        rows = self._lists(X)
        M = np.zeros((len(rows), len(index)), dtype=np.float32)
        for r, lst in enumerate(rows):
            for a in lst:
                j = index.get(a)
                if j is not None:
                    M[r, j] = 1.0
        return M

    def get_feature_names_out(self, input_features=None):
        return np.array([f"amenity::{a}" for a in self.vocabulary_], dtype=object)


def _target_encoder(seed: int) -> TargetEncoder:
    """Seeded in-fold target encoding on every supported scikit-learn version."""
    if inspect.signature(TargetEncoder.__init__).parameters["shuffle"].default == "deprecated":  # sklearn >= 1.9
        return TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=seed))
    return TargetEncoder(target_type="continuous", random_state=seed)


def make_preprocessor(cfg: dict, seed: int, scale: bool = False) -> ColumnTransformer:
    fcfg = cfg["features"]
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    from sklearn.pipeline import Pipeline
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), NUMERIC),
            ("cat_low", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=fcfg["onehot_min_frequency"], sparse_output=False), CAT_LOW),
            ("cat_high", _target_encoder(seed), CAT_HIGH),
            ("amen", AmenityMultiHot(fcfg["top_amenities"], fcfg["amenity_min_count"]), [AMENITY]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
