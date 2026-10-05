import numpy as np
import pandas as pd
import pytest

from airbnb_profit import features as F
from airbnb_profit import target as T
from airbnb_profit.synthetic import make_synthetic_raw


def test_target_formula():
    t = {"review_rate": 0.5, "avg_stay_nights": 3, "max_occupancy": 0.9, "min_reviews_ltm": 1, "price_min": 10, "price_max": 5000}
    raw = pd.DataFrame({"price": ["$100.00", "$1,000.00", "$5.00", "$100.00"], "number_of_reviews_ltm": [10, 500, 10, 0]})
    kept, tgt, audit = T.select_population_and_target(raw, t)
    # 10 reviews / 0.5 * 3 = 60 nights * 100 ; 500 reviews saturates at 0.9*365 nights
    assert tgt[T.REVENUE_COL].iloc[0] == pytest.approx(6000)
    assert tgt[T.REVENUE_COL].iloc[1] == pytest.approx(1000 * 0.9 * 365)
    assert audit["dropped_price_out_of_range_or_missing"] == 1 and audit["dropped_inactive"] == 1


def test_features_have_no_target_ingredients(tiny_cfg):
    raw = make_synthetic_raw(tiny_cfg, ["roma"])
    f = F.build_features(raw, tiny_cfg, "roma")
    assert not set(f.columns) & F.FORBIDDEN_FEATURES
    assert set(F.FEATURE_COLUMNS) <= set(f.columns)


def test_guard_raises():
    with pytest.raises(AssertionError):
        F.assert_no_leakage(["accommodates", "price"])
    with pytest.raises(AssertionError):
        F.assert_no_leakage(["review_scores_rating"])


def test_parsers():
    num, shared = F.parse_bathrooms(pd.Series(["1.5 baths", "Half-bath", "2 shared baths", None]))
    assert num.iloc[:3].tolist() == [1.5, 0.5, 2.0] and np.isnan(num.iloc[3])
    assert shared.iloc[2] == 1.0 and shared.iloc[0] == 0.0
    assert F.parse_amenities('["Wifi", "Kitchen"]') == ["wifi", "kitchen"]
    assert F.parse_amenities("['Wifi']") == ["wifi"]
    assert F.parse_amenities("__import__('os')") == []  # never eval()


def test_amenity_vocab_learned_from_fit_only():
    tr = pd.DataFrame({"a": [["wifi"], ["wifi", "tv"], ["wifi"]]})
    te = pd.DataFrame({"a": [["pool", "wifi"]]})
    enc = F.AmenityMultiHot(top_n=5, min_count=1).fit(tr)
    assert "amenity::pool" not in enc.get_feature_names_out()
    assert enc.transform(te).sum() == 1  # pool ignored, wifi kept


def test_distance_uses_nearest_anchor():
    d = F.distance_to_nearest_anchor(pd.Series([38.1157]), pd.Series([13.3615]), [[38.1157, 13.3615], [37.5, 15.0]])
    assert d[0] == pytest.approx(0, abs=1e-6)
