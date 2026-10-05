"""Synthetic listings that mimic the *raw string formats* of Inside Airbnb (for tests / demos).

The revenue signal is planted (size, room type, centrality, host professionalism) so that a
sound pipeline should recover R^2 clearly above zero while the baseline stays at ~0.
These numbers say nothing about the real Italian market.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import derive_seed

_AMEN = ["Wifi", "Kitchen", "Washer", "Air conditioning", "Heating", "TV", "Hair dryer", "Iron",
         "Dishwasher", "Pool", "Free parking on premises", "Balcony", "Coffee maker", "Elevator",
         "Hot water", "Essentials", "Hangers", "Dedicated workspace", "Crib", "Self check-in"]
_PROP = ["Entire rental unit", "Private room in rental unit", "Entire home", "Room in hotel",
         "Entire condo", "Private room in home", "Entire villa", "Shared room in hostel"]


def make_synthetic_raw(cfg: dict, cities: list[str] | None = None) -> pd.DataFrame:
    cities = cities or cfg["data"]["cities"]
    n_city = int(cfg["data"]["synthetic_rows_per_city"])
    frames = []
    for ci, city in enumerate(cities):
        rng = np.random.default_rng(derive_seed(cfg["project"]["seed"], "synthetic", city))
        n = n_city
        anchor = np.array(cfg["geo"]["anchors"][city][0])
        lat = anchor[0] + rng.normal(0, 0.05, n)
        lon = anchor[1] + rng.normal(0, 0.07, n)
        dist = np.hypot(lat - anchor[0], (lon - anchor[1]) * 0.75) * 111
        n_hosts = max(20, n // 4)
        host_id = rng.integers(1, n_hosts + 1, n) + ci * 10_000
        host_size = rng.integers(1, 40, n_hosts + 1)  # listings per host
        host_total = host_size[host_id - ci * 10_000]
        room = rng.choice(["Entire home/apt", "Private room", "Hotel room", "Shared room"], n, p=[.7, .22, .05, .03])
        acc = np.clip(rng.poisson(3, n) + 1, 1, 16)
        base_price = 35 + 14 * acc + np.where(room == "Entire home/apt", 40, 0) - 1.5 * dist + rng.normal(0, 15, n)
        price = np.clip(base_price, 12, None).round()
        n_amen = rng.integers(8, 20, n)
        amen = [json.dumps(list(rng.choice(_AMEN, k, replace=False))) for k in n_amen]
        demand = np.exp(0.9 + 0.12 * acc - 0.03 * dist + 0.01 * np.log1p(host_total) * 10 + rng.normal(0, .6, n))
        reviews_ltm = rng.poisson(np.clip(demand, 0.2, 80))
        tenure = rng.uniform(0.5, 12, n)
        scraped = pd.Timestamp("2022-09-15")
        beds = np.clip((acc / 2).round(), 1, None)
        baths = rng.choice(["1 bath", "1.5 baths", "2 baths", "1 shared bath", "Half-bath", None], n, p=[.5, .1, .15, .1, .05, .1])
        frames.append(pd.DataFrame({
            "id": np.arange(n) + ci * 1_000_000,
            "host_id": host_id,
            "city": city,
            "latitude": lat, "longitude": lon,
            "property_type": rng.choice(_PROP, n),
            "room_type": room,
            "accommodates": acc,
            "bathrooms_text": baths,
            "bedrooms": np.where(rng.random(n) < .05, np.nan, np.clip((acc / 2.5).round(), 1, None)),
            "beds": beds,
            "amenities": amen,
            "price": [f"${p:,.2f}" for p in price],
            "minimum_nights": rng.choice([1, 2, 3, 7, 30], n, p=[.45, .25, .15, .1, .05]),
            "maximum_nights": rng.choice([30, 365, 1125], n),
            "number_of_reviews_ltm": reviews_ltm,
            "host_since": (scraped - pd.to_timedelta((tenure * 365).astype(int), unit="D")).strftime("%Y-%m-%d"),
            "last_scraped": scraped.strftime("%Y-%m-%d"),
            "host_location": rng.choice([f"{city.title()}, Italy", "London, United Kingdom", None], n, p=[.8, .1, .1]),
            "host_response_time": rng.choice(["within an hour", "within a few hours", "within a day", None], n),
            "host_response_rate": [f"{x}%" if x == x else np.nan for x in np.where(rng.random(n) < .1, np.nan, rng.integers(50, 101, n))],
            "host_acceptance_rate": [f"{x}%" if x == x else np.nan for x in np.where(rng.random(n) < .1, np.nan, rng.integers(30, 101, n))],
            "host_is_superhost": rng.choice(["t", "f", None], n, p=[.2, .75, .05]),
            "host_total_listings_count": host_total,
            "host_about": rng.choice(["Hi!", None], n),
            "neighborhood_overview": rng.choice(["Nice area", None], n),
            "license": rng.choice(["IT0123", None], n, p=[.3, .7]),
            "instant_bookable": rng.choice(["t", "f"], n),
            "host_verifications": rng.choice(["['email', 'phone']", "['email', 'phone', 'work_email']", "['phone']", "[]"], n),
            "neighbourhood_cleansed": rng.choice([f"zone_{i}" for i in range(12)], n),
            "host_identity_verified": rng.choice(["t", "f"], n),
            "host_has_profile_pic": rng.choice(["t", "f"], n, p=[.95, .05]),
        }))
    return pd.concat(frames, ignore_index=True)
