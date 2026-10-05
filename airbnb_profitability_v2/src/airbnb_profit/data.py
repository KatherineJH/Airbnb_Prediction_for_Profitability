"""Raw data access: schema check, integrity hash."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "id", "host_id", "city", "latitude", "longitude", "property_type", "room_type",
    "accommodates", "bathrooms_text", "bedrooms", "beds", "amenities", "price",
    "minimum_nights", "maximum_nights", "number_of_reviews_ltm", "host_since",
    "last_scraped", "host_location", "host_response_time", "host_response_rate",
    "host_acceptance_rate", "host_is_superhost", "host_total_listings_count",
    "host_about", "neighborhood_overview", "license", "instant_bookable",
    "host_verifications", "neighbourhood_cleansed", "host_identity_verified",
    "host_has_profile_pic",
]

DOWNLOAD_HELP = (
    "Download 'merged_file.csv' (link in the original README / notebook 1) and place it at "
    "data/raw/merged_file.csv, or run `python scripts/download_data.py`. "
    "To try the workflow without real data use `--set data.source=synthetic`."
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_raw(path: Path, expected_sha256: str | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {DOWNLOAD_HELP}")
    if expected_sha256 and sha256_file(path) != expected_sha256:
        raise ValueError(f"SHA-256 of {path.name} does not match config data.expected_sha256")
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"Raw file is missing required columns: {missing}")
    return pd.read_csv(path, usecols=REQUIRED_COLUMNS, low_memory=False)
