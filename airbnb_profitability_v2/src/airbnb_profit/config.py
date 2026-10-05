"""Config loading, CLI overrides and hashing."""
from __future__ import annotations

import copy
import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML config and apply ``a.b.c=value`` overrides (values parsed as YAML)."""
    with open(path or DEFAULT_CONFIG, encoding="utf8") as fh:
        cfg = yaml.safe_load(fh)
    for item in overrides or []:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key: {key}")
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Hash of everything that influences results (``paths`` excluded)."""
    body = {k: v for k, v in cfg.items() if k != "paths"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:12]


def derive_seed(base: int, *keys: object) -> int:
    """Stable per-(city, model, ...) seed; independent of Python's randomised ``hash``."""
    return (base + zlib.crc32("|".join(map(str, keys)).encode())) % (2**31 - 1)


def resolve(cfg: dict[str, Any], key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p


def with_overrides(cfg: dict[str, Any], **flat: Any) -> dict[str, Any]:
    """Deep-copy helper for notebooks/tests: ``with_overrides(cfg, **{"tuning.n_trials": 3})``."""
    out = copy.deepcopy(cfg)
    for key, value in flat.items():
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return out
