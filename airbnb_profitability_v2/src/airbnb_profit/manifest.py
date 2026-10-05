"""Run manifest: everything needed to know exactly what produced a result."""
from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
from datetime import datetime, timezone

from .config import PROJECT_ROOT, config_hash

_PKGS = ["numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "xgboost", "optuna", "pyarrow", "PyYAML"]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def build_manifest(cfg: dict, data_fingerprint: str) -> dict:
    versions = {}
    for p in _PKGS:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": cfg["project"]["run_name"],
        "config_hash": config_hash(cfg),
        "seed": cfg["project"]["seed"],
        "data_fingerprint": data_fingerprint,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }
