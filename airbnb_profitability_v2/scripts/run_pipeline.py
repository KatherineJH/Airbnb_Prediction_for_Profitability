"""Convenience launcher that works without installing the package:  python scripts/run_pipeline.py --help"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from airbnb_profit.cli import main  # noqa: E402

raise SystemExit(main(["run", *sys.argv[1:]]))
