"""Command line entry point:  python -m airbnb_profit.cli run --help"""
from __future__ import annotations

import argparse
import logging
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless runs; the notebook sets its own backend

from .config import load_config  # noqa: E402
from .pipeline import STAGES, run_all  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="airbnb_profit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the workflow stages (cached, resumable)")
    r.add_argument("--config", default=None, help="YAML config (default: config/default.yaml)")
    r.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    r.add_argument("--cities", nargs="+", default=None, help="subset of cities")
    r.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override config, e.g. --set tuning.n_trials=5 (change project.run_name too)")
    r.add_argument("--synthetic", action="store_true", help="shortcut for --set data.source=synthetic")
    r.add_argument("--force", action="store_true", help="recompute even if cached outputs exist")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    overrides = list(args.overrides) + (["data.source=synthetic"] if args.synthetic else [])
    cfg = load_config(args.config, overrides)
    run = run_all(cfg, args.stages, args.cities, args.force)
    logging.info("done -> %s", run.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
