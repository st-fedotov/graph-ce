"""CLI entry point.

Usage:
    python -m graph_ce.run --config configs/default.yaml
    python -m graph_ce.run --config configs/default.yaml \
        --override parallelism.n_islands=4 --override problem.n=12

Creates ``<output_dir>/<timestamp>/`` and hands off to the coordinator.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

from .config import Config, apply_overrides
from .coordinator import run_coordinator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="graph-ce", description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "default.yaml",
        help="Path to YAML config (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value (dot-path). Repeatable. "
        "Example: --override parallelism.n_islands=4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override logging.output_dir from the config.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Subdirectory name inside output_dir (default: UTC timestamp).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict = apply_overrides(cfg_dict, args.override)
    config = Config.model_validate(cfg_dict)

    output_dir = args.output_dir or Path(config.logging.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    with open(run_dir / "config.yaml", "w") as f:
        f.write(config.dump_yaml())

    return run_coordinator(config, run_dir)


if __name__ == "__main__":
    sys.exit(main())
