"""CLI tool: summarize a graph_ce run from its on-disk artifacts.

Usage:
    graph-ce-status                       # defaults to runs/n19_main
    graph-ce-status --run-dir runs/foo
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="graph-ce-status", description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/n19_main"),
        help="Path to a run directory (default: runs/n19_main).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir: Path = args.run_dir
    if not run_dir.exists():
        print(f"run dir does not exist: {run_dir}", file=sys.stderr)
        return 1

    metrics = sorted(glob.glob(str(run_dir / "island_*_metrics.jsonl")))
    per_island = []
    global_best = -math.inf
    global_best_meta = None

    for path in metrics:
        iid = int(Path(path).stem.split("_")[1])
        n = 0
        best = -math.inf
        last = None
        step_sum = 0.0
        step_n = 0
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                n = row["iter"]
                last = row
                bs = row.get("best_score")
                if bs is not None and bs > best:
                    best = bs
                if bs is not None and bs > global_best:
                    global_best = bs
                    global_best_meta = {"island": iid, **row}
                ss = row.get("step_seconds")
                if ss is not None:
                    step_sum += ss
                    step_n += 1
        avg = step_sum / step_n if step_n else float("nan")
        per_island.append({"iid": iid, "n": n, "best": best, "last": last, "avg_step": avg})

    if not per_island:
        print(f"no island metrics files in {run_dir}")
        return 0

    print(f"run_dir: {run_dir}")
    print()
    print(f"{'island':>6} {'iters':>6} {'best':>10} {'avg_step_s':>11} {'last_step_s':>12} {'wall_s':>8}")
    for d in per_island:
        last = d["last"]
        wall = last.get("wall_seconds", 0.0) if last else 0.0
        last_step = last.get("step_seconds", float("nan")) if last else float("nan")
        best = d["best"] if math.isfinite(d["best"]) else float("nan")
        print(f"{d['iid']:>6} {d['n']:>6} {best:>10.4f} {d['avg_step']:>11.3f} {last_step:>12.3f} {wall:>8.1f}")

    print()
    if math.isfinite(global_best):
        gap = 0.0 - global_best
        print(f"Global best: {global_best:.6f}  (gap to victory > 0: {gap:.4f})")
        if global_best_meta:
            m = global_best_meta
            print(
                f"  island={m['island']} iter={m['iter']} "
                f"lambda_1={m['best_lambda1']:.4f} mu={m['best_mu']}"
            )
    else:
        print("Global best: not yet observed")

    coord_log = run_dir / "coordinator.log"
    if coord_log.exists():
        lines = coord_log.read_text().splitlines()
        crit = [l for l in lines if "CRITICAL" in l]
        bc = [l for l in lines if "broadcast |" in l]
        print()
        print(f"CRITICAL lines in coordinator.log: {len(crit)}")
        print(f"Migration broadcasts: {len(bc)}")
        if bc:
            tail = bc[-1].split("INFO coordinator: ", 1)[-1]
            print(f"  latest: {tail}")
        if crit:
            print("  most recent CRITICAL:")
            for l in crit[-3:]:
                print(f"   {l}")

    summary = run_dir / "summary.json"
    if summary.exists():
        print()
        print(f"summary.json present (run finished):")
        s = json.loads(summary.read_text())
        print(f"  status={s.get('status')} duration_seconds={s.get('duration_seconds')}")
        if s.get("winner"):
            print(f"  winner: {s['winner']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
