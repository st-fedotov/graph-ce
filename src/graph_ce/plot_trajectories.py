"""CLI tool: plot per-island best-score trajectories across multiple runs.

For each configured run, picks the top-K islands by final best-score and
overlays their trajectories. Runs are colored, islands within a run share
the run's color and linestyle (just as the user requested — same-style
sibling trajectories make seed variance visible at a glance).

Usage:
    graph-ce-plot --output plots/setup_comparison.png

By default the tool plots the six runs from the ablation sequence
described in the README; the SETUPS dict below documents the mapping.
Override the run selection with --setups "label=run_dir,label=run_dir,..."
if needed.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Default ablation sequence: label (for legend) -> run directory.
# Order chosen so legend reads "from worst setup to best setup".
SETUPS: dict[str, str] = {
    "PyTorch init, b=512, migration on":  "runs/n19_main_plateau_v3",
    "PyTorch init, b=512, no migration":  "runs/n19_main_plateau_v4",
    "PyTorch init, b=32, no migration":   "runs/n19_main_plateau_v5",
    "Keras init, b=512, no migration":    "runs/n19_b512_ablation",
    "Keras init, b=32, no migration":     "runs/n19_success",
    "Keras init, b=32, migration on":     "runs/n19_success_with_migration",
}


@dataclass
class IslandTrajectory:
    island_id: int
    iters: list[int]
    best_scores: list[float]

    @property
    def final_best(self) -> float:
        # Return the max best_score observed (best_score per row already is
        # an overall-pool max, so the last row's best_score is the global best
        # for this island).
        if not self.best_scores:
            return -math.inf
        return max(self.best_scores)


def load_run_trajectories(run_dir: Path) -> list[IslandTrajectory]:
    paths = sorted(glob.glob(str(run_dir / "island_*_metrics.jsonl")))
    out: list[IslandTrajectory] = []
    for p in paths:
        iid = int(Path(p).stem.split("_")[1])
        iters: list[int] = []
        scores: list[float] = []
        with open(p) as f:
            for line in f:
                row = json.loads(line)
                bs = row.get("best_score")
                if bs is None:
                    continue
                iters.append(int(row["iter"]))
                scores.append(float(bs))
        if iters:
            out.append(IslandTrajectory(island_id=iid, iters=iters, best_scores=scores))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="graph-ce-plot", description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("plots/setup_comparison.png"),
        help="Output image path (default: plots/setup_comparison.png).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top-performing islands to plot per setup (default: 3).",
    )
    p.add_argument(
        "--setups",
        type=str,
        default=None,
        help="Optional override of the default ablation set, formatted "
             "as 'label=path,label=path,...'. Quote the whole arg.",
    )
    p.add_argument(
        "--figsize",
        type=str,
        default="14,9",
        help="Figure size 'W,H' in inches (default: 14,9).",
    )
    p.add_argument(
        "--xlim",
        type=str,
        default="0,10000",
        help="X-axis limit 'lo,hi' (default: 0,10000 to keep early-iter dynamics "
             "readable; pass 'auto' to scale to the full data range).",
    )
    p.add_argument(
        "--ylim",
        type=str,
        default=None,
        help="Optional y-axis limit 'lo,hi' (default: auto from data).",
    )
    return p.parse_args(argv)


def parse_setups(s: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for chunk in s.split(","):
        if "=" not in chunk:
            raise ValueError(f"bad setup chunk: {chunk!r}")
        label, _, path = chunk.partition("=")
        out[label.strip()] = path.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setups = parse_setups(args.setups) if args.setups else SETUPS

    figsize = tuple(float(x) for x in args.figsize.split(","))
    fig, ax = plt.subplots(figsize=figsize)

    # Distinct colors across setups. tab10 gives 10 visually-distinct colors;
    # we only use the first len(setups).
    colors = plt.get_cmap("tab10")(range(len(setups)))

    # Track plotted iterations for sensible default x-axis behaviour.
    max_iter = 0
    min_score = math.inf
    max_score = -math.inf

    for color, (label, rd) in zip(colors, setups.items()):
        run_dir = Path(rd)
        if not run_dir.exists():
            print(f"skipping missing run dir: {run_dir}", file=sys.stderr)
            continue
        trajs = load_run_trajectories(run_dir)
        if not trajs:
            print(f"no trajectories in {run_dir}", file=sys.stderr)
            continue
        # Pick top-K by final best.
        trajs.sort(key=lambda t: t.final_best, reverse=True)
        top = trajs[: args.top_k]
        for i, t in enumerate(top):
            ax.plot(
                t.iters,
                t.best_scores,
                color=color,
                linewidth=1.4,
                alpha=0.85,
                # Only label the first trajectory of each setup so the legend
                # stays compact.
                label=label if i == 0 else None,
            )
            max_iter = max(max_iter, max(t.iters))
            min_score = min(min_score, min(t.best_scores))
            max_score = max(max_score, max(t.best_scores))

    # Victory threshold line.
    ax.axhline(
        0.0,
        color="black",
        linestyle=":",
        linewidth=1.0,
        alpha=0.6,
        label="victory threshold (score = 0)",
    )

    ax.set_xlabel("CEM iteration")
    ax.set_ylabel("best score this iter (sqrt(n-1) + 1 - lambda_1 - mu)")
    ax.set_title(
        f"Top-{args.top_k} island trajectories per setup, Conjecture 2.1 at n=19"
    )
    ax.grid(True, alpha=0.3)
    if args.xlim and args.xlim.lower() != "auto":
        lo, hi = (float(x) for x in args.xlim.split(","))
        ax.set_xlim(lo, hi)
    if args.ylim:
        lo, hi = (float(x) for x in args.ylim.split(","))
        ax.set_ylim(lo, hi)

    # Compact legend, single column, outside the plot area.
    ax.legend(loc="lower right", framealpha=0.92, fontsize=9)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=140)
    print(f"saved {args.output}  (iter range 0..{max_iter}, score range {min_score:.3f}..{max_score:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
