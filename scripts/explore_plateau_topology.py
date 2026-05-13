"""Probe: characterize the local optima that the bad-init runs get stuck in.

For each requested run directory, finds the top-K islands by final
``best_score``, then for each of them computes:

  * when the island's ``best_state`` last changed (the "freeze iter")
  * how many distinct best-states it ever held
  * the final graph's degree sequence, edge count, connectedness
  * pairwise Hamming distance between the final best-states of the
    top-K islands within the same run, to see whether they converged
    to the same topology or different ones

Renders:
  * ``plots/plateau_topology.png`` — top-K final graphs per run,
    one row per run, with captions.

Run:
    .venv/bin/python scripts/explore_plateau_topology.py \\
        runs/n19_main_plateau_v3 runs/n19_main_plateau_v4 runs/n19_main_plateau_v5 \\
        --top-k 3

Use ``--output`` to point at a different image file.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


@dataclass
class IslandSummary:
    island_id: int
    n_iters: int
    final_score: float
    final_lambda1: float
    final_mu: int
    final_state: tuple[int, ...]
    last_change_iter: int
    distinct_best_states: int


def adjacency_from_bits(bits: tuple[int, ...] | list[int] | np.ndarray, n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=int)
    iu = np.triu_indices(n, k=1)
    A[iu] = np.asarray(bits, dtype=int)
    return A + A.T


def summarize_island(path: Path) -> IslandSummary:
    last = None
    prev_state: tuple[int, ...] | None = None
    last_change_iter = 0
    distinct: set[tuple[int, ...]] = set()
    with path.open() as fh:
        for line in fh:
            j = json.loads(line)
            state = tuple(j["best_state"])
            distinct.add(state)
            if prev_state is None or state != prev_state:
                last_change_iter = int(j["iter"])
            prev_state = state
            last = j
    if last is None:
        raise RuntimeError(f"empty metrics file: {path}")
    return IslandSummary(
        island_id=int(last["island_id"]),
        n_iters=int(last["iter"]),
        final_score=float(last["best_score"]),
        final_lambda1=float(last["best_lambda1"]),
        final_mu=int(last["best_mu"]),
        final_state=tuple(last["best_state"]),
        last_change_iter=last_change_iter,
        distinct_best_states=len(distinct),
    )


def n_from_state(num_edges: int) -> int:
    # num_edges = n*(n-1)/2  →  n = (1 + sqrt(1 + 8*num_edges)) / 2
    return int(round((1 + (1 + 8 * num_edges) ** 0.5) / 2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="explore-plateau-topology", description=__doc__)
    p.add_argument("runs", nargs="+", type=Path, help="One or more run directories.")
    p.add_argument("--top-k", type=int, default=3,
                   help="Number of top islands (by final best_score) to plot per run "
                        "(ignored in --mode all-classes).")
    p.add_argument("--mode", choices=("top-k", "all-classes"), default="top-k",
                   help="top-k: render the top-K islands per run on one figure. "
                        "all-classes: render every distinct (non-isomorphic) tree "
                        "the run converged to, one figure per run.")
    p.add_argument("--output", type=Path, default=Path("plots/plateau_topology.png"),
                   help="Output image file (top-k mode). In all-classes mode this "
                        "is used as a stem and one image is written per run.")
    p.add_argument("--layout-seed", type=int, default=0,
                   help="Seed for the spring-layout that initializes kamada_kawai.")
    return p.parse_args(argv)


def hamming(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum(x != y for x, y in zip(a, b))


def color_for_degree(deg: int) -> str:
    if deg >= 3:
        return "#1f77b4"  # hub blue
    if deg == 2:
        return "#d62728"  # bridge red
    return "#bbbbbb"      # leaf grey


def render_graph(ax, bits, n: int, *, layout_seed: int, title: str) -> None:
    A = adjacency_from_bits(bits, n)
    G = nx.from_numpy_array(A)
    if G.number_of_edges() > 0:
        pos = nx.kamada_kawai_layout(G, pos=nx.spring_layout(G, seed=layout_seed))
    else:
        pos = nx.circular_layout(G)
    degrees = dict(G.degree())
    node_colors = [color_for_degree(degrees[v]) for v in G.nodes()]
    node_sizes = [180 if degrees[v] >= 3 else (130 if degrees[v] == 2 else 90)
                  for v in G.nodes()]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#666666", width=0.8)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, edgecolors="black", linewidths=0.5)
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()


def partition_by_isomorphism(islands: list[IslandSummary], n: int) -> list[dict]:
    """Group islands into isomorphism classes of their final trees.
    Each class is {rep_graph, rep_island (best-scoring member), members: [IslandSummary, ...]}.
    Classes are returned sorted by best score within the class, descending."""
    classes: list[dict] = []
    for s in islands:
        G = nx.from_numpy_array(adjacency_from_bits(s.final_state, n))
        placed = False
        for c in classes:
            if nx.is_isomorphic(c["rep_graph"], G):
                c["members"].append(s)
                if s.final_score > c["rep_island"].final_score:
                    c["rep_graph"] = G
                    c["rep_island"] = s
                placed = True
                break
        if not placed:
            classes.append({"rep_graph": G, "rep_island": s, "members": [s]})
    classes.sort(key=lambda c: -c["rep_island"].final_score)
    return classes


def render_all_classes(run_dir: Path, classes: list[dict], n: int,
                       output: Path, layout_seed: int) -> None:
    """One figure per run, showing every distinct (non-isomorphic) tree
    that this run converged to, with class multiplicity annotated."""
    K = len(classes)
    # Grid: roughly square, but cap columns at 4 so individual trees stay readable.
    n_cols = min(4, K)
    n_rows = (K + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.0 * n_cols, 3.4 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        if idx >= K:
            ax.set_axis_off()
            continue
        c = classes[idx]
        s = c["rep_island"]
        m = len(c["members"])
        islands = sorted(mem.island_id for mem in c["members"])
        islands_str = (", ".join(f"{i:02d}" for i in islands)
                       if m <= 16 else
                       f"{m} islands")
        edges = sum(s.final_state)
        title = (
            f"class {idx} (size {m}): score {s.final_score:+.3f}, "
            f"λ₁={s.final_lambda1:.3f}, μ={s.final_mu}\n"
            f"{edges} edges  |  islands: {islands_str}"
        )
        render_graph(ax, s.final_state, n=n, layout_seed=layout_seed, title=title)

    fig.suptitle(
        f"{run_dir.name}: {K} distinct (non-isomorphic) tree(s) "
        f"that this method converges to\n"
        f"(blue = degree ≥ 3, red = degree 2, grey = degree 1)",
        fontsize=12,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    runs: list[tuple[Path, list[IslandSummary]]] = []
    for run_dir in args.runs:
        islands = [summarize_island(f) for f in sorted(run_dir.glob("island_*_metrics.jsonl"))]
        if not islands:
            print(f"no island metrics in {run_dir}", file=sys.stderr)
            return 1
        islands.sort(key=lambda s: -s.final_score)
        runs.append((run_dir, islands))

    n = n_from_state(len(runs[0][1][0].final_state))

    if args.mode == "all-classes":
        for run_dir, islands in runs:
            classes = partition_by_isomorphism(islands, n)
            print(f"\n=== {run_dir.name}: {len(classes)} isomorphism class(es) "
                  f"from {len(islands)} islands ===")
            for ci, c in enumerate(classes):
                s = c["rep_island"]
                ids = sorted(m.island_id for m in c["members"])
                print(f"  class {ci}  size={len(c['members']):2d}  "
                      f"score={s.final_score:+.4f}  λ₁={s.final_lambda1:.4f}  μ={s.final_mu}"
                      f"  islands={ids}")
            stem = args.output.stem
            out = args.output.parent / f"{stem}_{run_dir.name}.png"
            render_all_classes(run_dir, classes, n, out, args.layout_seed)
        return 0

    # ---- text summary -------------------------------------------------
    print(f"\nn = {n}, top-{args.top_k} islands per run\n")
    for run_dir, islands in runs:
        print(f"=== {run_dir.name} ===")
        print(f"  {'rank':>4} {'island':>6} {'iter':>6} {'freeze':>6} "
              f"{'%life':>6} {'score':>8} {'lam1':>6} {'mu':>3} {'edges':>5} "
              f"{'distinct':>8}")
        top = islands[: args.top_k]
        for i, s in enumerate(top):
            life = s.last_change_iter / s.n_iters
            edges = sum(s.final_state)
            print(f"  {i:>4} {s.island_id:>6} {s.n_iters:>6} {s.last_change_iter:>6} "
                  f"{life:>6.0%} {s.final_score:>8.4f} {s.final_lambda1:>6.4f} "
                  f"{s.final_mu:>3} {edges:>5} {s.distinct_best_states:>8}")
        # pairwise Hamming among top-K
        if len(top) >= 2:
            print(f"  pairwise Hamming (edge differences) among top-{args.top_k}:")
            for a in range(len(top)):
                for b in range(a + 1, len(top)):
                    h = hamming(top[a].final_state, top[b].final_state)
                    print(f"    island {top[a].island_id:>2} vs island {top[b].island_id:>2}: "
                          f"{h} edges differ")
        print()

    # ---- figure -------------------------------------------------------
    n_rows = len(runs)
    n_cols = args.top_k
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows),
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row, (run_dir, islands) in enumerate(runs):
        top = islands[: n_cols]
        # row label on the leftmost axis
        for col, s in enumerate(top):
            ax = axes[row, col]
            life = s.last_change_iter / s.n_iters
            edges = sum(s.final_state)
            title = (
                f"island {s.island_id:02d} "
                f"(score {s.final_score:+.3f}, λ₁={s.final_lambda1:.3f}, μ={s.final_mu})\n"
                f"froze at iter {s.last_change_iter}/{s.n_iters} ({life:.0%} of run)  "
                f"|  {edges} edges  |  {s.distinct_best_states} distinct best-so-fars"
            )
            render_graph(ax, s.final_state, n=n, layout_seed=args.layout_seed, title=title)

        axes[row, 0].text(
            -0.08, 0.5, run_dir.name,
            transform=axes[row, 0].transAxes,
            rotation=90, ha="center", va="center",
            fontsize=11, fontweight="bold",
        )

    fig.suptitle(
        f"Local optima reached by the top-{n_cols} islands per bad-init run\n"
        f"(blue = degree ≥ 3, red = degree 2, grey = degree 1)",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
