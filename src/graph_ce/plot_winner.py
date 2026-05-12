"""CLI tool: render a winner-graph from a run's saved adjacency matrix.

Reads ``runs/<name>/winner_adjacency.npy`` and ``winner_certificate.json``
and produces a labelled figure suitable for the README / a slide / a paper.

Vertices are colored by structural role: high-degree hubs in one color,
the matching's "bridge"-style cut vertices in another, leaves in a third.

Usage:
    graph-ce-plot-winner runs/n19_success --output plots/winner_8_10.png
    graph-ce-plot-winner runs/n19_success_with_migration --output plots/winner_9_9.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# Colour scheme: hubs (degree >= 3) stand out; cut/bridge vertices (deg 2)
# are the connectors; leaves (deg 1) are the tail.
_HUB_COLOR = "#1f77b4"     # blue
_BRIDGE_COLOR = "#d62728"  # red
_LEAF_COLOR = "#bbbbbb"    # neutral grey


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="graph-ce-plot-winner", description=__doc__)
    p.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing winner_adjacency.npy and winner_certificate.json.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <run_dir>/winner_graph.png).",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title override. By default the title is derived from "
             "the certificate (n, lambda_1, mu, score).",
    )
    p.add_argument(
        "--figsize",
        type=str,
        default="9,7",
        help="Figure size 'W,H' in inches (default: 9,7).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Layout seed (default: 42). Kamada-Kawai is deterministic given "
             "the initial spring-layout, which uses this seed.",
    )
    return p.parse_args(argv)


def _color_for(degree: int) -> str:
    if degree >= 3:
        return _HUB_COLOR
    if degree == 2:
        return _BRIDGE_COLOR
    return _LEAF_COLOR


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir

    adj_path = run_dir / "winner_adjacency.npy"
    cert_path = run_dir / "winner_certificate.json"
    if not adj_path.exists() or not cert_path.exists():
        print(f"missing winner artifacts in {run_dir}", file=sys.stderr)
        return 1

    A = np.load(adj_path)
    cert = json.loads(cert_path.read_text())

    n = int(cert["n"])
    G = nx.from_numpy_array(A)

    # Kamada-Kawai gives clean, deterministic layouts for sparse graphs;
    # it's also what Adam Wagner uses in his reference code.
    initial = nx.spring_layout(G, seed=args.seed)
    pos = nx.kamada_kawai_layout(G, pos=initial)

    degrees = dict(G.degree())
    node_colors = [_color_for(degrees[v]) for v in G.nodes()]
    node_sizes = [500 if degrees[v] >= 3 else (350 if degrees[v] == 2 else 220)
                  for v in G.nodes()]

    figsize = tuple(float(x) for x in args.figsize.split(","))
    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#666666", width=1.2)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=0.8,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_color="white")

    if args.title:
        title = args.title
    else:
        title = (
            f"Counterexample to Conjecture 2.1 at n={n}  |  "
            f"λ₁={cert['lambda_1']:.4f}, μ={cert['mu']}, "
            f"score={cert['score']:+.4f}"
        )
    ax.set_title(title, fontsize=11)

    # Legend for the colour code.
    hub_patch = plt.Line2D([], [], marker="o", linestyle="",
                           markerfacecolor=_HUB_COLOR, markeredgecolor="black",
                           markersize=10, label="hub (deg ≥ 3)")
    bridge_patch = plt.Line2D([], [], marker="o", linestyle="",
                              markerfacecolor=_BRIDGE_COLOR, markeredgecolor="black",
                              markersize=10, label="bridge (deg = 2)")
    leaf_patch = plt.Line2D([], [], marker="o", linestyle="",
                            markerfacecolor=_LEAF_COLOR, markeredgecolor="black",
                            markersize=8, label="leaf (deg = 1)")
    ax.legend(handles=[hub_patch, bridge_patch, leaf_patch], loc="lower right",
              framealpha=0.92, fontsize=9)
    ax.set_axis_off()

    output = args.output or (run_dir / "winner_graph.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=140, bbox_inches="tight")
    print(
        f"saved {output}  "
        f"(n={n}, |E|={G.number_of_edges()}, "
        f"degrees: hub={sum(1 for d in degrees.values() if d >= 3)}, "
        f"bridge={sum(1 for d in degrees.values() if d == 2)}, "
        f"leaf={sum(1 for d in degrees.values() if d == 1)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
