"""Probe: what kind of graphs do the two init schemes actually produce
before any training? Samples N graphs from a freshly-initialized PolicyMLP
under each init mode and compares.

Run:
    .venv/bin/python scripts/compare_initializations.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from graph_ce.model import PolicyMLP
from graph_ce.sampler import sample_batch


N = 19
NUM_EDGES = N * (N - 1) // 2  # 171
HIDDEN_SIZES = [128, 64, 4]
N_SESSIONS = 500          # graphs sampled per (init, seed) combination
N_SEEDS = 4               # number of model seeds per init (for histograms)
N_EXAMPLES = 3            # example graphs to draw per (init, seed)
N_HEATMAP_SEEDS = 8       # seeds per init for the per-position heatmap


def sample_graphs_for_init(init: str, seed: int, n_sessions: int) -> np.ndarray:
    seed = int(seed)  # numpy integers don't satisfy torch.manual_seed's typing
    torch.manual_seed(seed)
    model = PolicyMLP(num_edges=NUM_EDGES, hidden_sizes=HIDDEN_SIZES, init=init)
    gen = torch.Generator()
    gen.manual_seed(seed + 1_000_000)
    return sample_batch(model, n_sessions, NUM_EDGES, generator=gen)


def edge_counts(states: np.ndarray) -> np.ndarray:
    return states.sum(axis=1)


def example_adjacency(bit_string: np.ndarray) -> np.ndarray:
    A = np.zeros((N, N), dtype=int)
    iu = np.triu_indices(N, k=1)
    A[iu] = bit_string
    return A + A.T


def measure_position_bias(init: str, seed: int) -> np.ndarray:
    """For each of the 171 positions, return P(bit=1) when the model is fed
    the all-zero state and that position's one-hot. This is the policy's
    "default" preference at each position, independent of autoregressive
    context. With ``keras`` init the values should hover around 0.5 for
    every position; with ``pytorch_default``, the non-zero biases shift
    different positions in different directions."""
    torch.manual_seed(seed)
    model = PolicyMLP(num_edges=NUM_EDGES, hidden_sizes=HIDDEN_SIZES, init=init)
    model.eval()
    state = torch.zeros(NUM_EDGES, NUM_EDGES, dtype=torch.float32)
    pos_eye = torch.eye(NUM_EDGES, dtype=torch.float32)
    x = torch.cat([state, pos_eye], dim=1)
    with torch.no_grad():
        probs = model.prob(x).squeeze(-1).numpy()
    return probs


def plot_position_heatmap(per_init_per_seed: dict[str, np.ndarray]) -> None:
    """per_init_per_seed[init] has shape (n_seeds, num_edges)."""
    init_modes = list(per_init_per_seed.keys())

    fig, axes = plt.subplots(
        len(init_modes) + 1, 1,
        figsize=(14, 11),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1] * len(init_modes) + [0.9]},
    )

    cmap = "RdBu_r"
    vmin, vmax = 0.0, 1.0

    for ax, init in zip(axes[: len(init_modes)], init_modes):
        data = per_init_per_seed[init]
        im = ax.imshow(
            data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"{init}", fontsize=12, loc="left", pad=8)
        ax.set_ylabel("seed")
        ax.set_yticks(range(data.shape[0]))
        # Only the bottom heatmap shows an x-label; top one borrows the
        # axis context from below.
        if ax is axes[len(init_modes) - 1]:
            ax.set_xlabel("position index (0..170, upper-triangle edge ordering)")

    # Bottom axis: |P - 0.5| summary per init.
    ax_bias = axes[-1]
    for offset, init in enumerate(init_modes):
        data = per_init_per_seed[init]
        bias = np.abs(data - 0.5)
        per_seed_mean = bias.mean(axis=1)
        per_seed_max = bias.max(axis=1)
        x = np.arange(data.shape[0]) + offset * 0.35 - 0.175
        ax_bias.bar(
            x, per_seed_mean, width=0.3,
            label=f"{init}: mean |P − 0.5|",
            color={"keras": "#1f77b4", "pytorch_default": "#d62728"}[init],
            alpha=0.75,
        )
        ax_bias.errorbar(
            x, per_seed_mean,
            yerr=[np.zeros_like(per_seed_mean), per_seed_max - per_seed_mean],
            fmt="none", ecolor="black", capsize=3, linewidth=0.8,
            label="max |P − 0.5| (whisker)" if offset == 0 else None,
        )
    ax_bias.set_ylabel("|P(bit=1) − 0.5|")
    ax_bias.set_xlabel("seed")
    ax_bias.set_title("Per-seed deviation from uniform Bernoulli(0.5)",
                      fontsize=12, loc="left", pad=8)
    ax_bias.legend(loc="upper right", fontsize=9)
    ax_bias.grid(alpha=0.3, axis="y")
    ax_bias.set_xticks(range(per_init_per_seed[init_modes[0]].shape[0]))

    fig.suptitle(
        "P(bit = 1) at each position with all-zero state, across 8 model seeds.\n"
        "Diverging colormap centred at 0.5: deep blue ≈ never, deep red ≈ always.",
        fontsize=11, y=1.02,
    )
    # Shared colorbar to the right of the two heatmap rows.
    fig.colorbar(im, ax=axes[: len(init_modes)].tolist(),
                 shrink=0.85, label="P(bit=1)", pad=0.015)

    out = Path("plots/init_position_bias.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


def main() -> None:
    init_modes = ["keras", "pytorch_default"]

    all_counts: dict[str, np.ndarray] = {}
    per_seed_counts: dict[str, list[np.ndarray]] = {init: [] for init in init_modes}
    examples: dict[str, list[np.ndarray]] = {init: [] for init in init_modes}

    rng_example = np.random.default_rng(0)
    for init in init_modes:
        all_graphs: list[np.ndarray] = []
        for s in range(N_SEEDS):
            graphs = sample_graphs_for_init(init, seed=42 + s, n_sessions=N_SESSIONS)
            all_graphs.append(graphs)
            per_seed_counts[init].append(edge_counts(graphs))
            chosen = rng_example.choice(graphs.shape[0], size=N_EXAMPLES, replace=False)
            for idx in chosen:
                examples[init].append(graphs[idx])
        all_counts[init] = np.concatenate(all_graphs, axis=0).sum(axis=1)

    # -- summary stats ----------------------------------------------------
    print(f"{'init':<18} {'mean_edges':>11} {'std_edges':>10} {'min':>5} {'max':>5}  per-seed means")
    for init in init_modes:
        c = all_counts[init]
        means_by_seed = [float(np.mean(arr)) for arr in per_seed_counts[init]]
        means_str = ", ".join(f"{m:5.1f}" for m in means_by_seed)
        print(
            f"{init:<18} {c.mean():>11.1f} {c.std():>10.1f} "
            f"{int(c.min()):>5} {int(c.max()):>5}  [{means_str}]"
        )
    print(f"\n(unbiased Bernoulli(0.5) baseline: mean = {NUM_EDGES / 2:.1f}, "
          f"std = sqrt({NUM_EDGES}/4) ≈ {math.sqrt(NUM_EDGES / 4):.2f})")

    # -- plot -------------------------------------------------------------
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1 + N_EXAMPLES, height_ratios=[1, 1],
                          width_ratios=[1.4] + [1] * N_EXAMPLES)

    # Top-left and bottom-left share the histogram column.
    ax_hist = fig.add_subplot(gs[:, 0])

    colors = {"keras": "#1f77b4", "pytorch_default": "#d62728"}
    for init in init_modes:
        ax_hist.hist(
            all_counts[init],
            bins=40,
            alpha=0.55,
            color=colors[init],
            label=f"{init} (mean={all_counts[init].mean():.1f}, "
                  f"std={all_counts[init].std():.1f})",
        )
    ax_hist.axvline(NUM_EDGES / 2, color="black", linestyle=":", linewidth=1,
                    label=f"unbiased = {NUM_EDGES / 2:.1f}")
    ax_hist.set_xlabel(f"edges per sampled graph (n_sessions={N_SESSIONS} × "
                       f"{N_SEEDS} model seeds = {N_SESSIONS * N_SEEDS} graphs)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Edge-count distribution before any training")
    ax_hist.legend(loc="upper right", fontsize=8)
    ax_hist.grid(alpha=0.3)

    # Top row: keras examples. Bottom row: pytorch_default examples.
    for col in range(N_EXAMPLES):
        for row, init in enumerate(init_modes):
            ax = fig.add_subplot(gs[row, 1 + col])
            bits = examples[init][col]  # first example from seed 0
            A = example_adjacency(bits)
            G = nx.from_numpy_array(A)
            if G.number_of_edges() > 0:
                pos = nx.kamada_kawai_layout(G, pos=nx.spring_layout(G, seed=col))
            else:
                pos = nx.circular_layout(G)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#666", width=0.6)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors[init],
                                   node_size=80, edgecolors="black",
                                   linewidths=0.5)
            ax.set_title(
                f"{init}: {int(bits.sum())} edges, "
                f"{'connected' if nx.is_connected(G) else 'DISCONNECTED'}",
                fontsize=9,
            )
            ax.set_axis_off()

    fig.suptitle(
        f"Initial sampling behaviour, n={N} (no training, no CEM updates)",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path("plots/init_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")

    # -- per-position heatmap --------------------------------------------
    per_init_per_seed: dict[str, np.ndarray] = {}
    for init in init_modes:
        rows = []
        for s in range(N_HEATMAP_SEEDS):
            rows.append(measure_position_bias(init, seed=42 + s))
        per_init_per_seed[init] = np.stack(rows)

    print("\n== per-position bias |P(bit=1) - 0.5| ==")
    print(f"{'init':<18} {'mean':>8} {'std':>8} {'max':>8}  per-seed max")
    for init in init_modes:
        bias = np.abs(per_init_per_seed[init] - 0.5)
        per_seed_max = bias.max(axis=1)
        per_seed_max_str = ", ".join(f"{m:.3f}" for m in per_seed_max)
        print(
            f"{init:<18} {bias.mean():>8.4f} {bias.std():>8.4f} {bias.max():>8.4f}"
            f"  [{per_seed_max_str}]"
        )

    plot_position_heatmap(per_init_per_seed)
    plot_init_examples(per_init_per_seed)


def plot_init_examples(per_init_per_seed: dict[str, np.ndarray]) -> None:
    """Standalone figure: 2 rows x 4 cols of example graphs, one per seed.

    For each init, pick the 4 seeds whose per-seed mean P(bit=1) is most
    spread out (so we see the seed-to-seed variation, not the average). Then
    sample one graph from each chosen seed."""
    init_modes = list(per_init_per_seed.keys())
    fig, axes = plt.subplots(len(init_modes), 4, figsize=(14, 7))

    colors = {"keras": "#1f77b4", "pytorch_default": "#d62728"}

    for row, init in enumerate(init_modes):
        # Pick 4 seeds spanning the bias range: extremes + 2 in between.
        per_seed = per_init_per_seed[init]
        means = per_seed.mean(axis=1)
        order = np.argsort(means)
        idx = list({order[0], order[len(order) // 3], order[2 * len(order) // 3], order[-1]})
        while len(idx) < 4:
            idx.append([i for i in range(len(order)) if i not in idx][0])
        idx = sorted(idx)

        for col, seed_idx in enumerate(idx[:4]):
            ax = axes[row, col]
            graphs = sample_graphs_for_init(init, seed=42 + seed_idx, n_sessions=1)
            bits = graphs[0]
            A = example_adjacency(bits)
            G = nx.from_numpy_array(A)
            if G.number_of_edges() > 0:
                pos = nx.kamada_kawai_layout(G, pos=nx.spring_layout(G, seed=col))
            else:
                pos = nx.circular_layout(G)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#666", width=0.7)
            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                node_color=colors[init], node_size=110,
                edgecolors="black", linewidths=0.5,
            )
            connected = "connected" if nx.is_connected(G) else "DISCONNECTED"
            seed_mean_p = per_seed[seed_idx].mean()
            ax.set_title(
                f"{init}, seed offset={seed_idx}\n"
                f"{int(bits.sum())} edges, {connected}\n"
                f"mean P(bit=1) for this seed = {seed_mean_p:.3f}",
                fontsize=9,
            )
            ax.set_axis_off()

    fig.suptitle(
        "Example graphs sampled from a freshly-initialized PolicyMLP\n"
        "(no training; one graph per seed; seeds chosen to span the bias range)",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path("plots/init_example_graphs.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
