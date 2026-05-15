"""Probe: how the iter-0 signal propagates through the four-layer MLP.

We feed the network 171 inputs — the *blank state* concatenated with each
position one-hot — and capture the post-activation of every hidden layer.
For each unit, we measure its variation *across positions* (which IS the
signal CEM needs in order to learn a position-dependent policy).

The plot is a grid:
  rows = init schemes (e.g. ``keras``, ``pytorch_default``).
  cols = ``hidden_1``, ``hidden_2``, ``hidden_3`` heatmaps + final
         output P(bit=1) curve.
Each heatmap shows (rows = position 0..170, cols = hidden unit) of the
post-activation value. Units are sorted left-to-right by their
across-position std, so "alive" units cluster on the left and the
flat-coloured slab on the right is the population of dead/saturated
units. An annotation gives "alive/total" units per layer (std > 1e-4)
and the mean across-position std over all units.

Run:
    .venv/bin/python scripts/activation_through_layers.py \\
        --inits keras,pytorch_default --activation relu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_ce.model import PolicyMLP


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="activation-through-layers", description=__doc__)
    p.add_argument("--n", type=int, default=19, help="Graph size.")
    p.add_argument("--hidden-sizes", type=str, default="128,64,4",
                   help="Comma-separated hidden sizes.")
    p.add_argument("--inits", type=str, default="keras,pytorch_default",
                   help="Comma-separated init schemes (rows of the figure).")
    p.add_argument("--activation", type=str, default="relu",
                   choices=("relu", "leaky_relu"),
                   help="Activation function for every cell (one per figure).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alive-threshold", type=float, default=1e-4,
                   help="Per-unit std-across-positions above which a unit "
                        "is considered 'alive'.")
    p.add_argument("--output", type=Path,
                   default=Path("plots/activation_through_layers.png"))
    return p.parse_args(argv)


@torch.no_grad()
def capture_layer_outputs(model: PolicyMLP, x: torch.Tensor) -> dict[str, np.ndarray]:
    """Forward pass; return {layer_name: (num_positions, units) array}."""
    captures: dict[str, torch.Tensor] = {}
    handles = []
    counter = {"act": 0}

    def hook_for(name: str):
        def hook(mod, inp, out):
            captures[name] = out.detach().clone()
        return hook

    for m in model.net:
        if isinstance(m, (nn.ReLU, nn.LeakyReLU)):
            counter["act"] += 1
            handles.append(m.register_forward_hook(hook_for(f"hidden_{counter['act']}")))

    logit = model(x).squeeze(-1)
    for h in handles:
        h.remove()

    out = {k: v.cpu().numpy() for k, v in captures.items()}
    out["logit"] = logit.detach().cpu().numpy()
    out["prob"] = torch.sigmoid(logit).detach().cpu().numpy()
    return out


def sort_units_by_std(activations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_activations, per_unit_std). Sort by std descending."""
    stds = activations.std(axis=0)
    order = np.argsort(-stds)
    return activations[:, order], stds[order]


def render_heatmap(ax, mat: np.ndarray, *, title: str, alive_threshold: float,
                   xlabel: str, ylabel: str) -> None:
    sorted_mat, stds = sort_units_by_std(mat)
    alive = int((stds > alive_threshold).sum())
    total = stds.size

    # Use a symmetric colormap if there are negatives (LeakyReLU can produce them).
    if sorted_mat.min() < 0:
        vmax = np.abs(sorted_mat).max()
        im = ax.imshow(sorted_mat, aspect="auto", origin="lower",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(sorted_mat, aspect="auto", origin="lower",
                       cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title}\n"
        f"alive units (std > {alive_threshold:g}): {alive}/{total}  "
        f"|  mean σ_pos = {stds.mean():.4f}",
        fontsize=9,
    )


def render_output_curve(ax, probs: np.ndarray, *, title: str) -> None:
    ax.plot(probs, color="#1f77b4", linewidth=1.2)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xlim(0, probs.size - 1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("position")
    ax.set_ylabel("P(bit=1)")
    ax.set_title(
        f"{title}\nmean = {probs.mean():.4f}  |  σ_pos = {probs.std():.4f}",
        fontsize=9,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hidden = [int(h) for h in args.hidden_sizes.split(",")]
    inits = [s.strip() for s in args.inits.split(",") if s.strip()]
    num_edges = args.n * (args.n - 1) // 2

    # Build the blank-state + position-one-hot batch.
    state = torch.zeros(num_edges, num_edges, dtype=torch.float32)
    pos = torch.eye(num_edges, dtype=torch.float32)
    x = torch.cat([state, pos], dim=1)

    n_layers = len(hidden)
    n_rows = len(inits)
    n_cols = n_layers + 1  # n_layers heatmaps + 1 output curve
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols + 0.5, 3.6 * n_rows + 0.3),
        constrained_layout=True, squeeze=False,
    )

    print(f"n = {args.n}, num_edges = {num_edges}, hidden = {hidden}, "
          f"activation = {args.activation}\n")
    print(f"{'init':>16}  {'layer':>10}  {'alive/total':>11}  "
          f"{'mean σ_pos':>10}")
    print("-" * 55)

    for ri, init in enumerate(inits):
        torch.manual_seed(args.seed)
        model = PolicyMLP(num_edges=num_edges, hidden_sizes=hidden,
                          init=init, activation=args.activation)
        caps = capture_layer_outputs(model, x)

        for li in range(n_layers):
            name = f"hidden_{li+1}"
            mat = caps[name]
            stds = mat.std(axis=0)
            alive = int((stds > args.alive_threshold).sum())
            print(f"{init:>16}  {name:>10}  "
                  f"{alive:>4}/{stds.size:<6d}  {stds.mean():>10.4f}")

            ax = axes[ri, li]
            render_heatmap(
                ax, mat,
                title=f"{init} · {name} (post-{args.activation})",
                alive_threshold=args.alive_threshold,
                xlabel="hidden unit (sorted by σ_pos desc)",
                ylabel="position",
            )

        ax = axes[ri, n_cols - 1]
        render_output_curve(ax, caps["prob"], title=f"{init} · output P(bit=1)")
        # log final stats
        print(f"{init:>16}  {'output':>10}  "
              f"{'':>4} {'':>6}  {caps['prob'].std():>10.4f}")
        print()

    fig.suptitle(
        f"Signal propagation through the policy MLP at iter 0 "
        f"(n={args.n}, hidden={hidden}, activation={args.activation}, "
        f"input = blank state + one-hot position)",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
