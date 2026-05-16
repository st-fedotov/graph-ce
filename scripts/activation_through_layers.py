"""Probe: how the iter-0 signal propagates through the four-layer MLP.

We feed the network 171 inputs — the *blank state* concatenated with each
position one-hot — and capture the post-activation of every hidden layer.
For each unit, we measure its variation *across positions* (which IS the
signal CEM needs in order to learn a position-dependent policy).

The plot is a grid:
  rows = ``hidden_1``, ``hidden_2``, ``hidden_3`` heatmaps + final output
         P(bit=1) curve.
  cols = init schemes (e.g. ``keras``, ``pytorch_default``).

Each heatmap shows (rows = position 0..170, cols = hidden unit) of the
post-activation value, with units sorted left-to-right by their
across-position std (so "alive" units cluster on the left and the
flat-coloured slab on the right is dead/saturated units). The color scale
is **shared across init schemes within each layer**, so two cells in the
same row are directly comparable in magnitude. The output row uses a
**shared y-axis tightly fitted to the actual data range**.

An annotation per heatmap reports "alive/total" units (std > 1e-4) and
the mean across-position std over all units.

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
                   help="Comma-separated init schemes (columns of the figure).")
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


def render_heatmap(ax, mat: np.ndarray, *, vmin: float, vmax: float, cmap: str,
                   alive_threshold: float, title: str,
                   xlabel: str, ylabel: str):
    """Render one heatmap with externally-supplied color scale.

    Returns the image handle so a shared colorbar can be attached to the
    row at the call site.
    """
    sorted_mat, stds = sort_units_by_std(mat)
    alive = int((stds > alive_threshold).sum())
    total = stds.size

    im = ax.imshow(sorted_mat, aspect="auto", origin="lower",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title}\n"
        f"alive {alive}/{total}  |  σ_pos = {stds.mean():.4f}",
        fontsize=9,
    )
    return im


def render_output_curve(ax, probs: np.ndarray, *, title: str,
                        ylim: tuple[float, float]) -> None:
    ax.plot(probs, color="#1f77b4", linewidth=1.2)
    if ylim[0] <= 0.5 <= ylim[1]:
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xlim(0, probs.size - 1)
    ax.set_ylim(*ylim)
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

    # Build the blank-state + position-one-hot batch (171 rows × 342 cols).
    state = torch.zeros(num_edges, num_edges, dtype=torch.float32)
    pos = torch.eye(num_edges, dtype=torch.float32)
    x = torch.cat([state, pos], dim=1)

    n_layers = len(hidden)
    n_inits = len(inits)

    # ---- Phase 1: capture all activations so we can compute global scales ----
    all_caps: list[tuple[str, dict[str, np.ndarray]]] = []
    for init in inits:
        torch.manual_seed(args.seed)
        model = PolicyMLP(num_edges=num_edges, hidden_sizes=hidden,
                          init=init, activation=args.activation)
        all_caps.append((init, capture_layer_outputs(model, x)))

    # Shared per-layer color range (across all inits).
    # Always use viridis with vmin >= 0 for visual consistency between the
    # ReLU and LeakyReLU figures. LeakyReLU's slope-0.01 negatives are at
    # ~1% of the positive magnitudes (noise level); a diverging palette
    # centered at 0 would map them to invisible-pale-blue and just break
    # the colormap parity with the ReLU figure for no informational gain.
    layer_specs: list[tuple[str, float, float, str]] = []
    for li in range(n_layers):
        name = f"hidden_{li+1}"
        vals = np.concatenate([c[name].flatten() for _, c in all_caps])
        layer_specs.append(
            (name, max(0.0, float(vals.min())), float(vals.max()), "viridis")
        )

    # Shared output ylim, tight around the actual data with a small pad.
    all_probs = np.concatenate([c["prob"] for _, c in all_caps])
    pmin, pmax = float(all_probs.min()), float(all_probs.max())
    pad = max(0.02, (pmax - pmin) * 0.15)
    out_ylim = (max(0.0, pmin - pad), min(1.0, pmax + pad))

    # ---- Phase 2: render ----
    fig, axes = plt.subplots(
        n_layers + 1, n_inits,
        figsize=(4.8 * n_inits, 2.8 * n_layers + 2.6),
        constrained_layout=True, squeeze=False,
        gridspec_kw={"height_ratios": [3] * n_layers + [1.6]},
    )

    print(f"n = {args.n}, num_edges = {num_edges}, hidden = {hidden}, "
          f"activation = {args.activation}\n")
    print(f"{'init':>16}  {'layer':>10}  {'alive/total':>11}  "
          f"{'mean σ_pos':>10}  {'data range':>22}")
    print("-" * 80)

    for ri, (name, vmin, vmax, cmap) in enumerate(layer_specs):
        im = None
        for ci, (init, caps) in enumerate(all_caps):
            mat = caps[name]
            stds = mat.std(axis=0)
            alive = int((stds > args.alive_threshold).sum())
            print(f"{init:>16}  {name:>10}  "
                  f"{alive:>4}/{stds.size:<6d}  {stds.mean():>10.4f}  "
                  f"[{mat.min():>8.4f}, {mat.max():>8.4f}]")
            ax = axes[ri, ci]
            im = render_heatmap(
                ax, mat, vmin=vmin, vmax=vmax, cmap=cmap,
                alive_threshold=args.alive_threshold,
                title=f"{init} · {name} (post-{args.activation})",
                xlabel="hidden unit (sorted by σ_pos desc)",
                ylabel="position",
            )
        # One shared colorbar for the row.
        fig.colorbar(im, ax=axes[ri, :].tolist(), fraction=0.025, pad=0.02)

    # Output row
    for ci, (init, caps) in enumerate(all_caps):
        ax = axes[n_layers, ci]
        probs = caps["prob"]
        print(f"{init:>16}  {'output':>10}  {'':>4} {'':>6}  "
              f"{probs.std():>10.4f}  [{probs.min():>8.4f}, {probs.max():>8.4f}]")
        render_output_curve(
            ax, probs,
            title=f"{init} · output P(bit=1)",
            ylim=out_ylim,
        )

    fig.suptitle(
        f"Signal propagation through the policy MLP at iter 0 "
        f"(n={args.n}, hidden={hidden}, activation={args.activation}, "
        f"input = blank state + one-hot position)\n"
        f"Color scales per layer shared across init schemes; "
        f"output y-range shared and clipped to actual data ± {pad:.2f}.",
        fontsize=11,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"\nsaved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
