"""Probe: signal during autoregressive generation, under each init.

The blank-state probe (``activation_through_layers.py``) feeds 171 inputs
of ``(state=zeros, position=p)`` and asks "what does the policy do at
position p in the empty graph". This script asks the complementary
question: as the policy actually generates a graph, how do its outputs
and activations evolve as the state fills in?

Two figures are produced.

  ``plots/generation_edge_probs.png``
      Heatmap (step × rollout) of P(bit=1) for each init, sharing one
      color scale; bottom strip shows σ-across-rollouts and
      mean-across-rollouts per step. This is the "variance of resulting
      edge distribution" picture.

  ``plots/activations_during_generation.png``
      4-row × n_inits-col figure. Rows 1–3 are heatmaps of (step × unit
      sorted by σ_step desc) of post-activation values along a single
      representative rollout. Row 4 is the P(bit=1) curve along that
      same rollout. Same layout as ``activation_through_layers.png``,
      so the two are directly comparable.

The "representative rollout" is the column at index ``--rep-rollout``
(default 0) of the edge-probs heatmap, sampled with the same RNG seed
for every init so the cross-reference is well-defined.

Run:
    .venv/bin/python scripts/generation_progression.py \\
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
    p = argparse.ArgumentParser(prog="generation-progression", description=__doc__)
    p.add_argument("--n", type=int, default=19, help="Graph size.")
    p.add_argument("--hidden-sizes", type=str, default="128,64,4",
                   help="Comma-separated hidden sizes.")
    p.add_argument("--inits", type=str, default="keras,pytorch_default",
                   help="Comma-separated init schemes (columns of both figures).")
    p.add_argument("--activation", type=str, default="relu",
                   choices=("relu", "leaky_relu"),
                   help="Activation function used for every cell.")
    p.add_argument("--n-rollouts", type=int, default=50,
                   help="Number of autoregressive rollouts sampled per init "
                        "(columns of the per-rollout heatmap).")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for model init.")
    p.add_argument("--rollout-seed", type=int, default=1,
                   help="Seed for the Bernoulli sampling RNG. Separate from "
                        "the model seed so all inits share sampling stochasticity.")
    p.add_argument("--rep-rollout", type=int, default=0,
                   help="Rollout index used for the activation heatmaps "
                        "(= the leftmost column of the edge-prob heatmap).")
    p.add_argument("--alive-threshold", type=float, default=1e-4,
                   help="Per-unit σ-across-step above which a unit counts as "
                        "'alive' in the alive/total annotation.")
    p.add_argument("--probs-output", type=Path,
                   default=Path("plots/generation_edge_probs.png"))
    p.add_argument("--act-output", type=Path,
                   default=Path("plots/activations_during_generation.png"))
    return p.parse_args(argv)


@torch.no_grad()
def rollout_and_capture(
    model: PolicyMLP, n_rollouts: int, num_edges: int, seed: int,
) -> dict[str, np.ndarray]:
    """Run ``n_rollouts`` autoregressive rollouts and capture per-step
    probabilities and post-activation values.

    Returns a dict with:
        * ``probs``:    (num_edges, n_rollouts) — P(bit=1) at each step.
        * ``state``:    (num_edges, n_rollouts) — sampled bits (int8).
        * ``hidden_i``: (num_edges, n_rollouts, hidden_i) — post-activation
                        values at each step (after the i-th ReLU/LeakyReLU).
    """
    model.eval()

    per_step_caps: dict[str, list[torch.Tensor]] = {}
    handles = []
    counter = {"act": 0}

    def hook_for(name: str):
        def hook(_mod, _inp, out):
            per_step_caps.setdefault(name, []).append(out.detach().clone())
        return hook

    for m in model.net:
        if isinstance(m, (nn.ReLU, nn.LeakyReLU)):
            counter["act"] += 1
            handles.append(
                m.register_forward_hook(hook_for(f"hidden_{counter['act']}"))
            )

    try:
        gen = torch.Generator().manual_seed(seed)
        state = torch.zeros(n_rollouts, num_edges, dtype=torch.float32)
        probs = torch.zeros(num_edges, n_rollouts, dtype=torch.float32)
        pos_eye = torch.eye(num_edges, dtype=torch.float32)
        for p in range(num_edges):
            pos_in = pos_eye[p].unsqueeze(0).expand(n_rollouts, -1)
            x = torch.cat([state, pos_in], dim=1)
            pp = model.prob(x).squeeze(-1)
            probs[p] = pp
            sampled = torch.bernoulli(pp, generator=gen)
            state[:, p] = sampled
    finally:
        for h in handles:
            h.remove()

    out: dict[str, np.ndarray] = {
        "probs": probs.numpy(),
        "state": state.numpy().T.astype(np.int8),
    }
    for name, caps in per_step_caps.items():
        # stack along step: (num_edges, n_rollouts, hidden_i)
        out[name] = torch.stack(caps, dim=0).cpu().numpy()
    return out


def render_edge_probs_figure(
    all_runs: list[tuple[str, dict[str, np.ndarray]]],
    *,
    out_path: Path,
    n: int,
    activation: str,
    n_rollouts: int,
    rep_rollout: int,
) -> None:
    n_inits = len(all_runs)
    num_edges = all_runs[0][1]["probs"].shape[0]

    # Shared color range for the heatmaps so two inits are directly comparable.
    flat_probs = np.concatenate([d["probs"].flatten() for _, d in all_runs])
    vmin, vmax = float(flat_probs.min()), float(flat_probs.max())

    # Shared y-axis for the bottom strip.
    sigmas = np.stack([d["probs"].std(axis=1) for _, d in all_runs])  # (n_inits, num_edges)
    means = np.stack([d["probs"].mean(axis=1) for _, d in all_runs])
    sig_y_max = max(float(sigmas.max()) * 1.15, 1e-3)
    mean_lo = float(means.min()) - 0.02
    mean_hi = float(means.max()) + 0.02

    fig, axes = plt.subplots(
        2, n_inits, figsize=(5.4 * n_inits + 1.0, 6.6),
        constrained_layout=True, squeeze=False,
        gridspec_kw={"height_ratios": [5.0, 1.4]},
    )

    print(f"\n--- edge-probs figure (n_rollouts={n_rollouts}, "
          f"rep_rollout={rep_rollout}, activation={activation}) ---")
    print(f"{'init':>22}  {'P̄':>8}  {'σ̄_step':>9}  {'σ_total':>9}  "
          f"{'P_range':>22}")
    im = None
    for ci, (init, data) in enumerate(all_runs):
        probs = data["probs"]  # (num_edges, n_rollouts)
        sigma_step = probs.std(axis=1)
        mean_step = probs.mean(axis=1)
        print(f"{init:>22}  {probs.mean():>8.4f}  "
              f"{sigma_step.mean():>9.4f}  {probs.std():>9.4f}  "
              f"[{probs.min():>8.4f}, {probs.max():>8.4f}]")

        ax_top = axes[0, ci]
        im = ax_top.imshow(
            probs, aspect="auto", origin="lower",
            cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        ax_top.set_xlabel("rollout #")
        if ci == 0:
            ax_top.set_ylabel("autoregressive step (edge position)")
        # Mark which column the activation figure pulls from.
        ax_top.axvline(rep_rollout, color="white", linewidth=0.8,
                       linestyle="--", alpha=0.7)
        ax_top.set_title(
            f"{init}\n"
            f"P̄ = {probs.mean():.4f}  |  "
            f"σ̄_across_rollouts = {sigma_step.mean():.4f}  |  "
            f"σ_total = {probs.std():.4f}",
            fontsize=9,
        )

        ax_bot = axes[1, ci]
        xs = np.arange(num_edges)
        ax_bot.plot(xs, sigma_step, color="#d62728", linewidth=1.2,
                    label="σ_step")
        ax_bot.set_xlim(0, num_edges - 1)
        ax_bot.set_ylim(0.0, sig_y_max)
        ax_bot.set_xlabel("autoregressive step")
        if ci == 0:
            ax_bot.set_ylabel("σ_step", color="#d62728")
        ax_bot.tick_params(axis="y", colors="#d62728")

        ax_bot_r = ax_bot.twinx()
        ax_bot_r.plot(xs, mean_step, color="#1f77b4", linewidth=1.2,
                      label="P̄_step")
        ax_bot_r.set_ylim(mean_lo, mean_hi)
        if ci == n_inits - 1:
            ax_bot_r.set_ylabel("P̄_step", color="#1f77b4")
        ax_bot_r.tick_params(axis="y", colors="#1f77b4")

    fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.025, pad=0.02,
                 label="P(bit=1)")
    fig.suptitle(
        f"P(bit=1) along {n_rollouts} autoregressive rollouts at iter 0 "
        f"(n={n}, activation={activation})\n"
        f"Top: heatmap of step × concrete rollout (white dashed line marks "
        f"rollout #{rep_rollout}, used by the activations figure).  "
        f"Bottom: per-step σ (red) and mean (blue) across rollouts.",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


def render_activations_figure(
    all_runs: list[tuple[str, dict[str, np.ndarray]]],
    *,
    hidden_sizes: list[int],
    out_path: Path,
    n: int,
    activation: str,
    rep_rollout: int,
    alive_threshold: float,
) -> None:
    n_inits = len(all_runs)
    n_layers = len(hidden_sizes)
    layer_names = [f"hidden_{i+1}" for i in range(n_layers)]

    # Shared per-layer color range (across inits) — over the FULL ensemble,
    # not just rep_rollout. Keeps the rep rollout honest against the global
    # signal scale at this layer. vmin clamped to 0 for visual parity with
    # the LeakyReLU figure (tiny negatives clip to dark, no diverging cmap).
    layer_vlims: dict[str, tuple[float, float]] = {}
    for name in layer_names:
        vals = np.concatenate([d[name].flatten() for _, d in all_runs])
        layer_vlims[name] = (max(0.0, float(vals.min())), float(vals.max()))

    # Shared output y-axis for the rep rollout.
    rep_probs = [data["probs"][:, rep_rollout] for _, data in all_runs]
    pmin = float(min(p.min() for p in rep_probs))
    pmax = float(max(p.max() for p in rep_probs))
    pad = max(0.02, (pmax - pmin) * 0.15)
    out_ylim = (max(0.0, pmin - pad), min(1.0, pmax + pad))

    fig, axes = plt.subplots(
        n_layers + 1, n_inits,
        figsize=(4.8 * n_inits, 2.8 * n_layers + 2.6),
        constrained_layout=True, squeeze=False,
        gridspec_kw={"height_ratios": [3] * n_layers + [1.6]},
    )

    print(f"\n--- activations figure (rep_rollout={rep_rollout}, "
          f"activation={activation}) ---")
    print(f"{'init':>22}  {'layer':>10}  {'alive/total':>11}  "
          f"{'mean σ_step':>11}  {'data range':>22}")
    for ri, name in enumerate(layer_names):
        vmin, vmax = layer_vlims[name]
        im = None
        for ci, (init, data) in enumerate(all_runs):
            rep = data[name][:, rep_rollout, :]  # (num_edges, hidden)
            stds = rep.std(axis=0)
            order = np.argsort(-stds)
            sorted_rep = rep[:, order]
            sorted_stds = stds[order]
            alive = int((sorted_stds > alive_threshold).sum())
            total = sorted_stds.size
            print(f"{init:>22}  {name:>10}  "
                  f"{alive:>4}/{total:<6d}  {sorted_stds.mean():>11.4f}  "
                  f"[{rep.min():>8.4f}, {rep.max():>8.4f}]")
            ax = axes[ri, ci]
            im = ax.imshow(
                sorted_rep, aspect="auto", origin="lower",
                cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest",
            )
            ax.set_xlabel("hidden unit (sorted by σ_step desc)")
            if ci == 0:
                ax.set_ylabel("autoregressive step")
            ax.set_title(
                f"{init} · {name} (post-{activation})\n"
                f"alive {alive}/{total}  |  σ_step = {sorted_stds.mean():.4f}",
                fontsize=9,
            )
        fig.colorbar(im, ax=axes[ri, :].tolist(), fraction=0.025, pad=0.02)

    for ci, (init, data) in enumerate(all_runs):
        ax = axes[n_layers, ci]
        probs = data["probs"][:, rep_rollout]
        ax.plot(probs, color="#1f77b4", linewidth=1.2)
        if out_ylim[0] <= 0.5 <= out_ylim[1]:
            ax.axhline(0.5, color="black", linestyle="--",
                       linewidth=0.7, alpha=0.6)
        ax.set_xlim(0, probs.size - 1)
        ax.set_ylim(*out_ylim)
        ax.set_xlabel("autoregressive step")
        if ci == 0:
            ax.set_ylabel("P(bit=1)")
        ax.set_title(
            f"{init} · output P(bit=1) on rollout #{rep_rollout}\n"
            f"mean = {probs.mean():.4f}  |  σ_step = {probs.std():.4f}",
            fontsize=9,
        )

    fig.suptitle(
        f"Signal propagation through the policy MLP DURING autoregressive "
        f"generation (n={n}, hidden={hidden_sizes}, activation={activation}, "
        f"rollout #{rep_rollout})\n"
        f"Per-layer color scales shared across inits; output y-range "
        f"shared and clipped to data ± {pad:.2f}.",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hidden = [int(h) for h in args.hidden_sizes.split(",")]
    inits = [s.strip() for s in args.inits.split(",") if s.strip()]
    num_edges = args.n * (args.n - 1) // 2

    print(f"n = {args.n}, num_edges = {num_edges}, hidden = {hidden}, "
          f"activation = {args.activation}, "
          f"n_rollouts = {args.n_rollouts}, rep_rollout = {args.rep_rollout}")

    all_runs: list[tuple[str, dict[str, np.ndarray]]] = []
    for init in inits:
        torch.manual_seed(args.seed)
        model = PolicyMLP(
            num_edges=num_edges, hidden_sizes=hidden,
            init=init, activation=args.activation,
        )
        data = rollout_and_capture(
            model, n_rollouts=args.n_rollouts,
            num_edges=num_edges, seed=args.rollout_seed,
        )
        all_runs.append((init, data))

    render_edge_probs_figure(
        all_runs, out_path=args.probs_output, n=args.n,
        activation=args.activation,
        n_rollouts=args.n_rollouts, rep_rollout=args.rep_rollout,
    )
    render_activations_figure(
        all_runs, hidden_sizes=hidden, out_path=args.act_output,
        n=args.n, activation=args.activation,
        rep_rollout=args.rep_rollout, alive_threshold=args.alive_threshold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
