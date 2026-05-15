"""Probe: per-position P(bit=1) DURING autoregressive sampling (iter 0).

The blank-slate probe (``init_policy_distribution.py``) shows what the
network outputs if you feed it ``(state=0, position=p)`` for each p
independently. But real CEM iteration-0 sampling is *autoregressive*:
position p's probability is conditioned on the bits actually sampled
at positions 0..p-1, which grows the input state vector over the course
of one rollout. This probe replays that conditional process.

For each (init, activation) cell:
  * Build a fresh ``PolicyMLP``.
  * Run ``--n-rollouts`` autoregressive rollouts, capturing the model's
    ``P(bit=1)`` at every position.
  * Plot the trajectory cloud:
      - thin lines: every individual rollout
      - heavy line: median across rollouts at each position
      - shaded band: 10th–90th percentile across rollouts

So you can see (a) the *starting* probability (position 0 = blank
state, equal to the blank-slate probe's column-0 value) and (b) how
much the probability moves around as the state fills in.

Run:
    .venv/bin/python scripts/generation_trajectory_distribution.py \\
        --init pytorch_default \\
        --activations relu,leaky_relu \\
        --n-rollouts 200
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

# Make ``graph_ce`` importable without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_ce.model import PolicyMLP


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="generation-trajectory-distribution",
                                description=__doc__)
    p.add_argument("--n", type=int, default=19, help="Graph size.")
    p.add_argument("--hidden-sizes", type=str, default="128,64,4",
                   help="Comma-separated hidden sizes.")
    p.add_argument("--inits", type=str, default="pytorch_default",
                   help="Comma-separated init schemes (rows of the figure).")
    p.add_argument("--activations", type=str, default="relu,leaky_relu",
                   help="Comma-separated activations (cols of the figure).")
    p.add_argument("--n-rollouts", type=int, default=200,
                   help="Autoregressive rollouts to record per cell.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for model init AND for the sampling RNG.")
    p.add_argument("--output", type=Path,
                   default=Path("plots/generation_trajectory_distribution.png"))
    return p.parse_args(argv)


@torch.no_grad()
def sample_and_record(model: PolicyMLP, n_rollouts: int, num_edges: int,
                      seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Autoregressive sampling that ALSO returns per-step probabilities.

    Returns:
        states: int8 (n_rollouts, num_edges) with 0/1 entries.
        probs:  float32 (n_rollouts, num_edges) — the P(bit=1) the model
                emitted at each (state-so-far, position) input.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    state = torch.zeros(n_rollouts, num_edges, dtype=torch.float32)
    probs = torch.zeros(n_rollouts, num_edges, dtype=torch.float32)
    pos_eye = torch.eye(num_edges, dtype=torch.float32)
    for p in range(num_edges):
        pos_in = pos_eye[p].unsqueeze(0).expand(n_rollouts, -1)
        x = torch.cat([state, pos_in], dim=1)
        pp = model.prob(x).squeeze(-1)
        probs[:, p] = pp
        sampled = torch.bernoulli(pp, generator=gen)
        state[:, p] = sampled
    return state.numpy().astype(np.int8), probs.numpy()


def render_cell(ax, probs: np.ndarray, *, title: str, n_rollouts: int,
                num_edges: int) -> None:
    # spaghetti
    for r in range(probs.shape[0]):
        ax.plot(probs[r], color="#1f77b4", alpha=max(0.02, 1.5 / probs.shape[0]),
                linewidth=0.5)
    # median + 10-90 band
    median = np.median(probs, axis=0)
    p10 = np.percentile(probs, 10, axis=0)
    p90 = np.percentile(probs, 90, axis=0)
    xs = np.arange(num_edges)
    ax.fill_between(xs, p10, p90, color="#1f77b4", alpha=0.25,
                    label="10–90 percentile across rollouts")
    ax.plot(xs, median, color="#1f77b4", linewidth=1.8, label="median")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.7, alpha=0.6)

    ax.set_xlim(0, num_edges - 1)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("edge position (autoregressive step)")
    ax.set_ylabel("P(bit=1) used at this step")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper right")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hidden = [int(h) for h in args.hidden_sizes.split(",")]
    inits = [s.strip() for s in args.inits.split(",") if s.strip()]
    acts = [s.strip() for s in args.activations.split(",") if s.strip()]
    num_edges = args.n * (args.n - 1) // 2

    n_rows = len(inits)
    n_cols = len(acts)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.2 * n_cols, 3.5 * n_rows + 0.3),
        constrained_layout=True, squeeze=False,
    )

    print(f"n = {args.n}, num_edges = {num_edges}, hidden = {hidden}, "
          f"n_rollouts = {args.n_rollouts}\n")
    header = (f"{'init':>30}  {'act':>10}  {'med_P(0)':>9}  "
              f"{'med_P_end':>10}  {'σ_across_pos(median)':>22}")
    print(header)
    print("-" * len(header))

    for ri, init in enumerate(inits):
        for ci, act in enumerate(acts):
            torch.manual_seed(args.seed)
            model = PolicyMLP(num_edges=num_edges, hidden_sizes=hidden,
                              init=init, activation=act)
            states, probs = sample_and_record(
                model, n_rollouts=args.n_rollouts,
                num_edges=num_edges, seed=args.seed + 1,
            )
            median = np.median(probs, axis=0)
            label = f"{init} + {act}"
            print(f"{init:>30}  {act:>10}  {median[0]:>9.4f}  "
                  f"{median[-1]:>10.4f}  {median.std():>22.4f}")

            ax = axes[ri, ci]
            render_cell(ax, probs, title=label, n_rollouts=args.n_rollouts,
                        num_edges=num_edges)

    fig.suptitle(
        f"Autoregressive-rollout P(bit=1) trajectories at iter 0 "
        f"(n={args.n}, hidden={hidden}, {args.n_rollouts} rollouts/cell)",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"\nsaved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
