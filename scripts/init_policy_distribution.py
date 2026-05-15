"""Probe: characterize the iter-0 policy distribution under each init scheme.

The bad-init plateau happens *before* CEM has done any training, so the
suspect lives somewhere in the freshly-initialized model. This probe
builds a fresh ``PolicyMLP`` under each requested init scheme and reports
three views of what CEM sees in iteration 0:

  1. Blank-slate per-position P(bit=1). Feed (state=0, position=p) for
     each p; ideally a Bernoulli(0.5) sampler would land tightly near 0.5
     for every p, regardless of how the model was initialized.

  2. Empirical per-position bit frequency from autoregressive sampling
     (what CEM actually sees as samples). With Bernoulli(0.5), each
     position's frequency would land near 0.5 with a standard error of
     1/(2 sqrt(N)).

  3. Mean pairwise Hamming distance between sampled graphs (low → all
     samples are similar → CEM is sampling from a delta, not from the
     uniform).

Run:
    .venv/bin/python scripts/init_policy_distribution.py \\
        --inits keras,pytorch_default \\
        --activations relu \\
        --n-samples 1000

Use ``--activations relu,leaky_relu`` to also see how the activation
swap interacts with the init scheme.
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
from graph_ce.sampler import sample_batch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="init-policy-distribution", description=__doc__)
    p.add_argument("--n", type=int, default=19, help="Graph size.")
    p.add_argument("--hidden-sizes", type=str, default="128,64,4",
                   help="Comma-separated hidden sizes.")
    p.add_argument("--inits", type=str, default="keras,pytorch_default",
                   help="Comma-separated init schemes to compare.")
    p.add_argument("--activations", type=str, default="relu",
                   help="Comma-separated activations to combine with each init.")
    p.add_argument("--n-samples", type=int, default=1000,
                   help="How many graphs to sample per (init, activation) cell.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zero-bias", action="store_true",
                   help="After building each model, force all Linear biases to zero. "
                        "Useful for ablating PyTorch's non-zero uniform bias init.")
    p.add_argument("--output", type=Path,
                   default=Path("plots/init_policy_distribution.png"))
    return p.parse_args(argv)


def zero_all_biases(model: PolicyMLP) -> None:
    import torch.nn as nn
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.zero_()


@torch.no_grad()
def blank_slate_probs(model: PolicyMLP, num_edges: int) -> np.ndarray:
    """P(bit=1) when the state is all zeros and the position one-hot is p,
    for every p. Returns shape (num_edges,)."""
    state = torch.zeros(num_edges, num_edges, dtype=torch.float32)  # all-zero state, one row per p
    pos = torch.eye(num_edges, dtype=torch.float32)
    x = torch.cat([state, pos], dim=1)
    return model.prob(x).squeeze(-1).numpy()


def hamming_stats(samples: np.ndarray, *, max_pairs: int = 20000,
                  rng: np.random.Generator) -> dict:
    n = samples.shape[0]
    # subsample at most max_pairs random pairs to keep this O(max_pairs * num_edges)
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    hd = (samples[i] != samples[j]).sum(axis=1).astype(float)
    return {"mean": float(hd.mean()), "std": float(hd.std()),
            "min": float(hd.min()), "max": float(hd.max())}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    hidden = [int(h) for h in args.hidden_sizes.split(",")]
    inits = [s.strip() for s in args.inits.split(",") if s.strip()]
    acts = [s.strip() for s in args.activations.split(",") if s.strip()]
    num_edges = args.n * (args.n - 1) // 2
    np_rng = np.random.default_rng(args.seed)

    cells = [(init, act) for init in inits for act in acts]
    n_rows = len(cells)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(13.0, 2.8 * n_rows + 0.6),
        constrained_layout=True, squeeze=False,
    )

    print(f"n = {args.n}, num_edges = {num_edges}, hidden = {hidden}, "
          f"n_samples = {args.n_samples}\n")
    header = f"{'init':>16}  {'act':>10}  {'P̄(blank)':>9}  {'σ(blank)':>9}  " \
             f"{'P̄(samp)':>9}  {'σ(samp)':>9}  {'⟨Hamming⟩':>10}"
    print(header)
    print("-" * len(header))

    for row, (init, act) in enumerate(cells):
        torch.manual_seed(args.seed)
        model = PolicyMLP(
            num_edges=num_edges, hidden_sizes=hidden,
            init=init, activation=act,
        )
        if args.zero_bias:
            zero_all_biases(model)

        # (1) Blank-slate probabilities.
        blank = blank_slate_probs(model, num_edges)

        # (2) Autoregressive samples.
        gen = torch.Generator().manual_seed(args.seed + 1)
        samples = sample_batch(
            model, n_sessions=args.n_samples, num_edges=num_edges, generator=gen,
        )  # (N, E) int8
        emp_freq = samples.mean(axis=0)  # per-position bit=1 frequency

        # (3) Pairwise Hamming.
        hd = hamming_stats(samples, rng=np_rng)

        # ---- print row ----
        print(f"{init:>16}  {act:>10}  "
              f"{blank.mean():>9.4f}  {blank.std():>9.4f}  "
              f"{emp_freq.mean():>9.4f}  {emp_freq.std():>9.4f}  "
              f"{hd['mean']:>10.2f}")

        # ---- plot row ----
        label = f"{init} + {act}"
        ax = axes[row, 0]
        ax.hist(blank, bins=30, range=(0.0, 1.0), color="#1f77b4", alpha=0.85)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"{label}: blank-slate P(bit=1) per position\n"
                     f"mean={blank.mean():.3f}, std={blank.std():.3f}",
                     fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("P(bit=1)"); ax.set_ylabel("count of positions")

        ax = axes[row, 1]
        ax.plot(emp_freq, color="#d62728", linewidth=1.0)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        se = 1.0 / (2.0 * np.sqrt(args.n_samples))
        ax.axhspan(0.5 - 2*se, 0.5 + 2*se, color="grey", alpha=0.15,
                   label=f"±2 SE under Bernoulli(½) [N={args.n_samples}]")
        ax.set_title(f"{label}: empirical bit frequency over {args.n_samples} samples\n"
                     f"mean={emp_freq.mean():.3f}, std={emp_freq.std():.3f}",
                     fontsize=10)
        ax.set_xlabel("position"); ax.set_ylabel("freq(bit=1)")
        ax.set_ylim(0.0, 1.0); ax.legend(fontsize=7, loc="lower right")

        ax = axes[row, 2]
        # plot Hamming histogram from the actual sampled pairs
        n_pairs = 20000
        i = np_rng.integers(0, samples.shape[0], size=n_pairs)
        j = np_rng.integers(0, samples.shape[0], size=n_pairs)
        keep = i != j
        hd_vals = (samples[i[keep]] != samples[j[keep]]).sum(axis=1).astype(float)
        ax.hist(hd_vals, bins=40, color="#2ca02c", alpha=0.85)
        ax.axvline(num_edges / 2.0, color="black", linestyle="--", linewidth=0.8,
                   label=f"E/2 = {num_edges/2:.0f}\n(uniform Bernoulli)")
        ax.set_title(f"{label}: pairwise Hamming between samples\n"
                     f"mean={hd['mean']:.1f}, std={hd['std']:.1f}",
                     fontsize=10)
        ax.set_xlabel("edge-differences between two samples")
        ax.set_ylabel("count of pairs")
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        f"Iter-0 policy distribution per init scheme (n={args.n}, "
        f"hidden={hidden}, N={args.n_samples} samples)",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"\nsaved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
