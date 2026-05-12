"""Vectorized autoregressive sampler and training-tuple builder.

Sampling: for each of ``num_edges`` positions, run one forward pass with all
``n_sessions`` partial states batched together. The model output gives
P(bit = 1) at this position; we sample Bernoulli and update the state.

Training tuples: for each elite session and each position p, build the
(state_at_p, position_one_hot_p, bit_p) triple. ``state_at_p`` is the final
elite bit string with positions >= p zeroed out, so we can reconstruct it
from the final string with a precomputed lower-triangular mask.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .model import PolicyMLP


@torch.no_grad()
def sample_batch(
    model: PolicyMLP,
    n_sessions: int,
    num_edges: int,
    generator: Optional[torch.Generator] = None,
) -> np.ndarray:
    """Sample ``n_sessions`` graphs by autoregressive bit-by-bit generation.

    Returns an int8 array of shape (n_sessions, num_edges) with 0/1 entries.
    """
    model.eval()
    state = torch.zeros(n_sessions, num_edges, dtype=torch.float32)
    pos_eye = torch.eye(num_edges, dtype=torch.float32)
    for p in range(num_edges):
        pos_in = pos_eye[p].unsqueeze(0).expand(n_sessions, -1)
        x = torch.cat([state, pos_in], dim=1)
        probs = model.prob(x).squeeze(-1)
        sampled = torch.bernoulli(probs, generator=generator)
        state[:, p] = sampled
    return state.numpy().astype(np.int8)


def build_training_tuples(
    elite_states: np.ndarray,
    num_edges: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand each elite bit string into ``num_edges`` (input, target) tuples.

    ``elite_states`` has shape (n_elites, num_edges) with 0/1 entries.
    Returns:
        inputs: float32 array of shape (n_elites * num_edges, 2 * num_edges)
        targets: float32 array of shape (n_elites * num_edges,) with 0/1
    """
    if elite_states.ndim != 2 or elite_states.shape[1] != num_edges:
        raise ValueError(
            f"elite_states must have shape (n, {num_edges}), got {elite_states.shape}"
        )
    n_elites = elite_states.shape[0]
    states_f = elite_states.astype(np.float32, copy=False)

    # pos_mask[p, j] = 1 iff j < p; zeros bits at positions >= p out of the state.
    pos_mask = np.tri(num_edges, num_edges, k=-1, dtype=np.float32)  # (E, E)

    # state_inputs[e, p, :] = states_f[e] * pos_mask[p]
    state_inputs = states_f[:, None, :] * pos_mask[None, :, :]   # (n_elites, E, E)

    # Position one-hots, broadcast to every elite.
    pos_onehot = np.eye(num_edges, dtype=np.float32)             # (E, E)
    pos_inputs = np.broadcast_to(
        pos_onehot[None, :, :], (n_elites, num_edges, num_edges)
    )

    inputs = np.concatenate([state_inputs, pos_inputs], axis=-1)  # (n_elites, E, 2E)
    inputs = inputs.reshape(n_elites * num_edges, 2 * num_edges)

    targets = states_f.reshape(n_elites * num_edges)
    return inputs, targets
