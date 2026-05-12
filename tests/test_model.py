"""Tests for the PolicyMLP, including the Keras-style initialization that
matters for CEM's initial exploration distribution."""
from __future__ import annotations

import numpy as np
import torch

from graph_ce.model import PolicyMLP


def test_biases_initialized_to_zero():
    """Keras Dense default: bias_initializer='zeros'. PyTorch's default
    initializes biases to a small non-zero uniform; the 'keras' init mode
    explicitly overrides."""
    torch.manual_seed(0)
    model = PolicyMLP(num_edges=10, hidden_sizes=[8, 4], init="keras")
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            assert torch.all(module.bias == 0), (
                "expected zero bias init for Keras parity"
            )


def test_pytorch_default_init_has_nonzero_biases():
    """Sanity check on the legacy/ablation path: PyTorch's default init
    produces non-zero biases. This is the 'bad' init mode that caused our
    CEM plateaus."""
    torch.manual_seed(0)
    model = PolicyMLP(num_edges=10, hidden_sizes=[8, 4], init="pytorch_default")
    found_nonzero = False
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and torch.any(module.bias != 0):
            found_nonzero = True
            break
    assert found_nonzero, "pytorch_default init should leave biases non-zero"


def test_unknown_init_scheme_raises():
    import pytest
    with pytest.raises(ValueError, match="unknown model.init"):
        PolicyMLP(num_edges=10, hidden_sizes=[8, 4], init="custom_thing")


def test_initial_output_is_near_half():
    """With Glorot weights and zero biases, the initial sigmoid output should
    be close to 0.5 for arbitrary inputs (no built-in bias toward 0 or 1).
    This is what gives CEM its uniform-random starting policy."""
    torch.manual_seed(0)
    n = 19
    num_edges = n * (n - 1) // 2  # 171
    model = PolicyMLP(num_edges=num_edges, hidden_sizes=[128, 64, 4])
    # Realistic inputs: a batch of (state, position_one_hot) pairs.
    state = torch.zeros(256, num_edges)
    pos = torch.eye(num_edges)[:256] if num_edges >= 256 else torch.eye(num_edges)
    pos = pos[: state.shape[0]] if pos.shape[0] >= state.shape[0] else pos.repeat(
        state.shape[0] // pos.shape[0] + 1, 1
    )[: state.shape[0]]
    x = torch.cat([state, pos], dim=1)
    with torch.no_grad():
        probs = model.prob(x).squeeze(-1).numpy()
    # Average prob should be near 0.5 (no built-in bias).
    assert 0.4 < float(np.mean(probs)) < 0.6, (
        f"initial policy is biased: mean prob = {np.mean(probs):.3f}"
    )
