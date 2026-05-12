"""Tests for sampling and training-tuple construction."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from graph_ce.model import PolicyMLP
from graph_ce.sampler import build_training_tuples, sample_batch


@pytest.mark.parametrize("n", [3, 5, 12])
def test_sample_batch_shape_and_dtype(n):
    num_edges = n * (n - 1) // 2
    model = PolicyMLP(num_edges=num_edges, hidden_sizes=[16, 8])
    states = sample_batch(model, n_sessions=7, num_edges=num_edges)
    assert states.shape == (7, num_edges)
    assert states.dtype == np.int8
    assert set(np.unique(states).tolist()).issubset({0, 1})


def test_sample_batch_determinism_with_seed():
    n = 5
    num_edges = n * (n - 1) // 2
    torch.manual_seed(42)
    model = PolicyMLP(num_edges=num_edges, hidden_sizes=[16, 8])

    g1 = torch.Generator()
    g1.manual_seed(123)
    s1 = sample_batch(model, n_sessions=10, num_edges=num_edges, generator=g1)

    g2 = torch.Generator()
    g2.manual_seed(123)
    s2 = sample_batch(model, n_sessions=10, num_edges=num_edges, generator=g2)

    assert np.array_equal(s1, s2)


@pytest.mark.parametrize("n", [3, 5, 7])
def test_training_tuple_shapes(n):
    num_edges = n * (n - 1) // 2
    n_elites = 4
    rng = np.random.default_rng(0)
    elite_states = rng.integers(0, 2, size=(n_elites, num_edges)).astype(np.int8)
    inputs, targets = build_training_tuples(elite_states, num_edges)
    assert inputs.shape == (n_elites * num_edges, 2 * num_edges)
    assert targets.shape == (n_elites * num_edges,)
    # The state portion has bit p zeroed at position p (causal mask).
    # Concretely the first tuple has the all-zero state.
    assert np.all(inputs[0, :num_edges] == 0)
    # The first position one-hot is at index num_edges + 0.
    assert inputs[0, num_edges] == 1
    # Bit p of the elite state lives at column p in the position one-hot.
    for p in range(num_edges):
        # Tuple (elite=0, position=p) lives at row p.
        assert inputs[p, num_edges + p] == 1
        # State bits at position >= p should be zero.
        assert np.all(inputs[p, p:num_edges] == 0)
        # Target should be elite_states[0, p].
        assert targets[p] == elite_states[0, p]


def test_training_tuples_empty_input_raises():
    with pytest.raises(ValueError):
        build_training_tuples(np.zeros((2, 3), dtype=np.int8), num_edges=10)
