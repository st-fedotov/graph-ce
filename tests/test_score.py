"""Tests for the score function. No hardcoded n anywhere — every test
parametrizes the graph size."""
from __future__ import annotations

import math

import numpy as np
import pytest

from graph_ce.score import (
    DISCONNECTED_PENALTY,
    build_adjacency,
    is_connected,
    matching_number,
    score_graph,
)


def bits_from_edges(edges: list[tuple[int, int]], n: int) -> np.ndarray:
    """Convert an edge list into the upper-triangle bit vector."""
    num_edges = n * (n - 1) // 2
    bits = np.zeros(num_edges, dtype=np.int8)
    idx = {}
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[(i, j)] = k
            k += 1
    for i, j in edges:
        if i > j:
            i, j = j, i
        bits[idx[(i, j)]] = 1
    return bits


def test_build_adjacency_symmetric_for_any_n():
    rng = np.random.default_rng(0)
    for n in [3, 5, 10, 19]:
        num_edges = n * (n - 1) // 2
        bits = rng.integers(0, 2, size=num_edges).astype(np.int8)
        A = build_adjacency(bits, n)
        assert A.shape == (n, n)
        assert np.array_equal(A, A.T)
        assert np.all(np.diag(A) == 0)


def test_path_p3_score():
    # P_3: nodes 0-1-2. lambda_1 = sqrt(2), mu = 1.
    n = 3
    bits = bits_from_edges([(0, 1), (1, 2)], n)
    result = score_graph(bits, n)
    assert result.connected
    assert math.isclose(result.lambda1, math.sqrt(2.0), rel_tol=1e-10)
    assert result.mu == 1
    threshold = math.sqrt(n - 1) + 1
    assert math.isclose(result.score, threshold - math.sqrt(2.0) - 1, abs_tol=1e-10)


@pytest.mark.parametrize("n", [4, 6, 10])
def test_complete_graph_score(n):
    # K_n: lambda_1 = n - 1, mu = n // 2.
    bits = np.ones(n * (n - 1) // 2, dtype=np.int8)
    result = score_graph(bits, n)
    assert result.connected
    assert math.isclose(result.lambda1, n - 1, rel_tol=1e-9, abs_tol=1e-9)
    assert result.mu == n // 2


def test_disconnected_gets_penalty():
    # n=4, only edge 0-1. Nodes 2, 3 isolated.
    n = 4
    bits = bits_from_edges([(0, 1)], n)
    result = score_graph(bits, n)
    assert not result.connected
    assert result.score == DISCONNECTED_PENALTY


def test_empty_graph_disconnected():
    n = 5
    bits = np.zeros(n * (n - 1) // 2, dtype=np.int8)
    result = score_graph(bits, n)
    assert not result.connected
    assert result.score == DISCONNECTED_PENALTY


def test_is_connected_on_cycle():
    n = 6
    edges = [(i, (i + 1) % n) for i in range(n)]
    bits = bits_from_edges(edges, n)
    A = build_adjacency(bits, n)
    assert is_connected(A)


def test_matching_number_perfect_matching():
    # 4 nodes, edges (0,1) and (2,3). Maximum matching = 2.
    n = 4
    bits = bits_from_edges([(0, 1), (2, 3)], n)
    A = build_adjacency(bits, n)
    assert matching_number(A) == 2
