"""Tests for the custom max_cardinality_matching implementation.

Includes a randomized stress test against NetworkX (kept as a dev-only
dependency) on hundreds of random graphs at the production graph size.
"""
from __future__ import annotations

import numpy as np
import pytest

from graph_ce.matching import max_cardinality_matching


def _from_edges(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.int8)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    return A


def test_empty_graph():
    assert max_cardinality_matching(np.zeros((0, 0), dtype=np.int8)) == 0
    assert max_cardinality_matching(np.zeros((1, 1), dtype=np.int8)) == 0


def test_no_edges():
    assert max_cardinality_matching(np.zeros((5, 5), dtype=np.int8)) == 0


def test_single_edge():
    A = _from_edges(2, [(0, 1)])
    assert max_cardinality_matching(A) == 1


def test_triangle_k3():
    A = _from_edges(3, [(0, 1), (1, 2), (0, 2)])
    assert max_cardinality_matching(A) == 1


def test_k4_perfect_matching():
    A = _from_edges(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    assert max_cardinality_matching(A) == 2


@pytest.mark.parametrize("n", [4, 6, 8, 10, 12])
def test_complete_graph_kn(n):
    A = np.ones((n, n), dtype=np.int8)
    np.fill_diagonal(A, 0)
    assert max_cardinality_matching(A) == n // 2


def test_path_p5():
    # Path on 5 vertices: maximum matching = 2 (edges (0,1) and (2,3) or (1,2) and (3,4)).
    A = _from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    assert max_cardinality_matching(A) == 2


def test_odd_cycle_c5():
    # Odd cycle C_5: max matching is 2 (n-1)/2.
    A = _from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
    assert max_cardinality_matching(A) == 2


def test_even_cycle_c6():
    A = _from_edges(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
    assert max_cardinality_matching(A) == 3


def test_petersen_graph():
    # Petersen graph (10 vertices, 15 edges) has a perfect matching (size 5).
    outer = [(i, (i + 1) % 5) for i in range(5)]
    inner = [(i + 5, (i + 2) % 5 + 5) for i in range(5)]
    spokes = [(i, i + 5) for i in range(5)]
    A = _from_edges(10, outer + inner + spokes)
    assert max_cardinality_matching(A) == 5


def test_disconnected_components():
    # Two triangles. Each contributes 1, total 2.
    A = _from_edges(6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])
    assert max_cardinality_matching(A) == 2


def test_self_loops_ignored():
    A = _from_edges(4, [(0, 1), (2, 3)])
    A[0, 0] = 1
    A[2, 2] = 1
    assert max_cardinality_matching(A) == 2


@pytest.mark.parametrize("n,p,seed", [
    (5, 0.3, 0), (5, 0.5, 1), (5, 0.8, 2),
    (10, 0.2, 0), (10, 0.4, 1), (10, 0.7, 2),
    (15, 0.3, 3), (15, 0.5, 4),
    (19, 0.3, 5), (19, 0.5, 6), (19, 0.7, 7),  # production size
])
def test_matches_networkx_on_random(n, p, seed):
    """Stress test: compare against NetworkX on random Erdos-Renyi graphs."""
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(seed)
    for trial in range(30):
        # Random symmetric adjacency
        upper = (rng.random((n, n)) < p).astype(np.int8)
        upper = np.triu(upper, k=1)
        A = upper + upper.T
        our = max_cardinality_matching(A)
        G = nx.from_numpy_array(A)
        ref = len(nx.max_weight_matching(G, maxcardinality=True, weight=None))
        assert our == ref, (
            f"mismatch n={n} p={p} seed={seed} trial={trial}: "
            f"ours={our} networkx={ref}"
        )
