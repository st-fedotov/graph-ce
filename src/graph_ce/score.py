"""Conjecture 2.1 score function.

For a simple graph G on n vertices with adjacency matrix A:
    score(G) = sqrt(n - 1) + 1  -  lambda_1(A)  -  mu(G)
where lambda_1 is the largest eigenvalue of A and mu is the size of a maximum
matching. Conjecture 2.1 asserts ``score(G) <= 0`` for every connected graph.
Any sample with ``score(G) > 0`` is therefore a counterexample.

Disconnected graphs are excluded from the conjecture, so we assign them a
large negative score (they sink to the bottom of the elite ranking and are
never selected for training or migration).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from .matching import max_cardinality_matching

logger = logging.getLogger(__name__)

DISCONNECTED_PENALTY: float = -1.0e6


@dataclass(frozen=True)
class ScoreResult:
    score: float
    connected: bool
    lambda1: float  # NaN if disconnected
    mu: int         # 0 if disconnected


def build_adjacency(bits: np.ndarray, n: int) -> np.ndarray:
    """Build symmetric n×n adjacency from upper-triangle bits.

    ``bits`` is a length n*(n-1)/2 array of 0/1, ordered as the upper triangle
    of an n×n matrix in row-major (i, j) order for i < j.
    """
    expected = n * (n - 1) // 2
    if bits.shape != (expected,):
        raise ValueError(f"bits must have shape ({expected},) for n={n}, got {bits.shape}")
    A = np.zeros((n, n), dtype=np.float64)
    iu = np.triu_indices(n, k=1)
    A[iu] = bits
    A = A + A.T
    return A


def is_connected(A: np.ndarray) -> bool:
    """BFS connectedness check on a dense adjacency matrix."""
    n = A.shape[0]
    if n == 0:
        return True
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    stack = [0]
    while stack:
        v = stack.pop()
        # Neighbors of v: indices where A[v] > 0 and not yet visited.
        neighbors = np.flatnonzero(A[v])
        for u in neighbors:
            if not visited[u]:
                visited[u] = True
                stack.append(int(u))
    return bool(visited.all())


def matching_number(A: np.ndarray) -> int:
    """Maximum-cardinality matching size (custom Edmonds blossom)."""
    return max_cardinality_matching(A)


def score_graph(bits: np.ndarray, n: int) -> ScoreResult:
    """Score a single sampled graph encoded as upper-triangle bits."""
    A = build_adjacency(bits, n)
    if not is_connected(A):
        return ScoreResult(
            score=DISCONNECTED_PENALTY,
            connected=False,
            lambda1=float("nan"),
            mu=0,
        )
    eigvals = np.linalg.eigvalsh(A)
    lambda1 = float(eigvals[-1])
    mu = matching_number(A)
    threshold = math.sqrt(n - 1) + 1.0
    score = threshold - lambda1 - mu
    return ScoreResult(score=score, connected=True, lambda1=lambda1, mu=mu)


def score_one(args: tuple[np.ndarray, int]) -> ScoreResult:
    """Picklable single-arg wrapper for use with multiprocessing.Pool.map."""
    bits, n = args
    return score_graph(bits, n)


def score_chunk(args: tuple[np.ndarray, int]) -> list[ScoreResult]:
    """Score a (chunk_size, num_edges) batch sequentially in one Pool worker.

    Chunking dramatically reduces IPC overhead vs. mapping one graph per task.
    """
    bits_batch, n = args
    return [score_graph(bits_batch[i], n) for i in range(bits_batch.shape[0])]


def score_batch(bits_batch: np.ndarray, n: int) -> list[ScoreResult]:
    """Sequential scoring of a batch. Use ``score_chunk`` via a Pool for
    parallel scoring; this is mainly for tests."""
    return [score_graph(bits_batch[i], n) for i in range(bits_batch.shape[0])]
