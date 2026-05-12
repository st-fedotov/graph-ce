"""Counterexample verification.

After the coordinator receives a SUCCESS message, we re-score the winning
bit string independently and write the certificate to disk. This guards
against bugs in the sampling/scoring loop reporting a false positive.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .score import (
    DISCONNECTED_PENALTY,
    build_adjacency,
    is_connected,
    matching_number,
    score_graph,
)

logger = logging.getLogger(__name__)


def verify_and_save_winner(
    winner_state: np.ndarray,
    config: Config,
    winner_meta: dict[str, Any],
    run_dir: Path,
) -> None:
    """Re-score the winner; save the adjacency matrix and a human-readable
    certificate. Raises if the winner doesn't actually beat the conjecture."""
    n = config.problem.n
    threshold = config.problem.conjecture_threshold
    result = score_graph(winner_state, n)

    logger.info(
        "verification | connected=%s lambda_1=%.10f mu=%d threshold=%.10f score=%.10f",
        result.connected,
        result.lambda1,
        result.mu,
        threshold,
        result.score,
    )

    if not result.connected:
        raise AssertionError("winner graph is not connected; cannot be a counterexample")
    if result.score <= config.stopping.score_threshold:
        raise AssertionError(
            f"winner re-scored at {result.score} but threshold is "
            f"{config.stopping.score_threshold}; not a counterexample"
        )

    A = build_adjacency(winner_state, n)
    np.save(run_dir / "winner_adjacency.npy", A)
    np.save(run_dir / "winner_bits.npy", winner_state)

    edges = [(int(i), int(j)) for i, j in zip(*np.where(np.triu(A, k=1) > 0))]

    cert_lines = []
    cert_lines.append("Counterexample to Conjecture 2.1 found by graph_ce")
    cert_lines.append("=" * 60)
    cert_lines.append(f"n            = {n}")
    cert_lines.append(f"lambda_1     = {result.lambda1:.12f}")
    cert_lines.append(f"mu           = {result.mu}")
    cert_lines.append(f"threshold    = sqrt(n-1) + 1 = {threshold:.12f}")
    cert_lines.append(f"score        = threshold - lambda_1 - mu = {result.score:.12f}")
    cert_lines.append(f"connected    = {result.connected}")
    cert_lines.append("")
    cert_lines.append(f"found by     = island_{winner_meta['island_id']:02d}")
    cert_lines.append(f"iter         = {winner_meta['iter_index']}")
    cert_lines.append(f"reported     = {winner_meta['reported_score']:.12f}")
    cert_lines.append("")
    cert_lines.append(f"|V|          = {n}")
    cert_lines.append(f"|E|          = {len(edges)}")
    cert_lines.append("edge list (i,j) with i<j:")
    for i, j in edges:
        cert_lines.append(f"  {i:>3} -- {j:>3}")
    cert_lines.append("")
    cert_lines.append("adjacency matrix:")
    for row in A.astype(int):
        cert_lines.append("  " + " ".join(str(v) for v in row))

    cert_text = "\n".join(cert_lines)
    (run_dir / "winner_certificate.txt").write_text(cert_text)

    cert_json = {
        "n": n,
        "lambda_1": result.lambda1,
        "mu": result.mu,
        "threshold": threshold,
        "score": result.score,
        "connected": result.connected,
        "island_id": winner_meta["island_id"],
        "iter_index": winner_meta["iter_index"],
        "reported_score": winner_meta["reported_score"],
        "edges": edges,
    }
    (run_dir / "winner_certificate.json").write_text(json.dumps(cert_json, indent=2))
    logger.info(
        "winner artifacts saved: winner_adjacency.npy, winner_bits.npy, "
        "winner_certificate.txt, winner_certificate.json"
    )
