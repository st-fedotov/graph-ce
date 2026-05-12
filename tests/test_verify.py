"""Tests for the verify_and_save_winner certificate writer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from graph_ce.config import Config
from graph_ce.score import score_graph
from graph_ce.verify import verify_and_save_winner


def _make_config(n: int, score_threshold: float) -> Config:
    return Config.model_validate(
        {
            "problem": {"n": n},
            "model": {"hidden_sizes": [4], "learning_rate": 0.01, "optimizer": "sgd", "init": "keras"},
            "cem": {
                "n_sessions": 4,
                "elite_percentile": 90.0,
                "super_elite_percentile": 95.0,
                "max_iters": 1,
                "train_epochs_per_iter": 1,
                "train_batch_size": 8,
            },
            "parallelism": {"n_islands": 1, "cores_per_island": 1, "start_method": "spawn"},
            "migration": {"enabled": False, "interval_iters": 1, "top_k": 1},
            "stopping": {"wall_clock_seconds": 60.0, "score_threshold": score_threshold},
            "logging": {
                "log_interval_iters": 1,
                "metrics_interval_iters": 1,
                "output_dir": "./runs",
                "stdout_mirror": False,
            },
            "seed": {"master_seed": 0},
        }
    )


def test_verify_saves_artifacts_when_score_above_threshold(tmp_path: Path):
    # K_3 (triangle) at n=3. Score = sqrt(2)+1 - 2 - 1 ≈ -0.586.
    # Set threshold = -10 so this "passes". This is purely a test of the
    # certificate writer, not a real counterexample.
    n = 3
    bits = np.array([1, 1, 1], dtype=np.int8)
    config = _make_config(n=n, score_threshold=-10.0)
    verify_and_save_winner(
        winner_state=bits,
        config=config,
        winner_meta={"island_id": 0, "iter_index": 7, "reported_score": -0.586},
        run_dir=tmp_path,
    )
    assert (tmp_path / "winner_adjacency.npy").exists()
    assert (tmp_path / "winner_bits.npy").exists()
    assert (tmp_path / "winner_certificate.txt").exists()
    cert_json = json.loads((tmp_path / "winner_certificate.json").read_text())
    assert cert_json["n"] == n
    assert cert_json["mu"] == 1
    assert cert_json["island_id"] == 0
    assert cert_json["iter_index"] == 7
    # Score should match an independent re-computation.
    rescore = score_graph(bits, n)
    assert abs(cert_json["score"] - rescore.score) < 1e-12


def test_verify_raises_when_disconnected(tmp_path: Path):
    n = 4
    bits = np.zeros(6, dtype=np.int8)  # empty graph
    config = _make_config(n=n, score_threshold=-1000.0)
    with pytest.raises(AssertionError, match="not connected"):
        verify_and_save_winner(
            winner_state=bits,
            config=config,
            winner_meta={"island_id": 0, "iter_index": 1, "reported_score": 0.0},
            run_dir=tmp_path,
        )


def test_verify_raises_when_score_below_threshold(tmp_path: Path):
    # K_3 has score ≈ -0.586. Setting threshold = 0 means it must raise.
    n = 3
    bits = np.array([1, 1, 1], dtype=np.int8)
    config = _make_config(n=n, score_threshold=0.0)
    with pytest.raises(AssertionError, match="not a counterexample"):
        verify_and_save_winner(
            winner_state=bits,
            config=config,
            winner_meta={"island_id": 0, "iter_index": 1, "reported_score": 0.1},
            run_dir=tmp_path,
        )
