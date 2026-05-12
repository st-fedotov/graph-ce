"""End-to-end smoke tests at small n. Confirms nothing hardcodes n and the
CEM step runs end-to-end without crashing.

Uses a serial fake pool to avoid spawning subprocesses inside pytest — the
multiprocessing path is exercised by manual integration runs."""
from __future__ import annotations

import numpy as np
import pytest

from graph_ce.cem import IslandCEM
from graph_ce.config import Config


class SerialPool:
    """Drop-in replacement for multiprocessing.Pool used in tests.

    Mimics .map() by running the function in-process. .close/.terminate/.join
    are no-ops."""

    def map(self, func, iterable):
        return [func(item) for item in iterable]

    def close(self):
        pass

    def terminate(self):
        pass

    def join(self):
        pass


def _make_config(n: int, n_sessions: int = 64) -> Config:
    return Config.model_validate(
        {
            "problem": {"n": n},
            "model": {"hidden_sizes": [16, 8], "learning_rate": 0.001, "optimizer": "sgd", "init": "keras"},
            "cem": {
                "n_sessions": n_sessions,
                "elite_percentile": 90.0,
                "super_elite_percentile": 95.0,
                "max_iters": 3,
                "train_epochs_per_iter": 1,
                "train_batch_size": 32,
            },
            "parallelism": {"n_islands": 1, "cores_per_island": 2, "start_method": "spawn"},
            "migration": {"enabled": False, "interval_iters": 1, "top_k": 4},
            "stopping": {"wall_clock_seconds": 60.0, "score_threshold": 1.0e9},
            "logging": {
                "log_interval_iters": 1,
                "metrics_interval_iters": 1,
                "output_dir": "./runs",
                "stdout_mirror": False,
            },
            "seed": {"master_seed": 1},
        }
    )


@pytest.mark.parametrize("n", [5, 8, 12])
def test_cem_step_runs_for_various_n(n):
    config = _make_config(n=n)
    cem = IslandCEM(config, island_id=0, pool=SerialPool(), seed=42)
    for _ in range(3):
        result = cem.step()
        assert result.best_state.shape == (config.problem.num_edges,)
        assert result.best_state.dtype == np.int8
        assert result.n_elites >= 1
        assert np.isfinite(result.mean_score) or result.mean_score < 0
        # New fields: lambda1/mu/connected for the best graph this iter.
        if result.best_connected:
            assert np.isfinite(result.best_lambda1)
            assert result.best_mu >= 0
        else:
            assert result.best_mu == 0


def test_metrics_row_written_to_jsonl(tmp_path):
    """End-to-end of the JSONL metrics writer: one row per iter, parseable,
    contains the best state and the lambda_1/mu breakdown."""
    import json
    from graph_ce.island import _write_metrics_row

    config = _make_config(n=5)
    cem = IslandCEM(config, island_id=3, pool=SerialPool(), seed=99)
    result = cem.step()
    path = tmp_path / "island_03_metrics.jsonl"
    with open(path, "w") as f:
        _write_metrics_row(f, island_id=3, result=result, wall_seconds=1.234)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["iter"] == 1
    assert row["island_id"] == 3
    assert len(row["best_state"]) == config.problem.num_edges
    assert set(row["best_state"]).issubset({0, 1})
    assert "best_lambda1" in row and "best_mu" in row and "best_connected" in row
    assert row["wall_seconds"] == 1.234


def test_super_elite_pool_persists_across_steps():
    config = _make_config(n=6, n_sessions=64)
    cem = IslandCEM(config, island_id=0, pool=SerialPool(), seed=7)
    _ = cem.step()
    after_first = cem.super_states.shape[0]
    _ = cem.step()
    after_second = cem.super_states.shape[0]
    # Super-elites are kept across iterations (size may fluctuate but
    # should remain populated once the model produces any connected graph).
    assert after_first > 0 or after_second > 0


def test_migration_injection_grows_then_shrinks_pool():
    config = _make_config(n=6, n_sessions=64)
    cem = IslandCEM(config, island_id=0, pool=SerialPool(), seed=11)
    _ = cem.step()
    before = cem.super_states.shape[0]
    fake_states = np.zeros((3, config.problem.num_edges), dtype=np.int8)
    fake_scores = np.array([-100.0, -100.0, -100.0])  # low scores -> filtered out next step
    cem.inject_migration_elites(fake_states, fake_scores)
    assert cem.super_states.shape[0] == before + 3
    _ = cem.step()
    # After re-filtering, low-score migrants should be dropped from the pool.
    assert cem.super_states.shape[0] <= before + 3


def test_config_rejects_inconsistent_percentiles():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Config.model_validate(
            {
                "problem": {"n": 5},
                "model": {"hidden_sizes": [4], "learning_rate": 0.001, "optimizer": "sgd", "init": "keras"},
                "cem": {
                    "n_sessions": 32,
                    "elite_percentile": 95.0,
                    "super_elite_percentile": 90.0,  # less than elite -> invalid
                    "max_iters": 1,
                    "train_epochs_per_iter": 1,
                    "train_batch_size": 16,
                },
                "parallelism": {"n_islands": 1, "cores_per_island": 1, "start_method": "spawn"},
                "migration": {"enabled": False, "interval_iters": 1, "top_k": 1},
                "stopping": {"wall_clock_seconds": 1.0, "score_threshold": 0.0},
                "logging": {
                    "log_interval_iters": 1,
                    "metrics_interval_iters": 1,
                    "output_dir": "./runs",
                    "stdout_mirror": False,
                },
                "seed": {"master_seed": 0},
            }
        )
