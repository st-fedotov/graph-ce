"""Island worker process.

One instance per island; spawned by the coordinator. Owns its own MLP,
optimizer, RNG, and a multiprocessing.Pool of score-evaluation workers.

Communication with the coordinator:
  * ``status_queue``  outbound: PROGRESS / MIGRATE_OUT / SUCCESS / FAILED.
  * ``migration_in_queue`` inbound: MIGRATE_IN packets with the global top-k.
  * ``stop_event``  inbound: when set, the island exits cleanly.
"""
from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
import queue as _queue
import time
import traceback
from pathlib import Path
from typing import TextIO

import numpy as np

from .cem import CemStepResult, IslandCEM
from .config import Config
from .logging_setup import pin_blas_threads, setup_logging


def _score_worker_init(log_path: str) -> None:
    """Initializer for pool workers: pin threads, attach to island log."""
    pin_blas_threads(1)
    setup_logging(log_path, mirror_stdout=False)


def island_main(
    config_dict: dict,
    island_id: int,
    seed: int,
    run_dir: str,
    status_queue: mp.Queue,
    migration_in_queue: mp.Queue,
    stop_event,
) -> None:
    """Entry point for one island worker. Designed to be the target of
    ``mp.get_context('spawn').Process``."""
    pin_blas_threads(1)

    log_path = Path(run_dir) / f"island_{island_id:02d}.log"
    setup_logging(log_path, mirror_stdout=False)
    logger = logging.getLogger(f"island_{island_id:02d}")

    config: Config
    last_iter = 0
    try:
        config = Config.model_validate(config_dict)
        import torch as _torch
        logger.info(
            "island starting | island_id=%d seed=%d n=%d num_edges=%d cores=%d "
            "OMP_NUM_THREADS=%s torch.get_num_threads=%d",
            island_id,
            seed,
            config.problem.n,
            config.problem.num_edges,
            config.parallelism.cores_per_island,
            os.environ.get("OMP_NUM_THREADS"),
            _torch.get_num_threads(),
        )

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=config.parallelism.cores_per_island,
            initializer=_score_worker_init,
            initargs=(str(log_path),),
        )
        metrics_path = Path(run_dir) / f"island_{island_id:02d}_metrics.jsonl"
        metrics_file = open(metrics_path, "a", buffering=1)  # line-buffered
        logger.info("metrics stream open at %s", metrics_path)
        try:
            cem = IslandCEM(config, island_id, pool, seed)
            last_iter = _run_cem_loop(
                cem=cem,
                config=config,
                island_id=island_id,
                status_queue=status_queue,
                migration_in_queue=migration_in_queue,
                stop_event=stop_event,
                logger=logger,
                metrics_file=metrics_file,
            )
        finally:
            try:
                metrics_file.close()
            except Exception:
                logger.exception("failed to close metrics file")
            pool.close()
            pool.terminate()
            pool.join()
        logger.info("island exiting cleanly at iter=%d", last_iter)
    except BaseException:
        tb = traceback.format_exc()
        logger.exception("island_%02d crashed (last_iter=%d)", island_id, last_iter)
        try:
            status_queue.put(("FAILED", island_id, last_iter, tb))
        except Exception:
            logger.exception("failed to report FAILED status to coordinator")
        raise


def _safe_float(x: float) -> float | None:
    """JSON cannot represent NaN/Inf; map them to null."""
    return None if x is None or not math.isfinite(x) else float(x)


def _write_metrics_row(
    f: TextIO,
    *,
    island_id: int,
    result: CemStepResult,
    wall_seconds: float,
) -> None:
    row = {
        "iter": result.iter_index,
        "wall_seconds": round(wall_seconds, 4),
        "step_seconds": round(result.step_seconds, 4),
        "best_score": _safe_float(result.best_score),
        "best_lambda1": _safe_float(result.best_lambda1),
        "best_mu": result.best_mu,
        "best_connected": result.best_connected,
        "mean_score": _safe_float(result.mean_score),
        "elite_mean_score": _safe_float(result.elite_mean_score),
        "n_elites": result.n_elites,
        "n_super_elites": result.n_super_elites,
        "mean_loss": _safe_float(result.mean_train_loss),
        "best_state": result.best_state.astype(int).tolist(),
        "island_id": island_id,
    }
    f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _run_cem_loop(
    cem: IslandCEM,
    config: Config,
    island_id: int,
    status_queue: mp.Queue,
    migration_in_queue: mp.Queue,
    stop_event,
    logger: logging.Logger,
    metrics_file: TextIO,
) -> int:
    log_interval = config.logging.log_interval_iters
    metrics_interval = config.logging.metrics_interval_iters
    migration_enabled = config.migration.enabled
    migration_interval = config.migration.interval_iters
    migration_top_k = config.migration.top_k
    threshold = config.stopping.score_threshold
    max_iters = config.cem.max_iters
    migration_wait_seconds = 300.0

    start_time = time.monotonic()
    while not stop_event.is_set() and cem.iter_index < max_iters:
        result = cem.step()

        wall = time.monotonic() - start_time
        if result.iter_index % metrics_interval == 0:
            _write_metrics_row(
                metrics_file,
                island_id=island_id,
                result=result,
                wall_seconds=wall,
            )

        # Counterexample check: any score > threshold ends the run.
        if result.best_score > threshold:
            logger.info(
                "FOUND counterexample | iter=%d best_score=%.6f",
                result.iter_index,
                result.best_score,
            )
            status_queue.put(
                (
                    "SUCCESS",
                    island_id,
                    result.iter_index,
                    {
                        "score": float(result.best_score),
                        "state": result.best_state.copy(),
                    },
                )
            )
            stop_event.set()
            return result.iter_index

        # Periodic progress log + status report.
        if result.iter_index == 1 or result.iter_index % log_interval == 0:
            logger.info(
                "iter=%d best=%.4f elite_mean=%.4f mean=%.4f "
                "n_elites=%d n_super=%d loss=%.5f step=%.2fs",
                result.iter_index,
                result.best_score,
                result.elite_mean_score,
                result.mean_score,
                result.n_elites,
                result.n_super_elites,
                result.mean_train_loss,
                result.step_seconds,
            )
            status_queue.put(
                (
                    "PROGRESS",
                    island_id,
                    result.iter_index,
                    {
                        "best_score": float(result.best_score),
                        "mean_score": float(result.mean_score),
                        "elite_mean_score": float(result.elite_mean_score),
                        "n_elites": int(result.n_elites),
                        "n_super_elites": int(result.n_super_elites),
                        "step_seconds": float(result.step_seconds),
                        "wall_seconds": time.monotonic() - start_time,
                    },
                )
            )

        # Migration.
        if migration_enabled and result.iter_index % migration_interval == 0:
            tick = result.iter_index // migration_interval
            out_states, out_scores = cem.current_top_k(migration_top_k)
            logger.info(
                "migration tick=%d sending top_k=%d (best=%.4f) to coordinator",
                tick,
                out_states.shape[0],
                float(out_scores[0]) if out_scores.size else float("nan"),
            )
            status_queue.put(
                ("MIGRATE_OUT", island_id, tick, (out_states, out_scores))
            )
            try:
                msg = migration_in_queue.get(timeout=migration_wait_seconds)
            except _queue.Empty:
                logger.warning(
                    "migration tick=%d: no response from coordinator after %ds; "
                    "continuing without injection",
                    tick,
                    int(migration_wait_seconds),
                )
            else:
                if msg[0] == "MIGRATE_IN":
                    _, in_tick, in_states, in_scores = msg
                    logger.info(
                        "migration tick=%d received %d migrants (global best=%.4f)",
                        in_tick,
                        in_states.shape[0],
                        float(in_scores[0]) if in_scores.size else float("nan"),
                    )
                    cem.inject_migration_elites(in_states, in_scores)
                elif msg[0] == "STOP":
                    logger.info("migration: received STOP signal; exiting")
                    try:
                        status_queue.put(
                            ("DONE", island_id, result.iter_index, "stop_signal")
                        )
                    except Exception:
                        logger.exception("failed to report DONE status")
                    return result.iter_index
                else:
                    logger.warning("migration: unexpected message type %r", msg[0])

    if stop_event.is_set():
        reason = "stop_event"
        logger.info("stop_event set; shutting down at iter=%d", cem.iter_index)
    else:
        reason = "max_iters"
        logger.info("max_iters=%d reached; shutting down", max_iters)
    try:
        status_queue.put(("DONE", island_id, cem.iter_index, reason))
    except Exception:
        logger.exception("failed to report DONE status")
    return cem.iter_index
