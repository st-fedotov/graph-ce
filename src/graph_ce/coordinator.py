"""Coordinator: spawns islands, brokers migration, watches for success.

Responsibilities:
  * Create the run directory, dump the resolved config, set up logging.
  * Spawn the configured number of island processes (one mp.Process each).
  * Listener thread: drain status_queue and act on PROGRESS / MIGRATE_OUT /
    SUCCESS / FAILED messages.
  * Liveness thread: detect islands that die without sending FAILED.
  * Main thread: wait for stop_event (set on success, timeout, or all-dead).
  * On shutdown: signal islands, join with timeout, terminate stragglers,
    write summary.txt, verify the winner (if any).
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import signal
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import Config
from .island import island_main
from .logging_setup import setup_logging
from .verify import verify_and_save_winner


@dataclass
class WinnerRecord:
    island_id: int
    iter_index: int
    score: float
    state: np.ndarray


@dataclass
class CoordinatorState:
    n_total_islands: int
    last_iter_by_island: dict[int, int] = field(default_factory=dict)
    failed_islands: dict[int, str] = field(default_factory=dict)  # id -> traceback
    done_islands: dict[int, str] = field(default_factory=dict)    # id -> reason
    migration_buffer: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )  # tick -> [(island_id, states, scores)]
    winner: Optional[WinnerRecord] = None
    start_time: float = 0.0

    @property
    def n_finished(self) -> int:
        """Islands that are no longer participating (finished cleanly, failed,
        or won)."""
        finished = set(self.failed_islands) | set(self.done_islands)
        if self.winner is not None:
            finished.add(self.winner.island_id)
        return len(finished)

    @property
    def n_active(self) -> int:
        return self.n_total_islands - self.n_finished


class Coordinator:
    def __init__(self, config: Config, run_dir: Path) -> None:
        self.cfg = config
        self.run_dir = run_dir
        self.logger = logging.getLogger("coordinator")

        self.ctx = mp.get_context(config.parallelism.start_method)
        self.status_queue: mp.Queue = self.ctx.Queue()
        self.migration_in_queues: dict[int, mp.Queue] = {}
        self.stop_event = self.ctx.Event()
        self.state = CoordinatorState(n_total_islands=config.parallelism.n_islands)

        # Coarse lock around CoordinatorState. Listener and liveness threads
        # both mutate it; main thread reads it.
        self._lock = threading.Lock()

        self.islands: list[mp.Process] = []
        self._listener_thread: Optional[threading.Thread] = None
        self._liveness_thread: Optional[threading.Thread] = None
        # Plain Python boolean set by the signal handler. The handler MUST
        # NOT touch mp.Event.set() directly: mp.Event.set() acquires the
        # same internal Lock that mp.Event.wait() holds across signal
        # interrupts, which deadlocks the handler. Instead, the main thread
        # polls this flag and calls stop_event.set() from its own context.
        self._sigterm_received = False

    # ------------------------------------------------------------------ launch

    def run(self) -> int:
        """Run the full coordination loop. Returns exit code: 0 on success,
        1 on timeout / failure / no counterexample found."""
        self.state.start_time = time.monotonic()
        self._install_signal_handlers()
        self._spawn_islands()
        self._start_helper_threads()

        try:
            self._wait_for_completion()
        finally:
            self._shutdown()

        return self._finalize()

    # --------------------------------------------------------------- internals

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):
            # Keep this minimal: only async-signal-safe operations. NO
            # mp.Event.set(), NO logging (logging acquires locks too).
            # The main loop polls _sigterm_received and translates it into
            # stop_event.set() safely.
            self._sigterm_received = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _spawn_islands(self) -> None:
        master_seed = self.cfg.seed.master_seed
        if master_seed is None:
            master_seed = int(time.time_ns() & 0xFFFFFFFF)
            self.logger.info("master_seed was null; generated %d from clock", master_seed)
        else:
            self.logger.info("using configured master_seed=%d", master_seed)

        cfg_dict = self.cfg.model_dump()
        for island_id in range(self.cfg.parallelism.n_islands):
            mig_q: mp.Queue = self.ctx.Queue()
            self.migration_in_queues[island_id] = mig_q
            seed = master_seed + island_id
            p = self.ctx.Process(
                target=island_main,
                name=f"island_{island_id:02d}",
                args=(
                    cfg_dict,
                    island_id,
                    seed,
                    str(self.run_dir),
                    self.status_queue,
                    mig_q,
                    self.stop_event,
                ),
            )
            p.start()
            self.islands.append(p)
            self.logger.info(
                "spawned island_%02d pid=%d seed=%d",
                island_id,
                p.pid,
                seed,
            )

    def _start_helper_threads(self) -> None:
        self._listener_thread = threading.Thread(
            target=self._listener_loop, name="status-listener", daemon=True
        )
        self._liveness_thread = threading.Thread(
            target=self._liveness_loop, name="liveness-monitor", daemon=True
        )
        self._listener_thread.start()
        self._liveness_thread.start()

    def _wait_for_completion(self) -> None:
        timeout = self.cfg.stopping.wall_clock_seconds
        self.logger.info(
            "coordinator waiting up to %.0fs for completion across %d islands",
            timeout,
            self.cfg.parallelism.n_islands,
        )
        # Poll in short slices instead of one big wait(timeout=12h). With one
        # big wait, multiprocessing.Event.wait() blocks the main thread in C
        # until either the event is set OR the C-level wait returns; Python
        # signal handlers can only run between Python opcodes, so SIGTERM
        # arriving while we're deep in the wait was being deferred for many
        # minutes. Polling every second guarantees the handler gets to run
        # promptly after any signal.
        deadline = self.state.start_time + timeout
        poll_seconds = 1.0
        while time.monotonic() < deadline:
            if self._sigterm_received:
                self.logger.warning(
                    "signal received in handler; setting stop_event from main thread"
                )
                self.stop_event.set()
                return
            if self.stop_event.wait(timeout=poll_seconds):
                self.logger.info(
                    "stop_event fired after %.1fs",
                    time.monotonic() - self.state.start_time,
                )
                return
        self.logger.warning(
            "wall-clock timeout of %.0fs reached; signaling shutdown",
            timeout,
        )
        self.stop_event.set()

    # ------------------------------------------------------------- listener

    def _listener_loop(self) -> None:
        try:
            while True:
                if self.stop_event.is_set():
                    # Drain remaining messages quickly, then exit.
                    drained = self._drain_status_queue(timeout=2.0)
                    if drained == 0:
                        return
                    continue
                try:
                    msg = self.status_queue.get(timeout=1.0)
                except Exception:
                    continue
                self._handle_status_msg(msg)
        except BaseException:
            self.logger.exception("listener thread crashed")

    def _drain_status_queue(self, timeout: float) -> int:
        """Best-effort drain; returns number of messages processed."""
        end = time.monotonic() + timeout
        count = 0
        while time.monotonic() < end:
            try:
                msg = self.status_queue.get(timeout=0.1)
            except Exception:
                break
            self._handle_status_msg(msg)
            count += 1
        return count

    def _handle_status_msg(self, msg: tuple) -> None:
        kind = msg[0]
        if kind == "PROGRESS":
            _, island_id, iter_index, payload = msg
            with self._lock:
                self.state.last_iter_by_island[island_id] = iter_index
            self.logger.info(
                "progress | island=%02d iter=%d best=%.4f mean=%.4f elite_mean=%.4f n_elites=%d step=%.2fs",
                island_id,
                iter_index,
                payload["best_score"],
                payload["mean_score"],
                payload["elite_mean_score"],
                payload["n_elites"],
                payload["step_seconds"],
            )
        elif kind == "MIGRATE_OUT":
            _, island_id, tick, (states, scores) = msg
            with self._lock:
                self.state.migration_buffer.setdefault(tick, []).append(
                    (island_id, states, scores)
                )
                ready = (
                    len(self.state.migration_buffer[tick]) >= self.state.n_active
                    and self.state.n_active > 0
                )
                self.logger.info(
                    "migration | island=%02d submitted tick=%d (%d/%d active)",
                    island_id,
                    tick,
                    len(self.state.migration_buffer[tick]),
                    self.state.n_active,
                )
                if ready:
                    self._broadcast_migration_locked(tick)
        elif kind == "SUCCESS":
            _, island_id, iter_index, payload = msg
            self.logger.info(
                "SUCCESS | island=%02d iter=%d score=%.6f",
                island_id,
                iter_index,
                payload["score"],
            )
            with self._lock:
                if self.state.winner is None:
                    self.state.winner = WinnerRecord(
                        island_id=island_id,
                        iter_index=iter_index,
                        score=float(payload["score"]),
                        state=np.array(payload["state"], dtype=np.int8),
                    )
            self.stop_event.set()
        elif kind == "FAILED":
            _, island_id, last_iter, tb = msg
            with self._lock:
                if island_id in self.state.failed_islands:
                    return  # already recorded
                if island_id in self.state.done_islands:
                    return  # already exited cleanly
                self.state.failed_islands[island_id] = tb
                self.state.last_iter_by_island.setdefault(island_id, last_iter)
                no_islands_left = self.state.n_active == 0
                # Pending migration ticks may now satisfy n_active threshold.
                for tick in list(self.state.migration_buffer.keys()):
                    if (
                        len(self.state.migration_buffer[tick]) >= self.state.n_active
                        and self.state.n_active > 0
                    ):
                        self._broadcast_migration_locked(tick)
            self.logger.critical(
                "island_%02d FAILED at iter=%d", island_id, last_iter
            )
            self.logger.critical("island_%02d traceback:\n%s", island_id, tb)
            if no_islands_left:
                self.logger.critical("all islands have failed; signaling shutdown")
                self.stop_event.set()
        elif kind == "DONE":
            _, island_id, last_iter, reason = msg
            with self._lock:
                self.state.done_islands.setdefault(island_id, reason)
                self.state.last_iter_by_island[island_id] = last_iter
                no_islands_left = self.state.n_active == 0
            self.logger.info(
                "island_%02d DONE at iter=%d (reason=%s)", island_id, last_iter, reason
            )
            if no_islands_left and self.state.winner is None:
                self.logger.info(
                    "all islands have finished without a counterexample; stopping"
                )
                self.stop_event.set()
        else:
            self.logger.warning("unknown status message kind=%r", kind)

    def _broadcast_migration_locked(self, tick: int) -> None:
        """Caller holds self._lock."""
        packets = self.state.migration_buffer.pop(tick)
        top_k = self.cfg.migration.top_k
        if not packets:
            return
        all_states = np.concatenate([s for _, s, _ in packets], axis=0)
        all_scores = np.concatenate([sc for _, _, sc in packets], axis=0)
        if all_states.shape[0] == 0:
            global_states = all_states
            global_scores = all_scores
        else:
            order = np.argsort(-all_scores)[:top_k]
            global_states = all_states[order].copy()
            global_scores = all_scores[order].copy()
        for island_id, in_q in self.migration_in_queues.items():
            if island_id in self.state.failed_islands:
                continue
            try:
                in_q.put(
                    (
                        "MIGRATE_IN",
                        tick,
                        global_states.copy(),
                        global_scores.copy(),
                    )
                )
            except Exception:
                self.logger.exception(
                    "failed to enqueue MIGRATE_IN to island_%02d", island_id
                )
        best = float(global_scores[0]) if global_scores.size else float("nan")
        self.logger.info(
            "migration tick=%d broadcast | size=%d global_best=%.4f",
            tick,
            global_states.shape[0],
            best,
        )

    # ------------------------------------------------------------- liveness

    def _liveness_loop(self) -> None:
        # Grace period before declaring a dead-without-status island as
        # failed: gives the status_queue time to deliver a DONE / FAILED /
        # SUCCESS message that the island sent just before exiting.
        grace_seconds = 5.0
        first_dead_seen: dict[int, float] = {}
        try:
            while not self.stop_event.is_set():
                time.sleep(2.0)
                for i, p in enumerate(self.islands):
                    if p.is_alive():
                        first_dead_seen.pop(i, None)
                        continue
                    with self._lock:
                        if i in self.state.failed_islands:
                            continue
                        if i in self.state.done_islands:
                            continue
                        if self.state.winner and self.state.winner.island_id == i:
                            continue
                    # Process is dead but coordinator has no terminal status.
                    # Wait one grace period in case the message is still in
                    # transit, then synthesize a FAILED.
                    now = time.monotonic()
                    first = first_dead_seen.setdefault(i, now)
                    if now - first < grace_seconds:
                        continue
                    exitcode = p.exitcode
                    fake_tb = (
                        f"island_{i:02d} died with exitcode={exitcode} "
                        f"without sending DONE/FAILED/SUCCESS. "
                        f"See island_{i:02d}.log for details."
                    )
                    self.logger.critical(fake_tb)
                    last_iter = self.state.last_iter_by_island.get(i, -1)
                    self.status_queue.put(("FAILED", i, last_iter, fake_tb))
                    first_dead_seen.pop(i, None)
        except BaseException:
            self.logger.exception("liveness thread crashed")

    # --------------------------------------------------------------- shutdown

    def _shutdown(self) -> None:
        self.logger.info("beginning shutdown")
        self.stop_event.set()
        # Unblock any island that may be waiting on migration_in_queue.
        for island_id, q in self.migration_in_queues.items():
            try:
                q.put(("STOP", -1))
            except Exception:
                pass

        join_timeout = 30.0
        for p in self.islands:
            p.join(timeout=join_timeout)
            if p.is_alive():
                self.logger.warning(
                    "island %s did not exit within %.0fs; terminating",
                    p.name,
                    join_timeout,
                )
                p.terminate()
                p.join(timeout=10.0)
                if p.is_alive():
                    self.logger.warning("island %s still alive; killing", p.name)
                    try:
                        p.kill()
                    except Exception:
                        pass
                    p.join(timeout=5.0)
            self.logger.info(
                "island %s exited (exitcode=%s)", p.name, p.exitcode
            )

        # Give the listener a moment to drain any final messages.
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=5.0)

    def _finalize(self) -> int:
        """Write summary.txt and verify the winner (if any). Return exit code."""
        duration = time.monotonic() - self.state.start_time
        winner = self.state.winner
        status: str
        exit_code: int
        if winner is not None:
            status = "success"
            exit_code = 0
            try:
                verify_and_save_winner(
                    winner_state=winner.state,
                    config=self.cfg,
                    winner_meta={
                        "island_id": winner.island_id,
                        "iter_index": winner.iter_index,
                        "reported_score": winner.score,
                    },
                    run_dir=self.run_dir,
                )
                self.logger.info("winner verification complete")
            except Exception:
                self.logger.exception("winner verification FAILED")
                status = "success_but_verification_failed"
                exit_code = 2
        elif self.state.failed_islands and not self.state.done_islands:
            status = "error_all_islands_failed"
            exit_code = 1
        elif self.state.done_islands and not self.state.failed_islands:
            status = "no_counterexample_found"
            exit_code = 1
        elif self.state.n_active == 0:
            status = "mixed_failed_and_done_no_counterexample"
            exit_code = 1
        else:
            status = "timeout"
            exit_code = 1

        summary = {
            "status": status,
            "duration_seconds": round(duration, 2),
            "islands_started": self.cfg.parallelism.n_islands,
            "islands_failed": sorted(self.state.failed_islands.keys()),
            "islands_done": {
                str(k): v for k, v in sorted(self.state.done_islands.items())
            },
            "n_islands_active_at_end": self.state.n_active,
            "last_iter_by_island": dict(self.state.last_iter_by_island),
            "winner": (
                None
                if winner is None
                else {
                    "island_id": winner.island_id,
                    "iter_index": winner.iter_index,
                    "score": winner.score,
                }
            ),
            "config_path": str((self.run_dir / "config.yaml").resolve()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        self.logger.info("summary written to %s", summary_path)
        self.logger.info("run complete | status=%s exit_code=%d", status, exit_code)
        return exit_code


def run_coordinator(config: Config, run_dir: Path) -> int:
    """Top-level entry point used by ``run.py``. Wraps Coordinator.run() with
    a top-level try/except so any uncaught crash is logged before exit."""
    setup_logging(run_dir / "coordinator.log", mirror_stdout=config.logging.stdout_mirror)
    logger = logging.getLogger("coordinator")
    logger.info("=" * 70)
    logger.info("coordinator starting in %s", run_dir)
    logger.info("config dump:\n%s", config.dump_yaml())
    try:
        coord = Coordinator(config, run_dir)
        return coord.run()
    except BaseException:
        tb = traceback.format_exc()
        logger.critical("coordinator crashed:\n%s", tb)
        return 3
