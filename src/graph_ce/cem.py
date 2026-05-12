"""Single-island Cross-Entropy Method step.

Each island owns one ``IslandCEM`` instance. ``step()`` runs one CEM
iteration: sample → score (via a process Pool) → percentile-filter →
update super-elite pool → supervised training pass on elite tuples.

Migration is handled by ``inject_migration_elites`` which adds external
high-scoring states to the super-elite pool before the next ``step()``.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from multiprocessing.pool import Pool

import numpy as np
import torch

from .config import Config
from .model import PolicyMLP
from .sampler import build_training_tuples, sample_batch
from .score import score_chunk, score_graph

logger = logging.getLogger(__name__)

_TIE_EPS: float = 1e-7


def select_by_percentile_with_tiebreak(
    scores: np.ndarray,
    percentile: float,
    budget: int,
) -> np.ndarray:
    """Percentile selection with explicit tie cap, following Adam Wagner's
    reference implementation.

    Returns the indices of selected sessions. Mirrors the logic in Adam
    Wagner's ``select_elites`` / ``select_super_sessions``:

    * Sessions with score strictly above the percentile threshold are always
      included.
    * Sessions at the threshold (within ``_TIE_EPS``) are included only while
      there is budget remaining, in iteration order.

    Without this cap, many sessions tying at the threshold flood the elite
    pool and over-reinforce the policy onto the local optimum.
    """
    n = scores.shape[0]
    if n == 0 or budget < 0:
        return np.empty(0, dtype=np.int64)
    threshold = float(np.percentile(scores, percentile))
    counter = int(budget)
    selected: list[int] = []
    for i in range(n):
        s = float(scores[i])
        if s >= threshold - _TIE_EPS:
            if counter > 0 or s > threshold + _TIE_EPS:
                selected.append(i)
            counter -= 1
    return np.asarray(selected, dtype=np.int64)


@dataclass
class CemStepResult:
    iter_index: int
    best_score: float
    best_state: np.ndarray             # int8 (num_edges,)
    best_lambda1: float                # NaN if best graph is disconnected
    best_mu: int                       # 0 if best graph is disconnected
    best_connected: bool
    mean_score: float
    elite_mean_score: float
    n_elites: int
    n_super_elites: int
    mean_train_loss: float
    step_seconds: float


class IslandCEM:
    def __init__(
        self,
        config: Config,
        island_id: int,
        pool: Pool,
        seed: int,
    ) -> None:
        self.cfg = config
        self.island_id = island_id
        self.pool = pool
        self.num_edges = config.problem.num_edges
        self.n = config.problem.n

        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(seed)
        torch.manual_seed(seed)  # also seeds default RNG used by parameter init
        self.np_rng = np.random.default_rng(seed)

        self.model = PolicyMLP(
            self.num_edges,
            config.model.hidden_sizes,
            init=config.model.init,
        )
        if config.model.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.model.learning_rate)
        elif config.model.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.model.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {config.model.optimizer}")
        self.bce = torch.nn.BCEWithLogitsLoss()

        # Super-elite carryover: states + scores survive into the next iteration.
        self.super_states: np.ndarray = np.zeros((0, self.num_edges), dtype=np.int8)
        self.super_scores: np.ndarray = np.zeros((0,), dtype=np.float64)

        self.iter_index = 0

    def _score_in_pool(self, states: np.ndarray) -> np.ndarray:
        """Score a (n_sessions, num_edges) batch across pool workers."""
        n_workers = self.cfg.parallelism.cores_per_island
        chunks = np.array_split(states, n_workers)
        chunk_args = [(chunk, self.n) for chunk in chunks if chunk.shape[0] > 0]
        chunk_results = self.pool.map(score_chunk, chunk_args)
        scores = np.empty(states.shape[0], dtype=np.float64)
        idx = 0
        for chunk_result in chunk_results:
            for r in chunk_result:
                scores[idx] = r.score
                idx += 1
        return scores

    def step(self) -> CemStepResult:
        t0 = time.monotonic()
        self.iter_index += 1

        # 1. Sample n_sessions graphs from the current policy.
        new_states = sample_batch(
            self.model,
            self.cfg.cem.n_sessions,
            self.num_edges,
            generator=self.torch_rng,
        )

        # 2. Score them in parallel.
        new_scores = self._score_in_pool(new_states)

        # 3. Pool new samples with super-elites carried from previous iter
        #    (and possibly migrants injected since the last step).
        all_states = np.concatenate([new_states, self.super_states], axis=0)
        all_scores = np.concatenate([new_scores, self.super_scores], axis=0)

        # 4. Percentile-based elite and super-elite filtering with Wagner's
        #    tie cap. Budgets are fixed (computed from n_sessions, not the
        #    combined pool size) so the pool can never balloon, even when many
        #    sessions land exactly on the percentile threshold.
        elite_budget = int(self.cfg.cem.n_sessions * (100.0 - self.cfg.cem.elite_percentile) / 100.0)
        super_budget = int(self.cfg.cem.n_sessions * (100.0 - self.cfg.cem.super_elite_percentile) / 100.0)
        elite_idx = select_by_percentile_with_tiebreak(
            all_scores, self.cfg.cem.elite_percentile, elite_budget
        )
        super_idx = select_by_percentile_with_tiebreak(
            all_scores, self.cfg.cem.super_elite_percentile, super_budget
        )

        elite_states = all_states[elite_idx]
        self.super_states = all_states[super_idx].copy()
        self.super_scores = all_scores[super_idx].copy()

        # 5. Supervised training on elite tuples.
        mean_loss = self._train_on_elites(elite_states)

        # Best overall this iteration (could come from super pool too).
        best_idx = int(np.argmax(all_scores))
        best_score = float(all_scores[best_idx])
        best_state = all_states[best_idx].astype(np.int8, copy=True)
        # Re-score the best to get lambda_1 and mu cheaply (1-2 ms at n=19).
        best_result = score_graph(best_state, self.n)
        mean_score = float(new_scores.mean()) if new_scores.size else float("nan")
        elite_mean = (
            float(all_scores[elite_idx].mean()) if elite_idx.size else float("nan")
        )

        result = CemStepResult(
            iter_index=self.iter_index,
            best_score=best_score,
            best_state=best_state,
            best_lambda1=float(best_result.lambda1),
            best_mu=int(best_result.mu),
            best_connected=bool(best_result.connected),
            mean_score=mean_score,
            elite_mean_score=elite_mean,
            n_elites=int(elite_idx.size),
            n_super_elites=int(super_idx.size),
            mean_train_loss=mean_loss,
            step_seconds=time.monotonic() - t0,
        )
        return result

    def _train_on_elites(self, elite_states: np.ndarray) -> float:
        if elite_states.shape[0] == 0:
            return float("nan")
        inputs, targets = build_training_tuples(elite_states, self.num_edges)
        inputs_t = torch.from_numpy(inputs)
        targets_t = torch.from_numpy(targets)

        self.model.train()
        n_examples = inputs_t.shape[0]
        batch_size = self.cfg.cem.train_batch_size
        total_loss = 0.0
        total_batches = 0
        for _ in range(self.cfg.cem.train_epochs_per_iter):
            perm = torch.randperm(n_examples, generator=self.torch_rng)
            for start in range(0, n_examples, batch_size):
                idx = perm[start : start + batch_size]
                x = inputs_t[idx]
                y = targets_t[idx]
                logits = self.model.forward(x).squeeze(-1)
                loss = self.bce(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_batches += 1
        return total_loss / max(total_batches, 1)

    def inject_migration_elites(
        self,
        states: np.ndarray,
        scores: np.ndarray,
    ) -> None:
        """Add migrants to the super-elite pool. They'll compete for survival
        on the next ``step()`` like any other super-elite."""
        if states.size == 0:
            return
        if states.shape[1] != self.num_edges:
            raise ValueError(
                f"migrant states have wrong width {states.shape[1]} != {self.num_edges}"
            )
        self.super_states = np.concatenate([self.super_states, states.astype(np.int8)], axis=0)
        self.super_scores = np.concatenate([self.super_scores, scores.astype(np.float64)], axis=0)
        logger.info(
            "island=%d injected %d migrants into super-elite pool (pool size now %d)",
            self.island_id,
            states.shape[0],
            self.super_states.shape[0],
        )

    def current_top_k(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the top-k super-elite (states, scores) for migration export."""
        if self.super_states.shape[0] == 0:
            return (
                np.zeros((0, self.num_edges), dtype=np.int8),
                np.zeros((0,), dtype=np.float64),
            )
        k = min(k, self.super_states.shape[0])
        order = np.argsort(-self.super_scores)[:k]
        return self.super_states[order].copy(), self.super_scores[order].copy()
