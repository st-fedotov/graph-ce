# graph-ce

A modern PyTorch reimplementation of Adam Zsolt Wagner's cross-entropy method
(CEM) for finding counterexamples to **Conjecture 2.1** from
[*Constructions in combinatorics via neural networks*](https://arxiv.org/abs/2104.14516):

> For every connected graph G on n vertices,
> &nbsp;&nbsp;λ₁(G) + μ(G) ≥ √(n − 1) + 1
> &nbsp;&nbsp;where λ₁ is the largest adjacency eigenvalue and μ is the size of a maximum matching.

The conjecture is false at n = 19. This codebase rediscovers a counterexample
in **about 10 minutes** of wall time on a 128-core CPU server (no GPU
required), with 16 parallel CEM islands.

## Result

```
status:  success
winner:  island 9, iter 1411, score = +0.0804
λ₁     = 3.1623  (= √10)
μ      = 2
λ₁ + μ = 5.1623 < √18 + 1 ≈ 5.2426
duration: 10.7 min on 128 CPU cores (no GPU)
```

The graph is a *double broom*: two stars (9 leaves each) joined through a
shared "bridge" vertex of degree 2. Wagner's paper highlights the same
structural family.

## Trajectories across configurations

![Top-3 island trajectories for six setups](plots/setup_comparison.png)

Each color is one configuration, each color shows 3 island trajectories
(same style — overlapping means migration synced them). The two configs
that cross the dotted victory line at score = 0 (purple and brown) are the
two correct runs. The bad-init configs (green, blue, orange) flatline well
below 0 regardless of batch size or migration — that's the "stuck-in-local-
optimum" signature.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Launch the n=19 search detached from your shell. Survives SSH disconnect,
# Claude Code session timeouts, etc.
scripts/launch_detached.sh n19_main

# Watch progress (re-runnable, defaults to runs/n19_main/).
.venv/bin/graph-ce-status

# Stop cleanly (the coordinator catches SIGTERM and tears down all 16
# island processes gracefully):
kill "$(cat runs/n19_main.pid)"
```

When a counterexample is found the coordinator writes
`runs/<run_name>/winner_certificate.txt` and exits.

## What's in the box

```
configs/default.yaml      every tunable — graph size, MLP shape, CEM percentiles,
                          parallelism, migration cadence, logging, RNG seed.
                          Nothing in code hardcodes these.

src/graph_ce/
  config.py               pydantic loader + override mechanism
  model.py                PolicyMLP (342 → 128 → 64 → 4 → 1, sigmoid)
  score.py                sqrt(n-1)+1 − λ₁ − μ (or -INF if disconnected)
  matching.py             pure-Python Edmonds blossom (~24× faster than NetworkX
                          on n=19, validated by stress test in tests/)
  sampler.py              vectorized autoregressive bit-by-bit generation
  cem.py                  one CEM iteration (sample → score → elite → train),
                          with Wagner's percentile tie-cap
  island.py               long-running worker: own MLP, own RNG, own score Pool
  coordinator.py          master process: 16 islands, migration broker,
                          liveness monitor, signal-safe graceful shutdown
  verify.py               independently re-scores winners and writes the
                          certificate
  status.py               graph-ce-status: at-a-glance run health
  plot_trajectories.py    graph-ce-plot: trajectory comparison figure
  run.py                  graph-ce CLI entry

scripts/
  launch_detached.sh      setsid+nohup launcher — runs survive harness/session
                          cleanup; writes runs/<name>.pid for `kill` later

tests/                    65 unit tests (matching ↔ NetworkX oracle, sampling
                          determinism, elite tie-cap, init schemes, smoke E2E)
```

## What "good init" means — and why it matters so much

The single change that turned the project from "stuck forever at score
-1.5" into "finds the counterexample in 10 minutes":

**PyTorch's `nn.Linear` default initialization** is Kaiming uniform with
`a=sqrt(5)` for weights, plus a **non-zero uniform bias** in
`[−1/√fan_in, +1/√fan_in]`.

**Keras' `Dense` default**, which Wagner relies on, is **Glorot uniform**
for weights with **zero biases**.

The difference looks cosmetic. It is not. With zero biases the initial
sigmoid output is unbiased around 0.5, so the initial CEM policy is
honest Bernoulli(0.5) over each edge bit. With PyTorch's non-zero biases,
four stacked layers shift the initial logit away from 0, the starting
policy is biased toward a particular family of bit strings, and CEM —
being a policy-improvement method whose entire signal is "do more of what
the elites did" — happily reinforces the bias and locks into a local
optimum from which it cannot escape.

The `model.init` config field toggles between the two:

```yaml
model:
  init: keras            # Glorot uniform weights + zero bias.   Default.
  # init: pytorch_default  # PyTorch nn.Linear default. Reproduces the bug.
```

We recommend leaving it on `keras`. The `pytorch_default` mode is kept so
the failure mode can be replicated; the plot above used both.

## Other gotchas worth knowing

1. **BLAS thread oversubscription**. `OMP_NUM_THREADS=1` and friends *must*
   be set before any module imports numpy/torch — env vars set after import
   have no effect. We pin them in `graph_ce/__init__.py` so they fire on
   any first import of the package. Without this pin, 16 islands × 8 score
   workers × 128 BLAS threads each thrashed the 128-core box; per-iter time
   was ~155s instead of ~0.3s. Also `torch.set_num_threads(1)` for belt
   and suspenders.

2. **mp.Event in signal handlers deadlocks**. `multiprocessing.Event.set()`
   acquires an internal `Lock` that the main thread already holds inside
   `mp.Event.wait()`. If a SIGTERM handler calls `.set()`, the handler
   deadlocks; the process can no longer be stopped except by SIGKILL. Our
   handler flips a plain Python `bool`; the main thread polls that flag in
   short slices and calls `.set()` safely from its own context.

3. **Elite-selection ties bloat the pool**. A plain
   `scores >= np.percentile(scores, 93)` admits every session tied at the
   threshold. Once CEM converges, many sessions land *exactly* at the
   threshold, the elite pool balloons (we saw 2168 instead of ~70), and
   training over-reinforces the current local optimum. Wagner's reference
   uses a counter-based tie cap; we ported that logic to
   `graph_ce.cem.select_by_percentile_with_tiebreak`. Without it, migration
   is actively destructive.

4. **Batch size matters but only after the init is right**. Keras'
   `model.fit()` default is `batch_size=32`; we initially used 512. With
   the init bug, switching to 32 didn't help; both still plateaued. With
   the init fix, b=32 reliably finds the counterexample in a few thousand
   iters and b=512 grinds for 6× the compute without converging. Our
   default is now 32.

5. **NetworkX's matching is slow on small graphs**. `nx.max_weight_matching`
   takes ~3.6 ms per 19-vertex graph in pure-Python overhead; the
   actual blossom algorithm should take microseconds. We rewrote Edmonds'
   blossom directly (`graph_ce.matching.max_cardinality_matching`) for a
   ~24× speedup. The replacement is randomized-stress-tested against
   NetworkX on hundreds of graphs at production size.

## Adapting the reward for a different graph conjecture

The CEM machinery — sampler, model, training, parallelism, migration,
logging, graceful shutdown, the whole island/coordinator setup — is
**generic over the score function**. To target a different conjecture
about simple undirected graphs on n vertices, you'll only need to edit
two-and-a-half files (typically 30–60 minutes of work):

### What's reusable as-is

- Bit-string → symmetric adjacency matrix (`score.build_adjacency`).
- Connectedness check via BFS (`score.is_connected`).
- Eigenvalue computation (`numpy.linalg.eigvalsh`).
- Maximum-cardinality matching via custom Edmonds blossom
  (`matching.max_cardinality_matching`, ~24× faster than NetworkX).
- The full CEM loop, all parallelism, all logging.

### What you'd change

1. **`src/graph_ce/score.py`**: rewrite `score_graph` to compute and return
   your conjecture's reward. The skeleton is:

   ```python
   def score_graph(bits: np.ndarray, n: int) -> ScoreResult:
       A = build_adjacency(bits, n)
       if not is_connected(A):                  # drop this line if your
           return ScoreResult(                  # conjecture allows
               score=DISCONNECTED_PENALTY,      # disconnected graphs
               ...,
           )
       # Compute whichever invariants your conjecture needs.
       # Examples already wired up: eigvalsh(A), matching_number(A).
       # Other invariants you might add: chromatic number, independence
       # number, diameter, girth, vertex cover, etc.
       my_score = your_threshold(n) - your_invariants(A)
       return ScoreResult(score=my_score, connected=True, ...)
   ```

   A score `> 0` is interpreted everywhere as "counterexample found", so
   write your reward so that you want to *maximize* it and victory means
   "strictly positive".

2. **`src/graph_ce/verify.py`**: update `verify_and_save_winner`'s
   certificate text to print whichever invariants your `score_graph`
   computes (so the human-readable proof shows the right numbers).

3. **`src/graph_ce/config.py`**: if your conjecture has a closed-form RHS
   in terms of `n`, replace the `ProblemConfig.conjecture_threshold`
   property; if not, just delete the property.

That's it. The MLP shape, batch size, percentiles, optimizer, migration,
all stay the same. Adam Wagner's paper applies CEM the same way to a
dozen conjectures with different score functions — only the score
function ever changes.

### What's *not* easy to swap

- **Directed graphs** (the bit-encoding length doubles from `n(n−1)/2` to
  `n(n−1)`; sampling and score input change accordingly).
- **Weighted edges** (the alphabet stops being binary; Wagner has a short
  note on this — you'd switch the model's output to softmax over k
  categories and use cross-entropy loss).
- **Variable `n` within one run** (we treat `n` as a single config value).
- **Multi-graph encodings, hypergraphs, or anything beyond simple graphs**
  (the encoding layer would need substantial extension).

If you adapt this for another conjecture from Adam Wagner's paper, you're
encouraged to keep your fork local — this repository is intentionally
scoped to Conjecture 2.1.

## Architecture

```
coordinator (1 process)
├── spawns 16 island processes
├── listener thread: PROGRESS | MIGRATE_OUT | SUCCESS | FAILED | DONE
├── liveness thread: detects islands that die without sending a terminal
│                    status (with a grace period for in-flight messages)
├── main loop: polls signal flag + stop_event in 1 s slices
└── on stop_event: drain queues, join islands, write summary.json

island_i (1 process, ×16)
├── own torch MLP, optimizer, RNG (seeded master_seed + i)
├── own multiprocessing.Pool(8) for score eval
├── CEM loop: sample(1000) → score → percentile-tie-cap elite filter →
│             train(1 epoch, BCE, b=32) → maybe migrate
└── on counterexample: notify coordinator, exit
```

128 cores total = 16 island main processes + 16 × 8 score workers, all
pinned to 1 BLAS thread per process.

## Reproducing the published-counterexample search

The default config is exactly the configuration that won. Running
`scripts/launch_detached.sh n19_main` and waiting should produce a winner
within roughly 10–30 minutes (CEM is noisy, single-run variance is high).
With migration **off** and everything else default, expect ~20 minutes.

To reproduce the *failure modes* described above (e.g. for a blog post or
to teach CEM gotchas), override the config:

```bash
# Reproduce the PyTorch-default init plateau:
scripts/launch_detached.sh n19_bad_init \
    --override model.init=pytorch_default
```

## Tests

```bash
.venv/bin/pytest
```

65 tests in ~2.5 seconds. Includes a randomized stress test that compares
our blossom matching against NetworkX on 330 random n=19 graphs at three
densities, and a tiny end-to-end smoke test that runs the full CEM step
without subprocesses.

## Reference and citation

The algorithm, the conjecture, the score function, and the original code
this repo is reproducing are all due to **Adam Zsolt Wagner**:

> Wagner, A. Z. *Constructions in combinatorics via neural networks*.
> arXiv preprint arXiv:2104.14516, 2021.

Original repository (TensorFlow 1.14 + Keras 2.3.1): https://github.com/zawagner22/cross-entropy-for-combinatorics

```bibtex
@article{wagner2021constructions,
  title   = {Constructions in combinatorics via neural networks},
  author  = {Wagner, Adam Zsolt},
  journal = {arXiv preprint arXiv:2104.14516},
  year    = {2021}
}
```

If you use this codebase, please cite Wagner's paper. This repository is
an engineering port — every interesting idea is his.
