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
shared "bridge" vertex of degree 2. Adam Wagner's paper highlights the same
structural family.

![9/9 symmetric double broom: λ₁=√10, μ=2, score=+0.0804](plots/winner_9_9_double_broom.png)

A second, slightly tighter counterexample falls out of the no-migration
configuration — an 8/10 split with score +0.0155:

![8/10 asymmetric double broom: λ₁=3.227, μ=2, score=+0.0155](plots/winner_8_10_double_broom.png)

Both are reproducible from saved adjacency matrices via
`graph-ce-plot-winner runs/n19_success` (and similarly for the
`runs/n19_success_with_migration` artifact tree).

## Trajectories across configurations

![Top-3 island trajectories for six setups](plots/setup_comparison.png)

Each color is one configuration, each color shows 3 island trajectories
(same style — overlapping means migration synced them). The two configs
that cross the dotted victory line at score = 0 (purple and brown) are the
two correct runs. The bad-init configs (green, blue, orange) flatline well
below 0 regardless of batch size or migration — that's the "stuck-in-local-
optimum" signature.

## What the failure looks like — anatomy of the bad-init plateau

The three bad-init runs (v3/v4/v5 above) each have 16 parallel islands.
The natural question: when those runs plateau, what graph is each island
*stuck on*? Are the 16 islands all stuck on the same graph, or different
ones? And how stable is the stuck state — does it fluctuate or freeze
solid?

We answered this by reading each island's per-iteration metrics file
(every iter records the best graph seen so far), partitioning the 16
final graphs into graph-isomorphism classes, and timing when each
island's best graph last changed. The probe lives at
`scripts/explore_plateau_topology.py`.

### With migration: all 16 islands collapse to the same tree

![One tree — v3 (PyTorch init + migration, b=512)](plots/plateau_trees_n19_main_plateau_v3.png)

In `n19_main_plateau_v3` (PyTorch init, migration on, b=512), all 16
islands converge to **the same tree**, up to graph isomorphism, and showed no further improvement.

### Without migration: every island finds its own local optimum

![16 distinct graphs — v4 (PyTorch init, no migration, b=512)](plots/plateau_trees_n19_main_plateau_v4.png)

In `n19_main_plateau_v4` (PyTorch init, no migration, b=512), every island
discovers a unique local optimum. The best one (island 01) has μ = 4 and
score −1.512 — a long thin tree with a degree-7 hub; the rest cluster
around μ = 5, λ₁ between 2.39 and 3.03.

![16 distinct graphs — v5 (PyTorch init, no migration, b=32)](plots/plateau_trees_n19_main_plateau_v5.png)

`n19_main_plateau_v5` (PyTorch init, no migration, b=32) shows the same
story: 16 unique non-isomorphic graphs. Some islands froze as early as
iter 1255 of 10500 and produced no further improvement across the
remaining 88% of the run.

### What the basin looks like

The shared signature across all three bad-init runs:

- **Sparse**: 18–20 edges (the counterexample we eventually want is
  also 18 edges, so density alone is not the giveaway).
- **Mostly trees, sometimes with one or two short cycles.**
- **μ = 5** (with one μ = 4 outlier).
- **λ₁ ≈ 2.4–3.0.**
- **Two hubs** of degree 4–6, a few degree-2 internal vertices, lots of
  degree-1 leaves — a "small spider" shape.

The counterexample we want is also a tree, but a *very* different one:
a *double broom* (long path of degree-2 vertices, with a bundle of
leaves at each end), with μ = 2 and λ₁ > √18 ≈ 4.24. To get from a
small two-hub spider to a double broom requires reshaping many edges
simultaneously, which a per-edge Bernoulli policy cannot do in one or
two CEM steps. The bad-init basin is therefore both *deep* (small
perturbations always score worse) and *far* (many edges separate it
from the counterexample basin). Together that explains the flatline.

You can reproduce the analysis with:

```bash
.venv/bin/python scripts/explore_plateau_topology.py \
    runs/n19_main_plateau_v3 runs/n19_main_plateau_v4 runs/n19_main_plateau_v5 \
    --mode all-classes \
    --output plots/plateau_trees.png
```

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

## Configuration

Every tunable lives in `configs/default.yaml`. Nothing in the code
hardcodes any of them. The shipped defaults are what won on a 128-core
box; the only knobs you'll typically touch are `parallelism.*` (CPU
budget) and `seed.master_seed` (reproducibility). Everything else is
faithful to Adam Wagner's setup and should stay put unless you're doing
an ablation.

### Matching parallelism to your CPU count

The total number of CPU-active processes during scoring is:

&nbsp;&nbsp;&nbsp;&nbsp;`parallelism.n_islands × parallelism.cores_per_island`

Set this product to your physical core count. The defaults are tuned
for **128 cores** (16 × 8). If your machine differs, here are sane
re-balancings:

| Cores | `n_islands` | `cores_per_island` | Notes |
|------:|------------:|-------------------:|-------|
|     8 |           4 |                  2 | Tight; expect 30–60 min per attempt. |
|    16 |           8 |                  2 | A laptop-class run. |
|    32 |           8 |                  4 | A modest workstation. |
|    64 |          16 |                  4 | Same seed diversity as production, half the throughput. |
|   128 |          16 |                  8 | **Default — what found the counterexample in 10 min.** |
|   256 |          32 |                  8 | More seeds in parallel, ~same per-island speed. |

Two principles when sizing:

1. **More islands = more independent seeds = higher chance one of them gets
   lucky.** Adam Wagner reports ~30% per-seed success; with 16 islands
   that compounds to ~99.5% (with migration on, the practical hit rate is
   higher still — the lucky seed shares its discovery with the others).
   Don't drop below ~4 islands unless you really have to.

2. **More cores per island ≠ much faster per iteration.** Per-iter time
   is dominated by the autoregressive sampling step (171 forward passes
   through a tiny MLP, single-threaded by design — see the BLAS-pinning
   note below), not by score evaluation. So cores beyond ~4 per island
   give diminishing returns. Prefer adding islands to adding cores.

Plus one hard rule: **never let `n_islands × cores_per_island` exceed your
physical core count**, even by a little. Once total processes go over
core count the BLAS-thread-pinning trick we rely on (everyone gets
exactly one core) breaks, and per-iter time can blow up by ~500× into
context-thrash hell. We've seen this happen.

### Other knobs (in roughly decreasing order of "actually useful to tune")

| Knob | Default | What it does |
|------|---------|--------------|
| `seed.master_seed` | `null` | `null` means time-based, logged. Set an integer to reproduce a specific run exactly. Each island gets `master_seed + island_id`. |
| `migration.enabled` | `true` | Off by default in the no-mig ablation; on for production. With the tie-cap in place, migration is roughly a 2× speedup. |
| `migration.interval_iters` | `50` | Iterations between migration syncs. Lower = more sharing, less seed diversity. Higher = closer to no-migration. |
| `migration.top_k` | `50` | Number of elites broadcast to every island at each sync. |
| `stopping.wall_clock_seconds` | `43200` | Hard 12 h budget. Drop to e.g. `1800` for a "try 30 min and bail" run. |
| `cem.max_iters` | `100000` | Per-island iteration cap. Defaults are effectively unbounded; wall clock is what stops you. |
| `logging.log_interval_iters` | `10` | How often each island writes a human-readable progress line to its log. |
| `logging.metrics_interval_iters` | `1` | How often the per-iter JSONL row is written. Includes the iteration's best graph; leave at 1 unless disk space is a concern. |
| `problem.n` | `19` | Graph size. The n=19 counterexample is the famous one; smaller `n` (e.g. 5–10) is useful for smoke testing. |

### Things you should not touch unless you know why

- `model.hidden_sizes`, `model.learning_rate`, `model.optimizer` — Adam
  Wagner's exact values. Deviating without an ablation is almost
  certainly worse.
- `cem.n_sessions`, `cem.elite_percentile`, `cem.super_elite_percentile`,
  `cem.train_batch_size`, `cem.train_epochs_per_iter` — same.
- `parallelism.start_method` — leave at `spawn`. `fork` shares torch
  state across processes and produces deeply weird bugs.
- `model.init` — `keras` is the right choice and the only setting under
  which we've ever found a counterexample. `pytorch_default` is kept as
  an ablation toggle to reproduce the plateau failure mode; see the
  "What 'good init' means" section.

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
`[−1/√d_in, +1/√d_in]`.

**Keras' `Dense` default**, which Wagner relies on, is **Glorot uniform**
for weights with **zero biases**.

The difference looks cosmetic. It is not. We sampled 500 graphs from a
freshly-initialized PolicyMLP under each scheme, repeated across 8 model
seeds, and measured the per-position `P(bit=1)` after feeding the
all-zero state at each of the 171 positions.

![Per-position P(bit=1) heatmap across 8 seeds × 2 inits](plots/init_position_bias.png)

The heatmap rows tell most of the story:

- **Keras rows are uniformly near-white.** Each seed, each of the 171
  positions: `P(bit=1)` ≈ 0.5. The policy starts as honest
  Bernoulli(0.5) — exactly the uniform prior CEM expects.
- **PyTorch-default rows are solid-coloured horizontal stripes.** Each
  individual seed is essentially *constant* across all positions —
  some seeds are uniformly blue (P ≈ 0.41, prefer 0 → start sparse),
  others uniformly red (P ≈ 0.6, prefer 1 → start dense). The
  position-to-position variation within a seed is tiny next to the
  seed-to-seed shift.

Numerically:

```
init                |P-0.5|  mean     std    max      per-seed max
keras                          0.008  0.009  0.059    [.020 .048 .022 .027 .059 .019 .021 .015]
pytorch_default                0.065  0.035  0.119    [.062 .094 .087 .016 .080 .058 .014 .119]
```

The pytorch_default per-seed maxima reach **±0.12 from 0.5**: a roughly
8× larger average bias and a 2× larger worst-case bias than keras.

### Decomposing the cause: two orthogonal pathologies

Three diagnostic probes (`scripts/init_policy_distribution.py`,
`scripts/generation_trajectory_distribution.py`,
`scripts/activation_through_layers.py`) pin the failure on the init
*geometry* — not the activation, not the optimizer.

We instantiated fresh PolicyMLPs under four init schemes and measured
the iter-0 `P(bit=1)` statistics (mean and std *across the 171
positions*, with the blank state as input):

![Iter-0 policy distribution per init scheme](plots/init_policy_distribution_4way.png)

| init                          | P̄(blank) | σ(blank)   |
|-------------------------------|----------|------------|
| keras                         | 0.51     | **0.0083** |
| pytorch_default               | **0.61** | 0.0008     |
| pytorch_weights_zero_bias     | 0.50     | 0.0004     |
| xavier_weights_pytorch_bias   | 0.64     | **0.0119** |

Two pathologies decouple along orthogonal axes:

- **Non-zero biases** set the offset P̄ (PyTorch's uniform biases push
  it to ≈ 0.6).
- **Kaiming `a=√5` weights** are too small to give the policy useful
  input-dependence (σ collapses by ~10× compared with Xavier).

Each pathology alone might be tolerable; together they make the policy
a near-constant function of input. CEM has no per-position signal to
learn from, locks onto the sparsest few elites among Bernoulli(0.61)
samples, and ends up in the tree basin documented above.

### Autoregressive sampling doesn't rescue it

You might hope the conditional structure of the rollout creates
variation as the state fills in — by position 170 the state has
~100 active bits, after all. It doesn't:

![Autoregressive-rollout P(bit=1) trajectories at iter 0](plots/generation_trajectory_distribution.png)

The model still emits P ≈ 0.607 throughout the rollout. The median
curve is flat (σ across positions: 0.0018), the 10–90 percentile band
barely opens, and ReLU and LeakyReLU agree to 4 decimal places. The
weights are simply too small to respond to the growing state input.

### The mechanism: signal collapse, layer by layer

Feeding `(blank state, one-hot position p)` through the 4-layer stack
for each `p ∈ [0, 170]` and capturing each post-activation:

![Signal propagation through the policy MLP](plots/activation_through_layers.png)

| layer            | keras (alive/total, σ_pos) | pytorch_default              |
|------------------|----------------------------|------------------------------|
| hidden_1 (128)   | 128/128, σ = **0.036**     | 128/128, σ = **0.018**       |
| hidden_2 (64)    | 64/64,   σ = **0.023**     | **46/64**, σ = **0.007**     |
| hidden_3 (4)     | 4/4,     σ = **0.028**     | 4/4,     σ = **0.005**       |
| output P(bit=1)  | σ = **0.008**              | σ = **0.0008**               |

Three things to note:

1. **Signal halves at the very first layer** — Kaiming `a=√5` weights
   have bound `1/√fan_in`; Xavier has `√(6/(fan_in+fan_out))`, roughly
   2× larger for the first layer.
2. **18 of 64 layer-2 units die** under `pytorch_default`. They didn't
   die in layer 1 — with sparse one-hot inputs, the bias is the same
   magnitude as the single active weight, and only ~50% of units land
   on the wrong side of zero. By layer 2 every layer-1 unit passes a
   bias-shifted offset, and combined with layer-2 biases this pushes
   28% of layer-2 pre-activations into the always-negative half-plane.
3. The **4-unit bottleneck** at hidden_3 amplifies the per-unit
   collapse into a 10× output suppression.

### Why LeakyReLU doesn't help

The dying-layer-2-units detail suggests an obvious fix: swap ReLU for
LeakyReLU (negative slope 0.01) so units can't go fully dead. We tried
it — both as the same diagnostic and as a 16-min CEM run with
`activation=leaky_relu`.

![Same probe under LeakyReLU](plots/activation_through_layers_leaky.png)

| layer    | pytorch + ReLU         | pytorch + LeakyReLU    |
|----------|------------------------|------------------------|
| hidden_2 | **46/64**, σ = 0.0067  | **63/64**, σ = 0.0068  |
| output   | σ = **0.0008**         | σ = **0.0008**         |

LeakyReLU resurrects 17 of 18 dead units — but mean σ at that layer
is unchanged (0.0067 → 0.0068), and the output collapse is identical.
The newly "alive" units pass signal at 1% strength (the leaky slope);
they're alive in name only. The CEM run plateaued in the same `v5`
basin within 2300 iterations.

**The bottleneck isn't the rectification — it's the weight magnitudes.**
Kaiming `a=√5` is calibrated for dense, unit-variance inputs; our
inputs are sparse one-hot vectors where the bias has the same magnitude
as the single active weight. The cure is bigger weights, which is what
Xavier does.

The `model.init` and `model.activation` config fields expose all
schemes used above:

```yaml
model:
  init: keras            # Glorot uniform weights + zero bias.  Default.
  # init: pytorch_default                # reproduces the bug
  # init: pytorch_weights_zero_bias      # ablation: isolate the bias
  # init: xavier_weights_pytorch_bias    # ablation: isolate the weight scale
  activation: relu       # 'leaky_relu' available for ablation; doesn't fix pytorch_default.
```

We recommend leaving it on `keras` + `relu`. The other modes are kept
so the failure can be replicated and decomposed.

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
