"""graph_ce package.

CRITICAL: This module sets BLAS/torch thread environment variables on import,
BEFORE any other graph_ce module (which would transitively import torch / numpy)
has a chance to load. Once torch loads it caches the thread count from the
environment, so setting these vars later has no effect.

With 16 island processes × 8 pool workers all running torch + numpy on CPU,
unrestricted BLAS threading would mean each of ~144 processes spawns ~128
threads, causing catastrophic oversubscription on a 128-core box. We saw
~150s per CEM iteration before pinning; ~2s expected after.
"""
from __future__ import annotations

import os as _os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

__version__ = "0.1.0"
