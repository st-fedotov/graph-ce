"""Per-process logging setup.

Each process (coordinator, island workers, score-pool workers) calls
``setup_logging`` once on startup to direct its log records to a file in the
run directory. The coordinator additionally mirrors to stdout.

We rely on POSIX append-mode atomicity for single-line writes < PIPE_BUF
(typically 4096 bytes) so that score-pool siblings sharing one island log
file don't tear each other's lines.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s [%(processName)s pid=%(process)d] %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_file: str | Path,
    *,
    mirror_stdout: bool = False,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure the root logger for this process.

    Safe to call from any process; each call replaces handlers on the root
    logger of that process only.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    if mirror_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        root.addHandler(stream_handler)

    return root


def pin_blas_threads(n_threads: int = 1) -> None:
    """Restrict BLAS/numpy/torch to ``n_threads`` threads per process.

    The env-var part is also done by ``graph_ce/__init__.py`` at the very top
    of package import so torch sees it at load time. We additionally call
    ``torch.set_num_threads`` here, which is effective at any time and pins
    the interpreter-side intra-op pool too.
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    try:
        import torch  # local import: this module must remain importable without torch
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(n_threads)
    except (ImportError, RuntimeError):
        # RuntimeError can fire if set_num_interop_threads is called after
        # the interop pool was already used; not fatal.
        pass
