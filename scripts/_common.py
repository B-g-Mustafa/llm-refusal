"""Shared helpers for the stage scripts: paths, seeding, IO, splits, progress logging."""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = REPO_ROOT / "logs"


def results_dir(model_key: str) -> Path:
    path = RESULTS_DIR / model_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def split_base_ids(base_ids: list[str], frac_train: float, seed: int) -> tuple[set[str], set[str]]:
    """Deterministic base-task split so extraction and evaluation never overlap."""
    unique = sorted(set(base_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    cut = int(len(unique) * frac_train)
    return set(unique[:cut]), set(unique[cut:])


# --------------------------------------------------------------------------- #
# Progress logging.
#
# The cluster's own PBS/qsub log (system-logs/*.o<jobid>, *.e<jobid>) is only
# written to disk once the *entire* job finishes, so `qstat`/`tail`-watching a
# running job shows nothing until it's over. ProgressLogger writes its own file
# under logs/<script_name>[_<model_key>].log, opened line-buffered and fsync'd
# after every write, so `tail -f logs/03_extract_directions_qwen3-14b.log` shows
# progress live regardless of how the scheduler buffers its own output.
# --------------------------------------------------------------------------- #


class ProgressLogger:
    """Writes timestamped lines to stdout and to a flushed file in logs/."""

    def __init__(self, name: str, model_key: str | None = None):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = f"_{model_key}" if model_key else ""
        self.path = LOGS_DIR / f"{name}{suffix}.log"
        self._fh = self.path.open("a", buffering=1, encoding="utf-8")
        self("=" * 70)
        self(f"logger started (pid={os.getpid()}, log file={self.path})")

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            pass  # some filesystems (e.g. certain network mounts) reject fsync; safe to skip

    def progress(self, done: int, total: int, every: int = 1, prefix: str = "") -> None:
        """Log every ``every``-th step plus the first/last, to avoid log spam."""
        if done == total or done == 1 or done % every == 0:
            pct = 100.0 * done / max(total, 1)
            label = f"{prefix} " if prefix else ""
            self(f"{label}progress {done}/{total} ({pct:.1f}%)")

    def close(self) -> None:
        self._fh.close()


def get_logger(script_name: str, model_key: str | None = None) -> ProgressLogger:
    return ProgressLogger(script_name, model_key)


@contextmanager
def stage(log: ProgressLogger, name: str):
    """Wrap one major step: logs START/DONE with elapsed time, or FAILED + traceback."""
    log(f"START  {name}")
    t0 = time.time()
    try:
        yield
    except Exception:
        log(f"FAILED {name} after {time.time() - t0:.1f}s")
        for line in traceback.format_exc().splitlines():
            log(f"    {line}")
        raise
    else:
        log(f"DONE   {name} ({time.time() - t0:.1f}s)")


@contextmanager
def script_run(script_name: str, model_key: str | None = None):
    """Top-level wrapper for a stage script's main(): yields a ProgressLogger,
    logs total wall time, and guarantees a FAILED line (with traceback) lands in
    the log file before the exception propagates and the job exits non-zero.
    """
    log = get_logger(script_name, model_key)
    t0 = time.time()
    try:
        yield log
    except Exception:
        log(f"FAILED {script_name} after {time.time() - t0:.1f}s total")
        for line in traceback.format_exc().splitlines():
            log(f"    {line}")
        log.close()
        raise
    else:
        log(f"DONE   {script_name} ({time.time() - t0:.1f}s total)")
        log.close()
