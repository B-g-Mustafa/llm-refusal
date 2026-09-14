"""Shared helpers for the stage scripts: paths, seeding, IO, splits."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


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
