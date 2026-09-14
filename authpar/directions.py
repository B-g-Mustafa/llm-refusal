"""Candidate directions and their geometric certificates.

Four directions, each estimated on disjoint data (PLAN.md "Candidate
representations"):

- ``d_harm``       : harmful vs. harmless instructions at ``t_inst``.
- ``d_refusal``    : refused vs. answered behavior at post-instruction positions.
- ``d_claim``      : authorization preamble vs. a matched *irrelevant* preamble on
                     identical tasks (template-controlled difference-in-differences:
                     using the irrelevant preamble as the control subtracts the
                     generic-preamble nuisance, leaving the authorization content).
- ``d_persuasion`` : persuasion/role-play wrappers vs. matched benign wrappers.

``d_claim`` is *not* pre-labelled a "manipulation" direction; whether it coincides
with ``d_persuasion`` or with ``d_harm`` is the empirical question (H1 vs. H2/H3).

Each direction carries the geometric statistics that C1 tests against causal
effect: raw difference-of-means norm, top-1 SVD energy, projection AUROC / Cohen's
d, and a cross-validated linear-probe AUROC. The scientific claim C1 is that these
certificates predict held-out causal control poorly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Direction:
    name: str
    layer_id: int
    position_spec: str
    vector: np.ndarray  # unit-norm [d_model]
    raw_norm: float
    scores: dict[str, float] = field(default_factory=dict)


def _unit(vec: np.ndarray) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        raise ValueError("Degenerate (near-zero) direction.")
    return vec / norm, norm


def _projection_auroc(pos: np.ndarray, neg: np.ndarray, direction: np.ndarray) -> float:
    """AUROC of separating pos from neg by projection onto ``direction``."""
    p = pos @ direction
    n = neg @ direction
    # Mann-Whitney U -> AUROC, no sklearn dependency.
    scores = np.concatenate([p, n])
    labels = np.concatenate([np.ones_like(p), np.zeros_like(n)])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _assign_tie_ranks(scores, ranks)
    n_pos, n_neg = len(p), len(n)
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(max(auc, 1 - auc))  # direction-agnostic separability


def _assign_tie_ranks(scores: np.ndarray, ranks: np.ndarray) -> None:
    order = np.argsort(scores, kind="mergesort")
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1


def _cohens_d(pos: np.ndarray, neg: np.ndarray, direction: np.ndarray) -> float:
    p = pos @ direction
    n = neg @ direction
    pooled = np.sqrt((p.var(ddof=1) + n.var(ddof=1)) / 2) + 1e-8
    return float(abs(p.mean() - n.mean()) / pooled)


def _probe_auroc(pos: np.ndarray, neg: np.ndarray, folds: int = 5) -> float:
    """Cross-validated logistic-probe AUROC (sklearn if present, else skip)."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return float("nan")
    X = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    if len(pos) < folds or len(neg) < folds:
        return float("nan")
    clf = LogisticRegression(max_iter=1000, C=1.0)
    proba = cross_val_predict(clf, X, y, cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def diff_in_means_direction(
    name: str,
    layer_id: int,
    position_spec: str,
    pos: np.ndarray,
    neg: np.ndarray,
    with_svd: bool = False,
) -> Direction:
    """Difference-of-means direction with geometric certificates.

    ``pos`` / ``neg`` are ``[n, d]`` activation matrices at one layer/position.
    With ``with_svd`` (paired case) also report top-1 SVD energy of the paired
    deltas; requires equal-length, row-aligned ``pos``/``neg``.
    """
    delta = pos.mean(axis=0) - neg.mean(axis=0)
    vector, raw_norm = _unit(delta)
    scores = {
        "raw_norm": raw_norm,
        "proj_auroc": _projection_auroc(pos, neg, vector),
        "cohens_d": _cohens_d(pos, neg, vector),
        "probe_auroc": _probe_auroc(pos, neg),
    }
    if with_svd and pos.shape == neg.shape:
        deltas = pos - neg
        _u, sv, _vh = np.linalg.svd(deltas, full_matrices=False)
        energy = float((sv[0] ** 2) / (np.square(sv).sum() + 1e-12))
        scores["svd_top1_energy"] = energy
    return Direction(name, layer_id, position_spec, vector, raw_norm, scores)


def random_direction(layer_id: int, d_model: int, seed: int, match_norm: float = 1.0) -> Direction:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(d_model)
    vec, _ = _unit(vec)
    return Direction("random", layer_id, "last", vec, match_norm, {"raw_norm": match_norm})


def save_directions(directions: list[Direction], path) -> None:
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path.with_suffix(".npz"),
        **{f"{d.name}|L{d.layer_id}|{d.position_spec}": d.vector for d in directions},
    )
    meta = [
        {
            "name": d.name,
            "layer_id": d.layer_id,
            "position_spec": d.position_spec,
            "raw_norm": d.raw_norm,
            "scores": d.scores,
        }
        for d in directions
    ]
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_directions(path) -> list[Direction]:
    import json
    from pathlib import Path

    path = Path(path)
    arrays = np.load(path.with_suffix(".npz"))
    meta = {(_key(m)): m for m in json.loads(path.with_suffix(".json").read_text())}
    directions: list[Direction] = []
    for key in arrays.files:
        name, layer, pos = key.split("|")
        layer_id = int(layer[1:])
        m = meta.get((name, layer_id, pos), {})
        directions.append(
            Direction(name, layer_id, pos, arrays[key], m.get("raw_norm", 1.0), m.get("scores", {}))
        )
    return directions


def _key(m: dict) -> tuple:
    return (m["name"], m["layer_id"], m["position_spec"])
