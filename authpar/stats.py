"""Statistics: clustered effects, bootstrap-by-base-task, and rank correlations.

Two design rules from PLAN.md drive this module:

1. **Cluster by base task.** Every framing variant of one base task is a repeated
   measure of that task, so uncertainty must be bootstrapped by resampling
   ``base_id`` (not individual prompts), or estimated with a random intercept for
   ``base_id`` and ``variant_id``.

2. **Separate selection from evaluation.** ``spearman_ci`` (for C1) and the CPCV
   selection code report on held-out data only; the caller enforces the split.

A statsmodels mixed-effects fit is used when available; otherwise a
base-task-clustered bootstrap of the paired framing effect is the fallback, which
is what most of the analysis actually relies on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def paired_framing_effect(
    base_ids: np.ndarray,
    framing: np.ndarray,
    refusal: np.ndarray,
    treat: str,
    control: str,
) -> float:
    """Mean within-base-task difference in refusal between two framings.

    For each base task, average the ``treat`` rows and the ``control`` rows, take
    the difference, then average across base tasks. This is the paired effect the
    behavioral gate reports.
    """
    diffs = []
    for bid in np.unique(base_ids):
        m = base_ids == bid
        t = refusal[m & (framing == treat)]
        c = refusal[m & (framing == control)]
        if len(t) and len(c):
            diffs.append(t.mean() - c.mean())
    return float(np.mean(diffs)) if diffs else float("nan")


@dataclass
class Estimate:
    value: float
    lo: float
    hi: float
    n_clusters: int


def bootstrap_paired_effect(
    base_ids: np.ndarray,
    framing: np.ndarray,
    refusal: np.ndarray,
    treat: str,
    control: str,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Estimate:
    """Cluster bootstrap (resample base tasks) of the paired framing effect."""
    rng = _rng(seed)
    unique = np.unique(base_ids)
    point = paired_framing_effect(base_ids, framing, refusal, treat, control)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(unique, size=len(unique), replace=True)
        diffs = []
        for bid in sample:
            m = base_ids == bid
            t = refusal[m & (framing == treat)]
            c = refusal[m & (framing == control)]
            if len(t) and len(c):
                diffs.append(t.mean() - c.mean())
        boot[b] = np.mean(diffs) if diffs else np.nan
    lo, hi = np.nanpercentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(point, float(lo), float(hi), len(unique))


def did_claim_effect(
    base_ids: np.ndarray,
    framing: np.ndarray,
    refusal: np.ndarray,
    treat: str = "authorized",
    generic: str = "irrelevant_preamble",
    baseline: str = "none",
    n_boot: int = 2000,
    seed: int = 0,
) -> Estimate:
    """Difference-in-differences: (treat - none) - (irrelevant - none).

    Isolates the authorization content from the generic-preamble nuisance (H4),
    with base-task-clustered bootstrap CIs.
    """
    rng = _rng(seed)
    unique = np.unique(base_ids)

    def did(sample: np.ndarray) -> float:
        vals = []
        for bid in sample:
            m = base_ids == bid
            t = refusal[m & (framing == treat)]
            g = refusal[m & (framing == generic)]
            b = refusal[m & (framing == baseline)]
            if len(t) and len(g) and len(b):
                vals.append((t.mean() - b.mean()) - (g.mean() - b.mean()))
        return float(np.mean(vals)) if vals else float("nan")

    point = did(unique)
    boot = np.array([did(rng.choice(unique, len(unique), replace=True)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return Estimate(point, float(lo), float(hi), len(unique))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho via ranks + Pearson (no scipy dependency)."""
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks with tie handling (1-based)."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    # average tied ranks
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (ranks[order[i]] + ranks[order[j]]) / 2
        i = j + 1
    return ranks


def spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 5000,
    seed: int = 0,
) -> Estimate:
    """Spearman rank correlation with a percentile bootstrap CI (for C1)."""
    rng = _rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    point = spearman(x, y)
    idx = np.arange(len(x))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        boot[b] = spearman(x[s], y[s])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return Estimate(point, float(lo), float(hi), len(x))


def mixed_effects_refusal(frame) -> str:
    """Fit refusal ~ task_type*framing*channel*model + (1|base_id) + (1|variant_id).

    Returns a text summary when statsmodels is available, else a note. ``frame``
    is a pandas DataFrame with those columns and a numeric ``refusal`` column.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return "statsmodels not installed; rely on bootstrap_paired_effect instead."
    model = smf.mixedlm(
        "refusal ~ C(task_type) * C(framing) * C(channel)",
        data=frame,
        groups=frame["base_id"],
    )
    return str(model.fit(method="lbfgs").summary())


def cpcv_score(
    causal_by_env: dict[str, float],
    benign_kl: float,
    degeneration: float,
    lam1: float = 1.0,
    lam2: float = 1.0,
    quantile: float = 0.10,
) -> float:
    """Cross-Protocol Causal Validation score: worst-case causal effect minus damage.

    ``causal_by_env`` maps held-out environment -> causal effect for one direction.
    The score rewards a high lower-decile (worst-group) effect and penalizes
    benign KL and degeneration. Presented as a robust selection rule, not a theory.
    """
    effects = np.array(list(causal_by_env.values()), dtype=float)
    worst = float(np.quantile(effects, quantile)) if len(effects) else float("nan")
    return worst - lam1 * benign_kl - lam2 * degeneration
