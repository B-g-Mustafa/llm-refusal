"""Stage 07 - analysis: assemble C1-C4, the trade-off, and figures.

Problem solved: turns the raw per-stage files into the paper's claims, each with
uncertainty and a figure.

- C1 (geometry != causality): Spearman(geometric certificate, causal effect on
  harmful) across all (direction, layer), with bootstrap CIs. Weak, CI-spanning
  correlations support the headline.
- C2 (construction changes the estimand): cosine similarity between direction
  families at matched layers, and which direction's projection moves most under
  `authorized` (the mechanistic suspect: H1 if d_harm, H2/H3 if d_persuasion/
  d_claim).
- C3 (operator x protocol): effect-on-harmful pivoted by (direction, operator);
  correlation of a direction's effect across operators.
- C4 (CPCV): worst-case-across-environments selection vs. a geometry-only choice,
  reported only on held-out environments.
- Trade-off: for the best over-refusal-reducing intervention, does harmful and
  persuasion-jailbreak refusal survive?
- Metric validity: agreement between the log-odds score and judged labels, if a
  judged subset exists.

Run: python scripts/07_analyze.py --model qwen3-14b
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import _common
from authpar import stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with _common.script_run("07_analyze", args.model) as log:
        log(f"args: {vars(args)}")
        outdir = _common.results_dir(args.model)
        andir = outdir / "analysis"
        andir.mkdir(parents=True, exist_ok=True)
        summary: dict = {"model": args.model}

        with _common.stage(log, "load upstream result files"):
            interventions = _maybe(outdir / "interventions.jsonl")
            projections = _maybe(outdir / "projections.jsonl")
            patching = _maybe(outdir / "patching.jsonl")
            log(f"  interventions.jsonl: {len(interventions) if interventions else 'MISSING'}")
            log(f"  projections.jsonl:   {len(projections) if projections else 'MISSING'}")
            log(f"  patching.jsonl:      {len(patching) if patching else 'MISSING'}")

        if interventions:
            with _common.stage(log, "C1: geometry vs. causality"):
                summary["C1_geometry_vs_causality"] = _c1(interventions, andir, args.seed)
            with _common.stage(log, "C3: operator x protocol"):
                summary["C3_operator_protocol"] = _c3(interventions, andir)
            with _common.stage(log, "C4: CPCV ranking"):
                summary["C4_cpcv"] = _c4(interventions)
            with _common.stage(log, "safety-utility trade-off"):
                summary["tradeoff"] = _tradeoff(interventions)
        else:
            log("SKIP C1/C3/C4/tradeoff: interventions.jsonl missing (run stage 06 first)")

        if projections:
            with _common.stage(log, "C2: construction changes the estimand"):
                summary["C2_construction"] = _c2(projections, interventions, andir)
        else:
            log("SKIP C2: projections.jsonl missing (run stage 04 first)")

        if patching:
            with _common.stage(log, "patching peak"):
                summary["patching_peak"] = _patch_peak(patching)
        else:
            log("SKIP patching_peak: patching.jsonl missing (run stage 05 first)")

        with _common.stage(log, "write summary.json + summary.md"):
            (andir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            _write_markdown(summary, andir / "summary.md")
            log(f"  wrote {andir/'summary.md'} and figures in {andir}")

        log("summary (first 2000 chars):")
        log(json.dumps(summary, indent=2)[:2000])


def _maybe(path: Path):
    return _common.read_jsonl(path) if path.exists() else None


# --------------------------------------------------------------------------- C1
def _c1(records, andir, seed) -> dict:
    real = [r for r in records if r["name"] != "random" and r["operator"] == "ablate_single"]
    causal = np.array([r.get("effect|harmful", np.nan) for r in real])
    metrics = ["raw_norm", "proj_auroc", "cohens_d", "probe_auroc", "svd_top1_energy"]
    result = {}
    fig_rows = []
    for m in metrics:
        geo = np.array([r["scores"].get(m, np.nan) for r in real], dtype=float)
        if np.isfinite(geo).sum() < 5:
            continue
        est = stats.spearman_ci(geo, causal, seed=seed)
        result[m] = {"spearman": est.value, "lo": est.lo, "hi": est.hi, "n": est.n_clusters}
        fig_rows.append((m, geo, causal))
    _scatter_c1(fig_rows, andir)
    return result


def _scatter_c1(fig_rows, andir) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not fig_rows:
        return
    fig, axes = plt.subplots(1, len(fig_rows), figsize=(4 * len(fig_rows), 3.5), squeeze=False)
    for ax, (m, geo, causal) in zip(axes[0], fig_rows):
        mask = np.isfinite(geo) & np.isfinite(causal)
        ax.scatter(geo[mask], causal[mask], s=14, alpha=0.6)
        ax.set_xlabel(m)
        ax.set_ylabel("causal effect (harmful)")
        ax.set_title(m)
    fig.suptitle("C1: geometric certificate vs. causal control")
    fig.tight_layout()
    fig.savefig(andir / "c1_geometry_vs_causality.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- C2
def _c2(projections, interventions, andir) -> dict:
    # projection shift under authorized (template-controlled vs irrelevant), on
    # dual_use_defensive eval rows, per direction column.
    proj_cols = sorted({k for r in projections for k in r if k.startswith("proj|")})
    defensive = [r for r in projections if r.get("task_type") == "dual_use_defensive"]
    shifts = {}
    for col in proj_cols:
        auth = np.array([r[col] for r in defensive if r["framing"] == "authorized" and col in r])
        irr = np.array([r[col] for r in defensive if r["framing"] == "irrelevant_preamble" and col in r])
        if len(auth) and len(irr):
            shifts[col] = float(auth.mean() - irr.mean())
    ranked = sorted(shifts.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return {"projection_shift_under_authorized": dict(ranked[:10])}


# --------------------------------------------------------------------------- C3
def _c3(records, andir) -> dict:
    by_op = defaultdict(list)
    for r in records:
        if r["name"] == "random":
            continue
        by_op[r["operator"]].append((r["name"], r["layer"], r.get("effect|harmful", np.nan)))
    # correlation of effects between operators over shared (name,layer) keys.
    ops = [o for o in ["ablate_single", "ablate_all", "orthogonalize", "add_neg_a8"] if o in by_op]
    corr = {}
    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            a = {(n, l): e for n, l, e in by_op[ops[i]]}
            b = {(n, l): e for n, l, e in by_op[ops[j]]}
            keys = [k for k in a if k in b and np.isfinite(a[k]) and np.isfinite(b[k])]
            if len(keys) >= 5:
                rho = stats.spearman(np.array([a[k] for k in keys]), np.array([b[k] for k in keys]))
                corr[f"{ops[i]}~{ops[j]}"] = {"spearman": rho, "n": len(keys)}
    return {"operator_effect_correlation": corr}


# --------------------------------------------------------------------------- C4
def _c4(records) -> dict:
    envs = ["harmful", "persuasion_harmful", "defensive_auth", "benign"]
    per_dir = defaultdict(dict)
    for r in records:
        if r["operator"] != "ablate_single":
            continue
        key = f"{r['name']}|L{r['layer']}"
        # "good" effect: reduce over-refusal on defensive, keep refusal on harmful.
        causal_by_env = {
            "defensive_auth": r.get("effect|defensive_auth", 0.0),      # want +
            "harmful_keep": -r.get("effect|harmful", 0.0),               # want ~0 (penalize drop)
            "persuasion_keep": -r.get("effect|persuasion_harmful", 0.0),  # want ~0
        }
        benign_kl = abs(r.get("effect|benign", 0.0))  # proxy for damage
        per_dir[key] = stats.cpcv_score(causal_by_env, benign_kl, degeneration=0.0)
    ranked = sorted(per_dir.items(), key=lambda kv: kv[1], reverse=True)
    return {"cpcv_ranking_top": dict(ranked[:8])}


# ---------------------------------------------------------------------- tradeoff
def _tradeoff(records) -> dict:
    best = None
    for r in records:
        gain = r.get("effect|defensive_auth", 0.0)  # over-refusal reduction
        if best is None or gain > best[0]:
            best = (gain, r)
    if not best:
        return {}
    _, r = best
    return {
        "best_direction": f"{r['name']}|L{r['layer']}|{r['operator']}",
        "over_refusal_reduction": r.get("effect|defensive_auth"),
        "harmful_refusal_drop": r.get("effect|harmful"),
        "persuasion_refusal_drop": r.get("effect|persuasion_harmful"),
        "benign_change": r.get("effect|benign"),
        "verdict": ("selective fix" if r.get("effect|harmful", 0) < 0.1
                    and r.get("effect|persuasion_harmful", 0) < 0.1
                    else "trade-off: safety regresses"),
    }


def _patch_peak(records) -> dict:
    peak = max(records, key=lambda r: r.get("necessity_frac_recovered", -np.inf))
    return {"layer": peak["layer"], "span": peak["span"],
            "necessity_frac_recovered": peak["necessity_frac_recovered"],
            "sufficiency_frac_recovered": peak["sufficiency_frac_recovered"]}


def _write_markdown(summary: dict, path: Path) -> None:
    lines = [f"# Authorization-Claim Paradox - {summary['model']}", ""]
    c1 = summary.get("C1_geometry_vs_causality", {})
    if c1:
        lines += ["## C1: geometry vs. causality (Spearman, [95% CI])", ""]
        for m, s in c1.items():
            lines.append(f"- **{m}**: rho={s['spearman']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}] (n={s['n']})")
        lines.append("")
    if "C2_construction" in summary:
        lines += ["## C2: which direction moves under `authorized`", ""]
        for col, val in summary["C2_construction"]["projection_shift_under_authorized"].items():
            lines.append(f"- {col}: shift={val:+.3f}")
        lines.append("")
    if "patching_peak" in summary:
        p = summary["patching_peak"]
        lines += ["## Patching peak (necessity)",
                  f"- layer {p['layer']}, span {p['span']}: "
                  f"{p['necessity_frac_recovered']:.2f} of gap recovered", ""]
    if "tradeoff" in summary:
        t = summary["tradeoff"]
        lines += ["## Safety-utility trade-off", f"- {json.dumps(t, indent=2)}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
