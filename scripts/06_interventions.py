"""Stage 06 - interventions: causal effect per direction x operator x environment.

Problem solved: produces the causal measurements the paper's claims rest on.

- C1 (geometry != causality): for each (direction, layer) it records a causal
  control effect on harmful prompts, to be correlated in stage 07 against the
  geometric certificates saved in stage 03.
- C3 (operator x protocol): each direction is intervened on with several operators
  (single-layer ablation, all-layer ablation, addition, and weight
  orthogonalization for top candidates), exposing operator-dependent effects.
- Safety-utility trade-off: every intervention is scored on four held-out
  environments - harmful refusal retention, dual-use over-refusal, benign
  retention, and persuasion-jailbreak robustness - so removing the authorization
  pathway can be judged on both helpfulness and safety.

Causal effect convention: baseline_refusal - intervened_refusal on that
environment, so a positive number means the intervention *reduced* refusal.

Run: python scripts/06_interventions.py --model qwen3-14b --alphas 4,8 --heavy-top 3
"""

from __future__ import annotations

import argparse

import numpy as np

import _common
from authpar import capture, data, directions, interventions, modeling, refusal, templates
from authpar.data import PromptRow
from authpar.modeling import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--alphas", default="4,8", help="Addition strengths (residual units).")
    p.add_argument("--heavy-top", type=int, default=3,
                   help="Top directions/name for all-layer + weight-orthogonalize ops.")
    p.add_argument("--names", default="d_harm,d_refusal,d_claim,d_persuasion")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _common.set_seed(args.seed)
    outdir = _common.results_dir(args.model)
    alphas = [float(a) for a in args.alphas.split(",")]
    names = set(args.names.split(","))

    envs = _environments(outdir)
    lm = load_model(args.model)
    scorer = refusal.RefusalScorer(lm)
    layers = modeling.get_layers(lm.model)

    dirs = [d for d in directions.load_directions(outdir / "directions") if d.name in names]
    # Add norm-matched random controls at a spread of layers.
    layer_set = sorted({d.layer_id for d in dirs})
    for l in layer_set:
        dirs.append(directions.random_direction(l, lm.model.config.hidden_size, args.seed + l))

    # Baseline refusal per environment.
    baseline = {name: scorer.score_rows(rows, args.batch_size).mean() for name, rows in envs.items()}
    print("Baseline refusal log-odds:", {k: round(float(v), 3) for k, v in baseline.items()})

    records = []

    def score_env(hooks) -> dict[str, float]:
        out = {}
        for name, rows in envs.items():
            scores = _score_with_hooks(lm, scorer, rows, hooks, args.batch_size)
            out[name] = float(scores.mean())
        return out

    # --- cheap operators over ALL directions (feeds C1) ---------------------
    for d in dirs:
        # single-layer ablation
        hooks = [(layers[d.layer_id], interventions.make_ablation_hook(d.vector))]
        _record(records, d, "ablate_single", baseline, score_env(hooks))
        # addition (negative = suppress the feature; positive = inject it)
        for alpha in alphas:
            for sign, tag in [(-1.0, "add_neg"), (1.0, "add_pos")]:
                hooks = [(layers[d.layer_id], interventions.make_addition_hook(d.vector, sign * alpha))]
                _record(records, d, f"{tag}_a{alpha:g}", baseline, score_env(hooks))

    # --- heavy operators on top candidates per name -------------------------
    top = _top_candidates(records, names, args.heavy_top)
    for d in dirs:
        key = (d.name, d.layer_id, d.position_spec)
        if key not in top:
            continue
        # all-layer ablation (global, single direction applied at every block)
        hooks = [(block, interventions.make_ablation_hook(d.vector)) for block in layers]
        _record(records, d, "ablate_all", baseline, score_env(hooks))
        # weight orthogonalization (destructive; restored right after)
        restore = interventions.orthogonalize_weights(lm, d.vector)
        try:
            _record(records, d, "orthogonalize", baseline,
                    {name: float(scorer.score_rows(rows, args.batch_size).mean())
                     for name, rows in envs.items()})
        finally:
            restore()

    _common.write_jsonl(records, outdir / "interventions.jsonl")
    print(f"\nWrote {len(records)} intervention records -> {outdir/'interventions.jsonl'}")


def _record(records, d, operator, baseline, intervened):
    rec = {"name": d.name, "layer": d.layer_id, "position": d.position_spec,
           "operator": operator, "scores": d.scores}
    for env, base in baseline.items():
        rec[f"effect|{env}"] = float(base - intervened[env])  # +ve = reduced refusal
        rec[f"intervened|{env}"] = float(intervened[env])
    records.append(rec)


def _score_with_hooks(lm, scorer, rows, hooks, batch_size):
    handles = [module.register_forward_pre_hook(hook) for module, hook in hooks]
    try:
        return scorer.score_rows(rows, batch_size=batch_size)
    finally:
        for h in handles:
            h.remove()


def _top_candidates(records, names, k):
    """Pick top-k (name,layer,pos) per name by |effect on harmful| from cheap ops."""
    best: dict[str, list] = {n: [] for n in names}
    for rec in records:
        if rec["name"] in names and rec["operator"] == "ablate_single":
            best[rec["name"]].append(
                ((rec["name"], rec["layer"], rec["position"]), abs(rec.get("effect|harmful", 0.0))))
    chosen = set()
    for n, items in best.items():
        for key, _ in sorted(items, key=lambda x: x[1], reverse=True)[:k]:
            chosen.add(key)
    return chosen


def _environments(outdir) -> dict[str, list[PromptRow]]:
    """Held-out evaluation environments (base tasks not used for extraction)."""
    all_rows = data.read_prompt_table(_common.DATA_DIR / "prompt_table.jsonl")
    train_ids = {r["base_id"] for r in _common.read_jsonl(outdir / "train_base_ids.jsonl")}
    ev = [r for r in all_rows if r.base_id not in train_ids]

    def pick(task_type, framing):
        return [r for r in ev if r.task_type == task_type and r.framing == framing
                and r.channel == "user"]

    harmful = pick("harmful", "none")
    persuasion = _wrap_persuasion(harmful)
    return {
        "harmful": harmful,                                   # want refusal retained
        "defensive_auth": pick("dual_use_defensive", "authorized"),  # want over-refusal cut
        "benign": pick("benign_neutral", "none"),            # want unchanged
        "persuasion_harmful": persuasion,                    # want refusal retained
    }


def _wrap_persuasion(harmful_rows):
    wrapped = []
    for i, r in enumerate(harmful_rows):
        t = templates.PERSUASION_TEMPLATES[i % len(templates.PERSUASION_TEMPLATES)]
        wrapped.append(PromptRow(
            prompt_id=f"jb.{r.base_id}", base_id=r.base_id, task_type="harmful",
            domain=r.domain, framing="persuasion", variant_id=f"jb.{i}",
            channel="user", base_text=r.base_text, instruction=t.format(task=r.base_text),
            system_text=""))
    return wrapped


if __name__ == "__main__":
    main()
