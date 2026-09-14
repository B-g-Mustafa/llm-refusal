"""Stage 05 - activation patching: localize the authorization effect (necessity/sufficiency).

Problem solved: projections (stage 04) are correlational. Patching is the causal
localization PLAN.md requires before saying a representation "mediates" the
effect. On held-out dual_use_defensive tasks we take matched `authorized` vs
`irrelevant_preamble` prompts (same base task) and, per layer and per span
(framing / task / post-instruction), do mean interchange patching:

- necessity   : overwrite the AUTHORIZED run's span with the IRRELEVANT run's
                mean representation. If refusal falls toward the irrelevant
                baseline, that layer/span carries the authorization effect.
- sufficiency : overwrite the IRRELEVANT run's span with the AUTHORIZED mean.
                If refusal rises toward the authorized baseline, that
                representation is sufficient to induce the effect.

Reporting the fraction of the behavioral gap recovered by each (layer, span)
gives a causal map of where authorization enters.

Run: python scripts/05_patching.py --model qwen3-14b --layers 8:40:4
"""

from __future__ import annotations

import argparse

import numpy as np

import _common
from authpar import capture, data, interventions, refusal
from authpar.modeling import load_model

SPANS = ["framing_all", "task_all", "post_all"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--layers", default="8:40:4")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_layers(spec: str, n: int) -> list[int]:
    a, b, *rest = spec.split(":")
    step = int(rest[0]) if rest else 1
    return [l for l in range(int(a), min(int(b), n), step)]


def main() -> None:
    args = parse_args()
    _common.set_seed(args.seed)
    outdir = _common.results_dir(args.model)

    all_rows = data.read_prompt_table(_common.DATA_DIR / "prompt_table.jsonl")
    train_ids = {r["base_id"] for r in _common.read_jsonl(outdir / "train_base_ids.jsonl")}
    defensive = [r for r in all_rows
                 if r.base_id not in train_ids and r.task_type == "dual_use_defensive"]

    auth = [r for r in defensive if r.framing == "authorized"]
    irr = [r for r in defensive if r.framing == "irrelevant_preamble"]
    auth, irr = _align(auth, irr)
    if not auth:
        raise SystemExit("No aligned authorized/irrelevant eval pairs; check stages 01/03.")
    print(f"Aligned eval pairs: {len(auth)}")

    lm = load_model(args.model)
    scorer = refusal.RefusalScorer(lm)
    layer_ids = parse_layers(args.layers, lm.num_layers)

    base_auth = scorer.score_rows(auth, batch_size=args.batch_size)
    base_irr = scorer.score_rows(irr, batch_size=args.batch_size)
    gap = float(base_auth.mean() - base_irr.mean())
    print(f"Behavioral gap (authorized - irrelevant): {gap:+.3f}")

    records = []
    for span in SPANS:
        # Per-row source means at each layer for both donors.
        irr_src = capture.capture_activations(lm, irr, layer_ids, span, args.batch_size)
        auth_src = capture.capture_activations(lm, auth, layer_ids, span, args.batch_size)
        for li, layer in enumerate(layer_ids):
            nec = interventions.patched_refusal_scores(
                lm, scorer, auth, irr_src[:, li, :], layer, span, args.batch_size)
            suf = interventions.patched_refusal_scores(
                lm, scorer, irr, auth_src[:, li, :], layer, span, args.batch_size)
            nec_recovered = (base_auth.mean() - nec.mean()) / gap if gap else float("nan")
            suf_recovered = (suf.mean() - base_irr.mean()) / gap if gap else float("nan")
            rec = {
                "layer": layer, "span": span, "gap": gap,
                "necessity_patched_mean": float(nec.mean()),
                "necessity_frac_recovered": float(nec_recovered),
                "sufficiency_patched_mean": float(suf.mean()),
                "sufficiency_frac_recovered": float(suf_recovered),
            }
            records.append(rec)
            print(f"L{layer:>2} {span:>12}: necessity {nec_recovered:5.2f} | "
                  f"sufficiency {suf_recovered:5.2f} (frac of gap)")

    _common.write_jsonl(records, outdir / "patching.jsonl")
    print(f"\nWrote {outdir/'patching.jsonl'}. High frac_recovered = that (layer,span) "
          "carries the authorization effect.")


def _align(a_rows, b_rows):
    b_by_base = {r.base_id: r for r in b_rows}
    a_out, b_out = [], []
    for r in a_rows:
        if r.base_id in b_by_base:
            a_out.append(r)
            b_out.append(b_by_base[r.base_id])
    return a_out, b_out


if __name__ == "__main__":
    main()
