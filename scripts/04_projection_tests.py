"""Stage 04 - held-out projections: which representation tracks the framing effect?

Problem solved: distinguishes H1 (authorization raises the harmfulness
representation) from H2/H3 (it acts through a persuasion / trust-channel
representation) *correlationally*, and supplies the projection features that
stage 07 relates to causal effect (C1) and to the behavioral shift.

For every evaluation row (base tasks NOT used in extraction), across all framings
and channels, this records the projection of the residual stream onto each
candidate direction at that direction's own layer/position, alongside the
first-token refusal log-odds. Comparing projections between `authorized`, `none`,
and `irrelevant_preamble` tells us which direction *moves* when authorization is
claimed; that direction is the mechanistic suspect.

Run: python scripts/04_projection_tests.py --model qwen3-14b
"""

from __future__ import annotations

import argparse

import numpy as np

import _common
from authpar import capture, data, directions, refusal
from authpar.modeling import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _common.set_seed(args.seed)
    outdir = _common.results_dir(args.model)

    all_rows = data.read_prompt_table(_common.DATA_DIR / "prompt_table.jsonl")
    train_ids = {r["base_id"] for r in _common.read_jsonl(outdir / "train_base_ids.jsonl")}
    eval_rows = [r for r in all_rows if r.base_id not in train_ids]
    print(f"Eval rows (held-out base tasks): {len(eval_rows)}")

    dirs = directions.load_directions(outdir / "directions")
    positions = sorted({d.position_spec for d in dirs})

    lm = load_model(args.model)
    scorer = refusal.RefusalScorer(lm)
    refusal_scores = scorer.score_rows(eval_rows, batch_size=args.batch_size,
                                       enable_thinking=args.thinking)

    layer_ids = sorted({d.layer_id for d in dirs})
    layer_index = {l: i for i, l in enumerate(layer_ids)}

    records = [dict(r.__dict__) for r in eval_rows]
    for i, s in enumerate(refusal_scores):
        records[i]["refusal_logodds"] = float(s)

    # Capture once per position spec, then project every direction at that spec.
    for pos in positions:
        acts = capture.capture_activations(lm, eval_rows, layer_ids, pos,
                                           batch_size=args.batch_size,
                                           enable_thinking=args.thinking)
        for d in dirs:
            if d.position_spec != pos:
                continue
            li = layer_index[d.layer_id]
            proj = acts[:, li, :] @ d.vector
            col = f"proj|{d.name}|L{d.layer_id}|{d.position_spec}"
            for i, value in enumerate(proj):
                records[i][col] = float(value)

    _common.write_jsonl(records, outdir / "projections.jsonl")
    print(f"Wrote projections for {len(records)} rows -> {outdir/'projections.jsonl'}")
    print("Stage 07 will test which direction's projection shift under `authorized` "
          "explains the refusal increase (H1 vs H2/H3).")


if __name__ == "__main__":
    main()
