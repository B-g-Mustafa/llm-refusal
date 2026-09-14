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
    with _common.script_run("04_projection_tests", args.model) as log:
        log(f"args: {vars(args)}")
        _common.set_seed(args.seed)
        outdir = _common.results_dir(args.model)

        with _common.stage(log, "load held-out eval rows"):
            all_rows = data.read_prompt_table(_common.DATA_DIR / "prompt_table.jsonl")
            train_ids = {r["base_id"] for r in _common.read_jsonl(outdir / "train_base_ids.jsonl")}
            eval_rows = [r for r in all_rows if r.base_id not in train_ids]
            log(f"  eval rows (held-out base tasks): {len(eval_rows)}")

        with _common.stage(log, "load directions"):
            dirs = directions.load_directions(outdir / "directions")
            positions = sorted({d.position_spec for d in dirs})
            log(f"  {len(dirs)} directions across position specs: {positions}")

        with _common.stage(log, f"load model {args.model}"):
            lm = load_model(args.model)

        with _common.stage(log, "score refusal log-odds on eval rows"):
            scorer = refusal.RefusalScorer(lm)
            refusal_scores = scorer.score_rows(
                eval_rows, batch_size=args.batch_size, enable_thinking=args.thinking,
                on_progress=lambda d, t: log.progress(d, t, every=max(1, t // 10), prefix="score"),
            )

        layer_ids = sorted({d.layer_id for d in dirs})
        layer_index = {l: i for i, l in enumerate(layer_ids)}

        records = [dict(r.__dict__) for r in eval_rows]
        for i, s in enumerate(refusal_scores):
            records[i]["refusal_logodds"] = float(s)

        # Capture once per position spec, then project every direction at that spec.
        for pos_i, pos in enumerate(positions, start=1):
            with _common.stage(log, f"capture + project @ {pos} ({pos_i}/{len(positions)})"):
                acts = capture.capture_activations(
                    lm, eval_rows, layer_ids, pos,
                    batch_size=args.batch_size, enable_thinking=args.thinking,
                    on_progress=lambda d, t: log.progress(d, t, every=max(1, t // 5), prefix=f"capture[{pos}]"),
                )
                n_dirs_here = 0
                for d in dirs:
                    if d.position_spec != pos:
                        continue
                    li = layer_index[d.layer_id]
                    proj = acts[:, li, :] @ d.vector
                    col = f"proj|{d.name}|L{d.layer_id}|{d.position_spec}"
                    for i, value in enumerate(proj):
                        records[i][col] = float(value)
                    n_dirs_here += 1
                log(f"  projected {n_dirs_here} directions at position {pos}")

        with _common.stage(log, "write projections.jsonl"):
            _common.write_jsonl(records, outdir / "projections.jsonl")
            log(f"  wrote projections for {len(records)} rows -> {outdir/'projections.jsonl'}")

        log("Stage 07 will test which direction's projection shift under `authorized` "
            "explains the refusal increase (H1 vs H2/H3).")


if __name__ == "__main__":
    main()
