"""Stage 03 - extract the four candidate directions on disjoint training data.

Problem solved: to explain the authorization effect we need well-defined,
non-circular directions. This stage estimates, per layer, on a *training* split
of base tasks that later evaluation never touches:

- d_harm       : harmful vs. benign_neutral at t_inst.
- d_refusal    : refused vs. answered completions (baseline generate + judge) at
                 post-instruction positions.
- d_claim      : authorized vs. matched irrelevant_preamble on identical
                 dual_use_defensive tasks (template-controlled DiD contrast).
- d_persuasion : persuasion/role-play wrappers vs. matched benign wrappers.

d_claim is deliberately NOT labelled a manipulation direction; whether it aligns
with d_persuasion (H2/H3) or with d_harm (H1) is measured in stage 07.

Run: python scripts/03_extract_directions.py --model qwen3-14b --layers 8:40:2
"""

from __future__ import annotations

import argparse

import numpy as np

import _common
from authpar import capture, data, directions, judge, refusal, templates
from authpar.data import BaseTask, PromptRow
from authpar.modeling import load_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--layers", default="8:40:2", help="start:stop:step (half-open).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--train-frac", type=float, default=0.5)
    p.add_argument("--gen-max-new", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_layers(spec: str, n: int) -> list[int]:
    parts = spec.split(":")
    start, stop = int(parts[0]), int(parts[1])
    step = int(parts[2]) if len(parts) > 2 else 1
    return [l for l in range(start, min(stop, n), step)]


def none_user_rows(base_tasks: list[BaseTask]) -> list[PromptRow]:
    rows = data.build_prompt_table(base_tasks, families=["none"], channels=["user"])
    return rows


def _cap(lm, rows, layer_ids, pos, batch_size, log, tag):
    """capture_activations with progress mirrored into the log file."""
    return capture.capture_activations(
        lm, rows, layer_ids, pos, batch_size,
        on_progress=lambda d, t: log.progress(d, t, every=max(1, t // 5), prefix=tag),
    )


def main() -> None:
    args = parse_args()
    with _common.script_run("03_extract_directions", args.model) as log:
        log(f"args: {vars(args)}")
        _common.set_seed(args.seed)

        with _common.stage(log, "load base tasks + train/eval split"):
            base_tasks = [BaseTask(**r) for r in
                          _common.read_jsonl(_common.DATA_DIR / "base_tasks.jsonl")]
            ids = [t.base_id for t in base_tasks]
            train_ids, _eval_ids = _common.split_base_ids(ids, args.train_frac, args.seed)
            train = [t for t in base_tasks if t.base_id in train_ids]
            _common.write_jsonl([{"base_id": i} for i in sorted(train_ids)],
                                _common.results_dir(args.model) / "train_base_ids.jsonl")
            log(f"  {len(base_tasks)} base tasks -> {len(train)} train / "
                f"{len(base_tasks) - len(train)} eval")

        with _common.stage(log, f"load model {args.model}"):
            lm = load_model(args.model)
            layer_ids = parse_layers(args.layers, lm.num_layers)
            log(f"  layers: {lm.num_layers}; extracting at {layer_ids}")

        harmful = [t for t in train if t.task_type == "harmful"]
        benign = [t for t in train if t.task_type == "benign_neutral"]
        defensive = [t for t in train if t.task_type == "dual_use_defensive"]
        log(f"train pools: harmful={len(harmful)} benign={len(benign)} defensive={len(defensive)}")

        all_dirs: list[directions.Direction] = []

        # --- d_harm : harmful vs benign at t_inst ---------------------------
        if harmful and benign:
            with _common.stage(log, "extract d_harm (harmful vs benign @ t_inst)"):
                h_rows, b_rows = none_user_rows(harmful), none_user_rows(benign)
                h_act = _cap(lm, h_rows, layer_ids, "t_inst", args.batch_size, log, "d_harm[harmful]")
                b_act = _cap(lm, b_rows, layer_ids, "t_inst", args.batch_size, log, "d_harm[benign]")
                for li, layer in enumerate(layer_ids):
                    all_dirs.append(directions.diff_in_means_direction(
                        "d_harm", layer, "t_inst", h_act[:, li, :], b_act[:, li, :]))
                log(f"  d_harm directions: {len(layer_ids)}")
        else:
            log("SKIP d_harm: empty harmful or benign train pool")

        # --- d_refusal : refused vs answered at post_all --------------------
        with _common.stage(log, "extract d_refusal (refused vs answered @ post_all)"):
            refused_rows, answered_rows = _refused_answered(lm, harmful + benign, args, log)
            if refused_rows and answered_rows:
                r_act = _cap(lm, refused_rows, layer_ids, "post_all", args.batch_size, log, "d_refusal[refused]")
                a_act = _cap(lm, answered_rows, layer_ids, "post_all", args.batch_size, log, "d_refusal[answered]")
                for li, layer in enumerate(layer_ids):
                    all_dirs.append(directions.diff_in_means_direction(
                        "d_refusal", layer, "post_all", r_act[:, li, :], a_act[:, li, :]))
                log(f"  d_refusal directions: {len(layer_ids)}")
            else:
                log("  SKIP d_refusal: no refused or no answered rows in baseline generation")

        # --- d_claim : authorized vs irrelevant_preamble on identical tasks -
        if defensive:
            with _common.stage(log, "extract d_claim (authorized vs irrelevant_preamble)"):
                auth = data.build_prompt_table(defensive, families=["authorized"], channels=["user"])
                irr = data.build_prompt_table(defensive, families=["irrelevant_preamble"], channels=["user"])
                auth, irr = _align_by_base(auth, irr)
                log(f"  aligned pairs: {len(auth)}")
                for pos in ["t_inst", "post_all", "last"]:
                    a_act = _cap(lm, auth, layer_ids, pos, args.batch_size, log, f"d_claim[{pos},auth]")
                    i_act = _cap(lm, irr, layer_ids, pos, args.batch_size, log, f"d_claim[{pos},irr]")
                    for li, layer in enumerate(layer_ids):
                        all_dirs.append(directions.diff_in_means_direction(
                            "d_claim", layer, pos, a_act[:, li, :], i_act[:, li, :], with_svd=True))
                    log(f"  d_claim[{pos}] directions: {len(layer_ids)}")
        else:
            log("SKIP d_claim: empty defensive train pool")

        # --- d_persuasion : persuasion vs matched control wrappers ----------
        with _common.stage(log, "extract d_persuasion (persuasion vs matched control)"):
            persuasion_rows, control_rows = _persuasion_rows(harmful + defensive)
            if persuasion_rows:
                p_act = _cap(lm, persuasion_rows, layer_ids, "post_all", args.batch_size, log, "d_persuasion[persuasion]")
                c_act = _cap(lm, control_rows, layer_ids, "post_all", args.batch_size, log, "d_persuasion[control]")
                for li, layer in enumerate(layer_ids):
                    all_dirs.append(directions.diff_in_means_direction(
                        "d_persuasion", layer, "post_all", p_act[:, li, :], c_act[:, li, :]))
                log(f"  d_persuasion directions: {len(layer_ids)}")
            else:
                log("  SKIP d_persuasion: no persuasion rows built")

        with _common.stage(log, "save directions"):
            outpath = _common.results_dir(args.model) / "directions"
            directions.save_directions(all_dirs, outpath)
            log(f"  saved {len(all_dirs)} directions -> {outpath}.npz")


def _refused_answered(lm, tasks, args, log):
    """Baseline-generate on `none/user` rows, judge, split into refused/answered."""
    rows = none_user_rows(tasks)
    if not rows:
        return [], []
    completions = refusal.generate(
        lm, rows, max_new_tokens=args.gen_max_new, batch_size=4,
        on_progress=lambda d, t: log.progress(d, t, every=max(1, t // 5), prefix="baseline-generate"),
    )
    judged = judge.judge_completions(rows, completions)
    refused = [row for row, j in zip(rows, judged) if j.is_refusal]
    answered = [row for row, j in zip(rows, judged) if not j.is_refusal]
    log(f"  d_refusal pool: {len(refused)} refused, {len(answered)} answered "
        f"(out of {len(rows)} baseline generations)")
    return refused, answered


def _align_by_base(a_rows, b_rows):
    b_by_base = {r.base_id: r for r in b_rows}
    a_out, b_out = [], []
    for r in a_rows:
        if r.base_id in b_by_base:
            a_out.append(r)
            b_out.append(b_by_base[r.base_id])
    return a_out, b_out


def _persuasion_rows(tasks):
    """Wrap a small task pool in persuasion vs. matched benign templates."""
    pool = tasks[: min(len(tasks), 24)]
    persuasion, control = [], []
    for i, task in enumerate(pool):
        pt = templates.PERSUASION_TEMPLATES[i % len(templates.PERSUASION_TEMPLATES)]
        ct = templates.PERSUASION_CONTROL_TEMPLATES[i % len(templates.PERSUASION_CONTROL_TEMPLATES)]
        persuasion.append(PromptRow(
            prompt_id=f"persuasion.{i}", base_id=task.base_id, task_type=task.task_type,
            domain=task.domain, framing="persuasion", variant_id=f"persuasion.{i}",
            channel="user", base_text=task.text, instruction=pt.format(task=task.text),
            system_text=""))
        control.append(PromptRow(
            prompt_id=f"pcontrol.{i}", base_id=task.base_id, task_type=task.task_type,
            domain=task.domain, framing="pcontrol", variant_id=f"pcontrol.{i}",
            channel="user", base_text=task.text, instruction=ct.format(task=task.text),
            system_text=""))
    return persuasion, control


if __name__ == "__main__":
    main()
