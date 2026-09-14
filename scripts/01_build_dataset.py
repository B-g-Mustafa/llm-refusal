"""Stage 01 - build the base-task corpus and crossed prompt table.

Problem solved: the whole study rests on a *clean factorial* where only the
framing/channel changes while the base task is fixed, and where harmful items
come from an existing benchmark rather than being authored here. This stage
produces that corpus once, so every later stage reads the same frozen prompts.

Outputs (under data/):
- base_tasks.jsonl   : all base tasks (benign_neutral, dual_use_defensive, harmful).
- prompt_table.jsonl : the full grid (base x framing x channel) for the main run.
- gate_table.jsonl   : a small diagnostic subset for the Gate-1 behavioral check.

Run: python scripts/01_build_dataset.py --harmful-size 120
Use  --no-harmful to build the benign-only table offline (no network).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import _common
from authpar import data, templates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--harmful-size", type=int, default=120)
    p.add_argument("--no-harmful", action="store_true", help="Skip network download of harmful items.")
    p.add_argument("--gate-per-type", type=int, default=20, help="Base tasks/type in the gate subset.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _common.set_seed(args.seed)

    base_tasks = data.authored_base_tasks()
    if not args.no_harmful:
        base_tasks += data.load_harmful_base_tasks(args.harmful_size)
    else:
        print("Skipping harmful items (--no-harmful); benign-only table.")

    data.write_jsonl(base_tasks, _common.DATA_DIR / "base_tasks.jsonl")
    print(f"Base tasks: {len(base_tasks)}")

    # Full grid for the main experiments.
    full = data.build_prompt_table(base_tasks)
    data.write_jsonl(full, _common.DATA_DIR / "prompt_table.jsonl")
    print(f"Full prompt table rows: {len(full)}")

    # Gate subset: a handful of base tasks per type, six diagnostic framings,
    # user channel only, so Gate 1 is quick and interpretable.
    gate_families = ["none", "authorized", "not_authorized", "role_claim",
                     "institutional", "irrelevant_preamble"]
    by_type: dict[str, list] = {}
    for task in base_tasks:
        by_type.setdefault(task.task_type, []).append(task)
    gate_tasks = []
    for tasks in by_type.values():
        gate_tasks.extend(tasks[: args.gate_per_type])
    gate = data.build_prompt_table(gate_tasks, families=gate_families, channels=["user"])
    data.write_jsonl(gate, _common.DATA_DIR / "gate_table.jsonl")
    print(f"Gate table rows: {len(gate)} over {len(gate_tasks)} base tasks")

    # Report the cell layout so the design is auditable at a glance.
    print("\nFraming families:", ", ".join(templates.FRAMING_FAMILIES))
    print("Channels:", ", ".join(templates.CHANNELS))


if __name__ == "__main__":
    main()
