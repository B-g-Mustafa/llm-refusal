# RUNBOOK — The Authorization-Claim Paradox

Mechanistic study of why self-asserted authorization *increases* refusal on
legitimate dual-use requests. Package: `authpar/`. Stage scripts: `scripts/`.

## Setup (GPU server, CUDA)

```bash
pip install -r requirements.txt
```

Models run in BF16 with `device_map="auto"`; Qwen3-14B needs ~30 GB VRAM.
Everything except the stages that forward the model runs on a laptop.

## Run order

Each stage writes to `results/<model_key>/` and reads the previous stage's files.
Model keys: `qwen3-14b`, `qwen3-8b`, `llama3.1-8b`, `gemma3-12b`.

```bash
# 1. Build the frozen corpus + crossed prompt table (once; needs network for harmful items)
python scripts/01_build_dataset.py --harmful-size 120

# 2. GATE 1 — does the effect exist? Run on >=2 models before anything else.
python scripts/02_behavioral_gate.py --model qwen3-14b
python scripts/02_behavioral_gate.py --model llama3.1-8b
#    STOP if authorized-vs-none and the DiD CIs do not exclude 0 on dual-use tasks.

# 3. Extract the four candidate directions on the disjoint TRAIN split
python scripts/03_extract_directions.py --model qwen3-14b --layers 8:40:2

# 4. Held-out projections: which representation moves under authorization?
python scripts/04_projection_tests.py --model qwen3-14b

# 5. Activation patching: localize the effect (necessity / sufficiency)
python scripts/05_patching.py --model qwen3-14b --layers 8:40:4

# 6. Interventions: causal effect x operator x environment (+ safety regression)
python scripts/06_interventions.py --model qwen3-14b --alphas 4,8 --heavy-top 3

# 7. Analysis: assemble C1–C4, the trade-off, and figures
python scripts/07_analyze.py --model qwen3-14b
```

Repeat 3–7 per model. Stage 02 also accepts `--thinking` (Qwen3) and
`--table prompt_table.jsonl` for the full grid.

## What each stage produces

| Stage | Output | Answers |
|---|---|---|
| 01 | `data/{base_tasks,prompt_table,gate_table}.jsonl` | the clean factorial |
| 02 | `results/<m>/gate_refusal_scores.jsonl` + console | Gate 1: is the paradox real? |
| 03 | `results/<m>/directions.{npz,json}` | d_harm, d_refusal, d_claim, d_persuasion + certificates |
| 04 | `results/<m>/projections.jsonl` | which direction moves (H1 vs H2/H3) |
| 05 | `results/<m>/patching.jsonl` | where the effect enters (layer × span) |
| 06 | `results/<m>/interventions.jsonl` | causal effect per operator/environment |
| 07 | `results/<m>/analysis/summary.{md,json}` + PNGs | C1–C4, trade-off |

## Notes

- Results are cheap first-token refusal log-odds; validate against generations by
  producing a judged subset and checking `authpar.judge.agreement` (metric-validity gate).
- All CIs are base-task-clustered bootstraps; extraction and evaluation never
  share base tasks (`scripts/_common.split_base_ids`).
- Harmful prompts are loaded from HarmBench/AdvBench at runtime, never authored here.
