"""Layer-local, low-rank refusal-representation experiments for Qwen3.

This is intentionally an *inference-time* experiment: it never edits or saves
model weights. The old rank-one weight edit was invalid for a layer-specific
representation and made repeated experiments hard to reproduce.

Input is JSONL with one object per syntactically matched pair:
    {"harmful": "...", "benign": "..."}

Use a curated paired set. Do not mix a harm benchmark with a generic assistant
dataset: that learns topic/style features instead of the response-policy signal.

Example:
    python refusal_ablation.py --pairs paired_refusal_prompts.jsonl \
        --layers 16:32 --rank 4 --mode ablate

Qwen3 is run in non-thinking mode for both extraction and generation, so the
position being measured is the assistant-generation position, not a reasoning
token. The script is a mechanistic research tool, not a safety evaluation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.request import urlopen

import torch
from colorama import Fore, init
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class PromptPair:
    harmful: str
    benign: str


@dataclass(frozen=True)
class ContrastData:
    train_harmful: list[str]
    train_benign: list[str]
    eval_harmful: list[str]
    paired: bool
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and intervene on layer-specific paired-difference subspaces."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument(
        "--data-source",
        choices=("paired", "reference"),
        default="paired",
        help=(
            "'paired' uses --pairs and supports low-rank SVD. 'reference' reproduces the "
            "unpaired AdvBench-versus-Alpaca difference-of-means setup and requires --rank 1."
        ),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("paired_refusal_prompts.jsonl"),
        help="JSONL paired corpus used when --data-source paired (the default).",
    )
    parser.add_argument(
        "--layers",
        default="16:32",
        help="Half-open layer range START:STOP. Defaults to the middle/late band 16:32.",
    )
    parser.add_argument("--rank", type=int, default=4, help="SVD subspace rank (usually 3-8).")
    parser.add_argument(
        "--mode",
        choices=("ablate", "steer"),
        default="ablate",
        help="Ablate the SVD subspace, or steer opposite the mean paired difference.",
    )
    parser.add_argument(
        "--steering-strength",
        type=float,
        default=2.0,
        help="Only used with --mode steer; multiplier for the unit mean-difference vector.",
    )
    parser.add_argument("--train-size", type=int, default=48)
    parser.add_argument("--eval-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    custom_eval = parser.add_mutually_exclusive_group()
    custom_eval.add_argument(
        "--prompt",
        dest="custom_prompts",
        action="append",
        help=(
            "Evaluate this prompt instead of the source's held-out prompts. "
            "Repeat --prompt to evaluate multiple prompts."
        ),
    )
    custom_eval.add_argument(
        "--prompt-file",
        type=Path,
        help="Plain UTF-8 text file with one evaluation prompt per non-empty line.",
    )
    return parser.parse_args()


def load_pairs(path: Path) -> list[PromptPair]:
    """Load and validate pairs early; ordering is preserved until the seeded split."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Paired prompt file not found: {path}. Create JSONL objects with 'harmful' and 'benign' keys."
        )

    pairs: list[PromptPair] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
            harmful, benign = row["harmful"].strip(), row["benign"].strip()
        except (json.JSONDecodeError, KeyError, AttributeError) as exc:
            raise ValueError(f"Invalid pair at {path}:{line_number}") from exc
        if not harmful or not benign:
            raise ValueError(f"Empty prompt at {path}:{line_number}")
        pairs.append(PromptPair(harmful=harmful, benign=benign))

    if len(pairs) < 12:
        raise ValueError("Use at least 12 pairs; 48-128 matched pairs is a better starting point.")
    return pairs


def load_reference_splits(seed: int) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load the same two sources and 80/20 split used by the reference notebook."""
    try:
        from datasets import load_dataset
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError(
            "--data-source reference needs the optional dependencies: "
            "pip install datasets scikit-learn"
        ) from exc

    advbench_url = (
        "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
        "data/advbench/harmful_behaviors.csv"
    )
    print("Downloading AdvBench harmful_behaviors.csv...")
    with urlopen(advbench_url, timeout=30) as response:
        rows = csv.DictReader(io.TextIOWrapper(response, encoding="utf-8"))
        harmful = [row["goal"].strip() for row in rows if row.get("goal", "").strip()]

    print("Loading tatsu-lab/alpaca...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    benign = [
        row["instruction"].strip()
        for row in alpaca
        if str(row.get("input", "")).strip() == "" and row["instruction"].strip()
    ]
    harmful_train, harmful_test = train_test_split(
        harmful, test_size=0.2, random_state=seed
    )
    benign_train, benign_test = train_test_split(benign, test_size=0.2, random_state=seed)
    return harmful_train, harmful_test, benign_train, benign_test


def make_contrast_data(args: argparse.Namespace) -> ContrastData:
    """Select the paired method or the original notebook's unpaired comparison."""
    required = args.train_size + args.eval_size
    if args.data_source == "paired":
        pairs = load_pairs(args.pairs)
        random.Random(args.seed).shuffle(pairs)
        if len(pairs) < required:
            raise ValueError(
                f"Need {required} pairs for this split, but {args.pairs} contains {len(pairs)}"
            )
        train_pairs, eval_pairs = pairs[: args.train_size], pairs[args.train_size : required]
        return ContrastData(
            train_harmful=[pair.harmful for pair in train_pairs],
            train_benign=[pair.benign for pair in train_pairs],
            eval_harmful=[pair.harmful for pair in eval_pairs],
            paired=True,
            description=f"{len(train_pairs)} syntactically matched prompt pairs",
        )

    if args.rank != 1:
        raise ValueError(
            "--data-source reference is unpaired, so it reproduces only the original "
            "rank-one difference-of-means method. Use --rank 1."
        )
    harmful_train, harmful_test, benign_train, _benign_test = load_reference_splits(args.seed)
    if len(harmful_train) < args.train_size or len(benign_train) < args.train_size:
        raise ValueError("The reference training splits are smaller than --train-size.")
    if len(harmful_test) < args.eval_size:
        raise ValueError("The reference harmful test split is smaller than --eval-size.")
    return ContrastData(
        train_harmful=harmful_train[: args.train_size],
        train_benign=benign_train[: args.train_size],
        eval_harmful=harmful_test[: args.eval_size],
        paired=False,
        description=(
            f"{args.train_size} AdvBench prompts versus {args.train_size} Alpaca prompts "
            "(unpaired reference comparison)"
        ),
    )


def load_custom_eval_prompts(args: argparse.Namespace) -> list[str] | None:
    """Return explicit runtime prompts, if requested, otherwise retain source evaluation data."""
    if args.custom_prompts:
        prompts = [prompt.strip() for prompt in args.custom_prompts if prompt.strip()]
    elif args.prompt_file:
        if not args.prompt_file.is_file():
            raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
        prompts = [
            line.strip()
            for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        return None
    if not prompts:
        raise ValueError("Provide at least one non-empty --prompt or prompt-file line.")
    return prompts


def parse_layer_range(spec: str, number_of_layers: int) -> list[int]:
    try:
        start_text, stop_text = spec.split(":", maxsplit=1)
        start, stop = int(start_text), int(stop_text)
    except ValueError as exc:
        raise ValueError("--layers must be a half-open range such as 16:32") from exc
    if not (0 <= start < stop <= number_of_layers):
        raise ValueError(f"--layers {spec!r} is outside the model's 0:{number_of_layers} layer range")
    return list(range(start, stop))


def get_layers(model: AutoModelForCausalLM) -> Iterable[torch.nn.Module]:
    """Qwen3's decoder is model.model.layers; fail clearly for an incompatible model."""
    try:
        return model.model.layers
    except AttributeError as exc:
        raise TypeError("This script expects a decoder-only model exposing model.model.layers") from exc


def make_prompts(tokenizer: AutoTokenizer, instructions: list[str]) -> list[str]:
    messages = [[{"role": "user", "content": instruction}] for instruction in instructions]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        # Kept for template compatibility; Qwen3 supports enable_thinking.
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize(tokenizer: AutoTokenizer, instructions: list[str], input_device: torch.device):
    prompts = make_prompts(tokenizer, instructions)
    return tokenizer(prompts, padding=True, return_tensors="pt").to(input_device)


@torch.inference_mode()
def capture_layer_inputs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instructions: list[str],
    layer_ids: list[int],
    input_device: torch.device,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Record residual *inputs* for each requested block at the final prompt token.

    Capturing through forward-pre-hooks avoids the hidden_states off-by-one ambiguity
    and ensures extraction and intervention occur at exactly the same site.
    """
    layers = list(get_layers(model))
    captured: dict[int, list[torch.Tensor]] = {layer_id: [] for layer_id in layer_ids}
    handles = []

    def capture_hook(layer_id: int) -> Callable:
        def hook(_module, args):
            hidden_states = args[0]
            # Left padding makes -1 the assistant-generation position for every item.
            captured[layer_id].append(
                hidden_states[:, -1, :].detach().to(device="cpu", dtype=torch.float32)
            )
        return hook

    for layer_id in layer_ids:
        handles.append(layers[layer_id].register_forward_pre_hook(capture_hook(layer_id)))

    try:
        model.eval()
        for start in tqdm(range(0, len(instructions), batch_size), desc="capturing activations"):
            inputs = tokenize(tokenizer, instructions[start : start + batch_size], input_device)
            model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    return {layer_id: torch.cat(chunks, dim=0) for layer_id, chunks in captured.items()}


def extract_layer_subspaces(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    harmful: list[str],
    benign: list[str],
    layer_ids: list[int],
    input_device: torch.device,
    batch_size: int,
    rank: int,
    paired: bool,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Return orthonormal SVD bases and signed unit mean differences per layer.

    Paired data uses principal components of the uncentred paired-difference matrix.
    The reference data is unpaired, so rank one is the original difference-of-means
    direction; an SVD subspace would depend on arbitrary row pairing.
    """
    if len(harmful) != len(benign):
        raise ValueError("Harmful and benign extraction sets must have equal size.")
    if not paired and rank != 1:
        raise ValueError("Unpaired reference data supports only rank-one difference-of-means.")
    harmful_inputs = capture_layer_inputs(
        model, tokenizer, harmful, layer_ids, input_device, batch_size
    )
    benign_inputs = capture_layer_inputs(model, tokenizer, benign, layer_ids, input_device, batch_size)

    bases: dict[int, torch.Tensor] = {}
    mean_directions: dict[int, torch.Tensor] = {}
    for layer_id in layer_ids:
        if paired:
            deltas = harmful_inputs[layer_id] - benign_inputs[layer_id]
            usable_rank = min(rank, deltas.shape[0], deltas.shape[1])
            if usable_rank < 1:
                raise ValueError(f"No usable SVD components at layer {layer_id}")
            _u, singular_values, vh = torch.linalg.svd(deltas, full_matrices=False)
            bases[layer_id] = vh[:usable_rank].contiguous()
            explained = singular_values[:usable_rank].square().sum() / singular_values.square().sum()
        mean_delta = harmful_inputs[layer_id].mean(dim=0) - benign_inputs[layer_id].mean(dim=0)
        mean_norm = mean_delta.norm()
        if mean_norm <= torch.finfo(mean_delta.dtype).eps:
            raise ValueError(f"Mean difference is zero at layer {layer_id}")
        mean_directions[layer_id] = mean_delta / mean_norm
        if paired:
            print(
                f"layer {layer_id:02d}: mean-delta norm={mean_norm.item():.3f}; "
                f"top-{usable_rank} energy={explained.item():.1%}"
            )
        else:
            bases[layer_id] = mean_directions[layer_id].unsqueeze(0)
            print(f"layer {layer_id:02d}: reference difference-of-means norm={mean_norm.item():.3f}")

    del harmful_inputs, benign_inputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return bases, mean_directions


def make_intervention_hook(
    basis: torch.Tensor,
    mean_direction: torch.Tensor,
    mode: str,
    steering_strength: float,
) -> Callable:
    """Create a hook for one layer; device copies are cached for accelerate device maps."""
    device_tensors: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def hook(_module, args):
        hidden_states = args[0]
        device = hidden_states.device
        if device not in device_tensors:
            device_tensors[device] = (
                basis.to(device=device, dtype=hidden_states.dtype),
                mean_direction.to(device=device, dtype=hidden_states.dtype),
            )
        local_basis, local_mean = device_tensors[device]
        if mode == "ablate":
            # local_basis has shape [rank, d_model] and is orthonormal from SVD.
            coefficients = hidden_states @ local_basis.T
            edited = hidden_states - coefficients @ local_basis
        else:
            # The signed mean keeps steering semantically oriented; PCA axes have arbitrary signs.
            edited = hidden_states - steering_strength * local_mean
        return (edited, *args[1:])

    return hook


@torch.inference_mode()
def generate_completions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instructions: list[str],
    input_device: torch.device,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    model.eval()
    generations: list[str] = []
    for start in tqdm(range(0, len(instructions), batch_size), desc="generating"):
        inputs = tokenize(tokenizer, instructions[start : start + batch_size], input_device)
        prompt_length = inputs.input_ids.shape[1]
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        generations.extend(tokenizer.batch_decode(output_ids[:, prompt_length:], skip_special_tokens=True))
    return generations


def main() -> None:
    args = parse_args()
    if args.rank < 1 or args.train_size < 2 or args.eval_size < 1 or args.batch_size < 1:
        raise ValueError("rank, train-size, eval-size, and batch-size must be positive (train-size >= 2)")

    init(autoreset=True)
    data = make_contrast_data(args)
    custom_eval_prompts = load_custom_eval_prompts(args)
    eval_prompts = custom_eval_prompts or data.eval_harmful
    if len(data.train_harmful) < 32:
        print(Fore.YELLOW + "Warning: fewer than 32 training pairs; subspace estimates may be noisy.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    input_device = model.get_input_embeddings().weight.device
    layer_ids = parse_layer_range(args.layers, len(list(get_layers(model))))
    print(
        f"Extracting rank-{args.rank} layer-local subspaces for {len(layer_ids)} layers "
        f"from {data.description}."
    )
    bases, mean_directions = extract_layer_subspaces(
        model,
        tokenizer,
        data.train_harmful,
        data.train_benign,
        layer_ids,
        input_device,
        args.batch_size,
        args.rank,
        data.paired,
    )

    print("\nGenerating baseline completions...")
    baseline = generate_completions(
        model, tokenizer, eval_prompts, input_device, args.batch_size, args.max_new_tokens
    )

    layers = list(get_layers(model))
    handles = [
        layers[layer_id].register_forward_pre_hook(
            make_intervention_hook(
                bases[layer_id], mean_directions[layer_id], args.mode, args.steering_strength
            )
        )
        for layer_id in layer_ids
    ]
    try:
        print(f"\nGenerating {args.mode} completions with reversible layer-local hooks...")
        intervened = generate_completions(
            model, tokenizer, eval_prompts, input_device, args.batch_size, args.max_new_tokens
        )
    finally:
        for handle in handles:
            handle.remove()

    for index, (instruction, baseline_text, intervened_text) in enumerate(
        zip(eval_prompts, baseline, intervened), start=1
    ):
        print(f"\nPROMPT {index}: {instruction!r}")
        print(Fore.GREEN + "BASELINE:")
        print(textwrap.fill(repr(baseline_text), width=100, initial_indent="  ", subsequent_indent="  "))
        print(Fore.CYAN + f"{args.mode.upper()}:")
        print(textwrap.fill(repr(intervened_text), width=100, initial_indent="  ", subsequent_indent="  "))


if __name__ == "__main__":
    main()
