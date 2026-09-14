"""Model/tokenizer loading, chat construction, and token-span alignment.

This module is the single place that knows about chat templates, the
system/user channel split, Qwen3 thinking mode, and *where* the meaningful token
spans fall inside the rendered prompt. Everything downstream (capture, refusal
scoring, patching) addresses positions through :class:`SpanIndex` so that a
"post-instruction position" means the same thing across framings and models.

Loading is deferred and lazy so that the whole package imports on a laptop with
no GPU; only functions that actually forward the model need CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Registry of the models in scope (PLAN.md "Initial model scope").
MODEL_REGISTRY: dict[str, str] = {
    "qwen3-14b": "Qwen/Qwen3-14B",
    "qwen3-8b": "Qwen/Qwen3-8B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma3-12b": "google/gemma-3-12b-it",
}

# Models that expose an `enable_thinking` switch in their chat template.
THINKING_CAPABLE = {"qwen3-14b", "qwen3-8b"}


@dataclass
class LoadedModel:
    key: str
    name: str
    model: object
    tokenizer: object
    input_device: torch.device
    num_layers: int
    supports_thinking: bool


def load_model(model_key: str, dtype: torch.dtype = torch.bfloat16) -> LoadedModel:
    """Load a model+tokenizer by registry key with left padding for batched extraction."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = MODEL_REGISTRY.get(model_key, model_key)
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, device_map="auto", low_cpu_mem_usage=True
    )
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    layers = get_layers(model)
    return LoadedModel(
        key=model_key,
        name=name,
        model=model,
        tokenizer=tokenizer,
        input_device=input_device,
        num_layers=len(list(layers)),
        supports_thinking=model_key in THINKING_CAPABLE,
    )


def get_layers(model) -> list:
    """Return the decoder blocks. Matches the accessor used in refusal_ablation.py."""
    try:
        return list(model.model.layers)
    except AttributeError as exc:  # pragma: no cover - guards incompatible models
        raise TypeError("Expected a decoder-only model exposing model.model.layers") from exc


def _messages(system_text: str, user_text: str) -> list[dict]:
    messages: list[dict] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    return messages


def render_chat(
    tokenizer,
    system_text: str,
    user_text: str,
    enable_thinking: bool = False,
) -> str:
    """Render one prompt string with the chat template.

    ``enable_thinking`` is passed through only when the template accepts it; for
    non-thinking models the extra kwarg raises TypeError and we fall back.
    """
    messages = _messages(system_text, user_text)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


@dataclass
class SpanIndex:
    """Named token positions inside a single (unpadded) rendered prompt.

    All indices are into the row's own token sequence (no padding). ``capture``
    converts these to padded positions per batch. Spans we care about:

    - ``framing``          : the framing-prefix tokens (empty for `none`).
    - ``task``             : the base-task tokens.
    - ``final_instruction``: the last task token (t_inst; harmfulness read site).
    - ``post_instruction`` : template tokens after the task up to generation
                             (refusal read sites, Arditi-style).
    - ``length``           : total token count (last real token is length-1).
    """

    framing: tuple[int, int]
    task: tuple[int, int]
    final_instruction: int
    post_instruction: tuple[int, int]
    length: int


def build_chat(
    tokenizer,
    *,
    system_text: str,
    base_text: str,
    instruction: str,
    channel: str,
    enable_thinking: bool = False,
) -> tuple[list[int], SpanIndex]:
    """Return token ids and a :class:`SpanIndex` for one prompt row.

    We locate the base-task span by tokenizing the user turn and finding the base
    text's token run inside it. Because framings keep the base task as a trailing
    suffix (see :mod:`authpar.templates`), the task span and everything after it
    align across framings, which is the precondition for activation patching.
    """
    user_text = base_text if channel == "system" else instruction
    rendered = render_chat(tokenizer, system_text, user_text, enable_thinking)
    ids = tokenizer(rendered, add_special_tokens=False).input_ids

    task_ids = tokenizer(base_text, add_special_tokens=False).input_ids
    task_start, task_end = _find_subsequence(ids, task_ids)
    if task_start < 0:
        # Tokenizer merged a boundary; fall back to a suffix-anchored estimate so
        # the pipeline still runs. Post-instruction spans remain valid.
        task_end = _estimate_task_end(ids, tokenizer)
        task_start = max(0, task_end - len(task_ids))

    length = len(ids)
    framing_span = (0, task_start)
    return ids, SpanIndex(
        framing=framing_span,
        task=(task_start, task_end),
        final_instruction=max(task_start, task_end - 1),
        post_instruction=(task_end, length),
        length=length,
    )


def _find_subsequence(haystack: list[int], needle: list[int]) -> tuple[int, int]:
    """Return [start, end) of the last occurrence of ``needle`` in ``haystack``."""
    if not needle:
        return -1, -1
    n = len(needle)
    for start in range(len(haystack) - n, -1, -1):
        if haystack[start : start + n] == needle:
            return start, start + n
    return -1, -1


def _estimate_task_end(ids: list[int], tokenizer) -> int:
    """Best-effort: the task ends just before the assistant generation header."""
    # Most chat templates end the user turn with a small, fixed header run; the
    # generation prompt is the tail. We treat the last ~6 tokens as template.
    return max(1, len(ids) - 6)


def resolve_positions(span: SpanIndex, spec: str) -> list[int]:
    """Map a position/span spec string to concrete indices within one prompt.

    Position specs (single vector when meaned by the caller):
    - ``last``     : the final token (length-1).
    - ``t_inst``   : the last task token (harmfulness read site).
    - ``post-k``   : the k-th post-instruction position (k>=1).

    Span specs (a contiguous run; the caller means over it for capture, or
    overwrites all of it for patching):
    - ``post_all`` : all post-instruction positions.
    - ``framing_all`` : the framing-prefix span (empty for `none`).
    - ``task_all`` : the base-task span.
    """
    if spec == "last":
        return [span.length - 1]
    if spec == "t_inst":
        return [span.final_instruction]
    if spec == "post_all":
        return list(range(*span.post_instruction)) or [span.length - 1]
    if spec == "framing_all":
        return list(range(*span.framing)) or [0]
    if spec == "task_all":
        return list(range(*span.task)) or [span.final_instruction]
    if spec.startswith("post-"):
        k = int(spec.split("-", 1)[1])
        start, end = span.post_instruction
        idx = start + (k - 1)
        return [idx] if start <= idx < end else [max(start, end - 1)]
    raise ValueError(f"Unknown position/span spec: {spec!r}")
