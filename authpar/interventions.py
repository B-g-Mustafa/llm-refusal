"""Intervention operators: ablate, add, weight-orthogonalize, and patch.

These realize the "operator" axis of the audit (C3) and the necessity/sufficiency
tests of the authorization mechanism.

- ``make_ablation_hook``      : remove the projection onto a unit direction from
                                the residual stream (Arditi-style, reversible).
- ``make_addition_hook``      : add ``alpha * direction`` (signed steering).
- ``orthogonalize_weights``   : bake the ablation into the weights that write to
                                the residual stream (returns a restore closure).
- ``patched_refusal_scores``  : mean activation patching. Overwrite a named span
                                at one layer with a per-row source vector and read
                                the first-token refusal log-odds. Used for
                                necessity (patch control -> authorization run) and
                                sufficiency (patch authorization -> control run).

All hooks are forward-pre-hooks on decoder blocks, matching the capture site, so
an intervention acts exactly where a direction was measured.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from . import capture, modeling
from .modeling import LoadedModel


def make_ablation_hook(direction: np.ndarray) -> Callable:
    """Remove the component along ``direction`` (unit) at every position."""
    cache: dict = {}

    def hook(_module, args):
        hidden = args[0]
        key = (hidden.device, hidden.dtype)
        if key not in cache:
            cache[key] = torch.tensor(direction, device=hidden.device, dtype=hidden.dtype)
        vec = cache[key]
        coeff = hidden @ vec  # [B, T]
        edited = hidden - coeff.unsqueeze(-1) * vec
        return (edited, *args[1:])

    return hook


def make_addition_hook(direction: np.ndarray, alpha: float) -> Callable:
    """Add ``alpha * direction`` at every position (signed steering)."""
    cache: dict = {}

    def hook(_module, args):
        hidden = args[0]
        key = (hidden.device, hidden.dtype)
        if key not in cache:
            cache[key] = torch.tensor(direction, device=hidden.device, dtype=hidden.dtype)
        edited = hidden + alpha * cache[key]
        return (edited, *args[1:])

    return hook


def orthogonalize_weights(lm: LoadedModel, direction: np.ndarray) -> Callable:
    """Project ``direction`` out of every residual-stream write. Returns restore().

    Edits the embedding matrix and each block's attention output and MLP down
    projection, following refusal_demo.ipynb's weight-orthogonalization. The
    returned closure restores the original tensors (so a sweep is reversible).
    """
    vec = torch.tensor(direction, dtype=torch.float32)
    saved: list[tuple[torch.nn.Parameter, torch.Tensor]] = []

    def _orthogonalize(weight: torch.Tensor) -> torch.Tensor:
        v = vec.to(weight.device, weight.dtype)
        # weight rows live in d_model; remove component along v from each row.
        proj = (weight @ v).unsqueeze(-1) * v
        return weight - proj

    def _register(param: torch.nn.Parameter) -> None:
        saved.append((param, param.data.clone()))
        param.data = _orthogonalize(param.data)

    embed = lm.model.get_input_embeddings().weight
    _register(embed)
    for block in modeling.get_layers(lm.model):
        _register(block.self_attn.o_proj.weight)
        _register(block.mlp.down_proj.weight)

    def restore() -> None:
        for param, original in saved:
            param.data = original

    return restore


@torch.inference_mode()
def patched_refusal_scores(
    lm: LoadedModel,
    scorer,
    target_rows: list,
    source_vectors: np.ndarray,
    layer_id: int,
    span_name: str,
    batch_size: int = 8,
    enable_thinking: bool = False,
) -> np.ndarray:
    """First-token refusal log-odds for ``target_rows`` with a patched span.

    ``source_vectors[i]`` (shape ``[d_model]``) overwrites the residual input to
    ``layer_id`` at the ``span_name`` positions of ``target_rows[i]``. Mean
    patching (one source vector per row) sidesteps length mismatches between the
    donor and recipient framings.
    """
    layers = modeling.get_layers(lm.model)
    ref = torch.tensor(scorer.refusal_ids)
    com = torch.tensor(scorer.compliance_ids)
    out = np.zeros(len(target_rows), dtype=np.float32)

    for start in tqdm(range(0, len(target_rows), batch_size), desc=f"patch[{span_name}]"):
        batch = target_rows[start : start + batch_size]
        input_ids, attention, spans, offsets = capture._tokenize_batch(lm, batch, enable_thinking)
        input_ids = input_ids.to(lm.input_device)
        attention = attention.to(lm.input_device)

        # Precompute, per batch row, the padded positions to overwrite.
        row_positions: list[list[int]] = []
        for span, offset in zip(spans, offsets):
            positions = modeling.resolve_positions(span, span_name)
            row_positions.append([offset + p for p in positions])

        def hook(_module, args):
            hidden = args[0]
            for bi, positions in enumerate(row_positions):
                src = torch.tensor(
                    source_vectors[start + bi], device=hidden.device, dtype=hidden.dtype
                )
                hidden[bi, positions, :] = src
            return (hidden, *args[1:])

        handle = layers[layer_id].register_forward_pre_hook(hook)
        try:
            logits = lm.model(
                input_ids=input_ids, attention_mask=attention, use_cache=False
            ).logits[:, -1, :].float().cpu()
        finally:
            handle.remove()

        log_probs = torch.log_softmax(logits, dim=-1)
        r = torch.logsumexp(log_probs[:, ref], dim=-1)
        c = torch.logsumexp(log_probs[:, com], dim=-1)
        out[start : start + len(batch)] = (r - c).numpy()
    return out
