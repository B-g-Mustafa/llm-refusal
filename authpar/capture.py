"""Multi-position residual-stream capture via forward-pre-hooks.

We capture the *input* to each selected decoder block (the residual stream
entering layer L), exactly as ``refusal_ablation.capture_layer_inputs`` does, so
that extraction and later intervention happen at the same site. The generalization
here is that each prompt may have a different length and we read a *named*
position (via :class:`authpar.modeling.SpanIndex`) rather than always the last
token.

With left padding, the padded index of an unpadded position ``p`` in a row of
length ``L`` inside a batch padded to ``M`` is ``p + (M - L)``. We compute that
offset per row and gather.

The public entry point :func:`capture_activations` returns a float32 array of
shape ``[n_rows, n_layers, d_model]`` (one vector per row per layer, meaned over
the resolved positions when the spec yields several) plus the row order.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from . import modeling
from .modeling import LoadedModel, SpanIndex


def _tokenize_batch(
    lm: LoadedModel,
    rows: list,
    enable_thinking: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[SpanIndex], list[int]]:
    """Build a left-padded batch and per-row spans/pad offsets."""
    ids_list: list[list[int]] = []
    spans: list[SpanIndex] = []
    for row in rows:
        ids, span = modeling.build_chat(
            lm.tokenizer,
            system_text=row.system_text,
            base_text=row.base_text,
            instruction=row.instruction,
            channel=row.channel,
            enable_thinking=enable_thinking,
        )
        ids_list.append(ids)
        spans.append(span)

    max_len = max(len(ids) for ids in ids_list)
    pad_id = lm.tokenizer.pad_token_id
    input_ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
    attention = torch.zeros((len(ids_list), max_len), dtype=torch.long)
    pad_offsets: list[int] = []
    for i, ids in enumerate(ids_list):
        offset = max_len - len(ids)
        pad_offsets.append(offset)
        input_ids[i, offset:] = torch.tensor(ids, dtype=torch.long)
        attention[i, offset:] = 1
    return input_ids, attention, spans, pad_offsets


@torch.inference_mode()
def capture_activations(
    lm: LoadedModel,
    rows: list,
    layer_ids: list[int],
    position_spec: str,
    batch_size: int = 8,
    enable_thinking: bool = False,
) -> np.ndarray:
    """Capture residual-stream inputs at ``position_spec`` for each row.

    Returns ``[n_rows, n_layers, d_model]`` float32 (CPU numpy). When
    ``position_spec`` resolves to several positions in a row, their mean is taken,
    which is how we realize span means such as the post-instruction region.
    """
    layers = modeling.get_layers(lm.model)
    d_model = lm.model.config.hidden_size
    out = np.zeros((len(rows), len(layer_ids), d_model), dtype=np.float32)

    for start in tqdm(range(0, len(rows), batch_size), desc=f"capture[{position_spec}]"):
        batch = rows[start : start + batch_size]
        input_ids, attention, spans, pad_offsets = _tokenize_batch(lm, batch, enable_thinking)
        input_ids = input_ids.to(lm.input_device)
        attention = attention.to(lm.input_device)

        captured: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_id: int) -> Callable:
            def hook(_module, args):
                captured[layer_id] = args[0].detach()

            return hook

        for layer_id in layer_ids:
            handles.append(layers[layer_id].register_forward_pre_hook(make_hook(layer_id)))
        try:
            lm.model(input_ids=input_ids, attention_mask=attention, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        for li, layer_id in enumerate(layer_ids):
            hidden = captured[layer_id]  # [B, M, d]
            for bi, (span, offset) in enumerate(zip(spans, pad_offsets)):
                positions = modeling.resolve_positions(span, position_spec)
                padded = [offset + p for p in positions]
                vec = hidden[bi, padded, :].to(torch.float32).mean(dim=0)
                out[start + bi, li, :] = vec.cpu().numpy()
    return out
