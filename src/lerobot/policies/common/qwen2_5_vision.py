#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a transformers Qwen2.5-VL vision tower on one fixed image grid.

The stock tower turns ``grid_thw``, and tensors built from it, into Python ints on every call
(``.tolist()``), which ``torch.export`` cannot trace. When every batch uses the same camera grid,
everything derived from it can be computed once, in eager, with the tower's own methods.
``qwen2_5_vision_forward`` then runs the tower's own submodules with that metadata and gives the
same numbers as the stock forward, without reading any tensor value on the host. Attention runs once
per image or window, like the stock non-flash path, so a flash-attention backend is called per chunk
instead of on one packed sequence.

``Qwen2_5VisionGrid.build`` calls the tower's ``rot_pos_emb`` and ``get_window_index``. Transformers
removed both in 5.17, while LeRobot pins it below 5.6. From 5.9 on, the tower accepts these
precomputed tensors itself (``transformers.vision_utils``), which can replace this module once the
pin moves.
"""

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_rotary_pos_emb_vision,
        eager_attention_forward,
    )
    from transformers.utils import torch_compilable_check


@dataclass
class Qwen2_5VisionGrid:  # noqa: N801
    """What a Qwen2.5-VL vision tower derives from one ``grid_thw``, computed once in eager."""

    grid_thw: list[list[int]]
    device: torch.device
    dtype: torch.dtype
    position_embeddings: tuple[Tensor, Tensor]
    window_index: Tensor
    reverse_index: Tensor
    full_lengths: list[int]
    window_lengths: list[int]

    @classmethod
    def build(cls, visual: nn.Module, grid_thw: Tensor) -> "Qwen2_5VisionGrid":
        """Mirror the preamble of ``Qwen2_5_VisionTransformerPretrainedModel.forward``."""
        grid = grid_thw.tolist()
        unit = visual.spatial_merge_unit
        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = visual.get_window_index(grid_thw)
        seq_len = rotary_pos_emb.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // unit, unit, -1)[window_index].reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        window_index = window_index.to(emb.device)
        return cls(
            grid_thw=grid,
            device=visual.device,
            dtype=visual.dtype,
            position_embeddings=(emb.cos(), emb.sin()),
            window_index=window_index,
            reverse_index=torch.argsort(window_index),
            full_lengths=[h * w for t, h, w in grid for _ in range(t)],
            # The stock forward drops the repeated boundaries with unique_consecutive.
            window_lengths=[b - a for a, b in pairwise(cu_window_seqlens) if b > a],
        )

    @classmethod
    def reuse_or_build(
        cls, cache: "Qwen2_5VisionGrid | None", visual: nn.Module, grid_thw: Tensor
    ) -> "Qwen2_5VisionGrid":
        """Return ``cache`` if it was built for ``grid_thw`` and the tower's current device and dtype.

        Otherwise build a new grid, which reads ``grid_thw`` on the host, so run one eager call before
        ``torch.export``. Reusing reads nothing: the grid values are compared with
        ``torch_compilable_check``, which raises in eager and is a runtime assert in an exported program.
        """
        if (
            cache is None
            or len(cache.grid_thw) != grid_thw.shape[0]
            or (cache.device, cache.dtype) != (visual.device, visual.dtype)
        ):
            return cls.build(visual, grid_thw)
        torch_compilable_check(
            grid_thw == grid_thw.new_tensor(cache.grid_thw),
            "The image grid differs from the one cached for precompute_grid_thw.",
        )
        return cache


def _chunked_vision_attention(
    attn: nn.Module, hidden_states: Tensor, position_embeddings: tuple[Tensor, Tensor], lengths: list[int]
) -> Tensor:
    """The non-flash path of ``Qwen2_5_VLVisionAttention.forward``, split by Python ``lengths``."""
    seq_length = hidden_states.shape[0]
    query, key, value = (
        attn.qkv(hidden_states).reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    query, key = apply_rotary_pos_emb_vision(query, key, *position_embeddings)
    query, key, value = (state.transpose(0, 1).unsqueeze(0) for state in (query, key, value))
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        attn.config._attn_implementation, eager_attention_forward
    )
    outputs = [
        attention_interface(
            attn, q, k, v, attention_mask=None, scaling=attn.scaling, dropout=0.0, is_causal=False
        )[0]
        for q, k, v in zip(
            *(torch.split(state, lengths, dim=2) for state in (query, key, value)), strict=True
        )
    ]
    return attn.proj(torch.cat(outputs, dim=1).reshape(seq_length, -1).contiguous())


def qwen2_5_vision_forward(visual: nn.Module, pixel_values: Tensor, grid: Qwen2_5VisionGrid) -> Tensor:
    """Return what ``visual(pixel_values, grid_thw).pooler_output`` returns, using precomputed ``grid``."""
    hidden_states = visual.patch_embed(pixel_values)
    seq_len, unit = hidden_states.shape[0], visual.spatial_merge_unit
    hidden_states = hidden_states.reshape(seq_len // unit, unit, -1)[grid.window_index].reshape(seq_len, -1)
    for layer_num, block in enumerate(visual.blocks):
        lengths = grid.full_lengths if layer_num in visual.fullatt_block_indexes else grid.window_lengths
        hidden_states = hidden_states + _chunked_vision_attention(
            block.attn, block.norm1(hidden_states), grid.position_embeddings, lengths
        )
        hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))
    return visual.merger(hidden_states)[grid.reverse_index]


def qwen2_5_image_rope_index(
    input_ids: Tensor,
    attention_mask: Tensor,
    image_grid_thw: list[list[int]],
    spatial_merge_size: int,
    image_token_id: int,
) -> Tensor:
    """``get_rope_index`` of Qwen2.5-VL for an image-only prompt, with tensor ops only.

    Text tokens count up by one. An image's tokens get ``(start, start + row, start + col)`` and move the
    next start by ``max(rows, cols)``. Padding gets 0. ``image_grid_thw`` lists every image in prompt order
    across the batch, as Python ints, and must match the image tokens. This is what transformers 4.x and
    5.5.4+ compute; 5.4.0 to 5.5.3 also scale the temporal position of still images (a bug fixed upstream
    in 5.5.4, transformers issue 45325), so their stock positions differ.
    """
    valid = attention_mask == 1
    image_tokens = input_ids == image_token_id
    local, advance = [], []
    for t, h, w in image_grid_thw:
        h, w = h // spatial_merge_size, w // spatial_merge_size
        rows = torch.arange(h).repeat_interleave(w * t)
        cols = torch.arange(w).repeat(h * t)
        local.append(torch.stack([torch.zeros_like(rows), rows, cols]))
        step = torch.zeros(t * h * w, dtype=torch.long)
        step[-1] = max(h, w)
        advance.append(step)
    local = torch.cat(local, dim=1).to(input_ids.device)
    advance = torch.cat(advance).to(input_ids.device)
    steps = torch.ones_like(input_ids).masked_scatter(image_tokens, advance) * valid
    start = steps.cumsum(-1) - steps
    offsets = torch.stack([torch.zeros_like(input_ids).masked_scatter(image_tokens, axis) for axis in local])
    return (start + offsets).masked_fill(~valid, 0)
