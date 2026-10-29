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

"""Run a transformers Qwen2.5-VL vision tower on one fixed image grid (see `qwen_vision.py`).

`Qwen2_5VisionGrid` calls the tower's `rot_pos_emb` and `get_window_index`. Transformers removed both in
5.17, while LeRobot pins it below 5.6. From 5.9 on, the tower accepts these precomputed tensors itself
(`transformers.vision_utils`), which can replace this module once the pin moves.
"""

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Self

import torch
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

from .qwen_vision import FixedImageGrid, cumulative_lengths, vision_block_forward

if TYPE_CHECKING or _transformers_available:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_rotary_pos_emb_vision,
        eager_attention_forward,
    )
else:
    apply_rotary_pos_emb_vision = None
    eager_attention_forward = None


@dataclass
class Qwen2_5VisionGrid(FixedImageGrid):  # noqa: N801
    """What a Qwen2.5-VL vision tower derives from one image grid, as its stock forward computes it."""

    position_embeddings: tuple[Tensor, Tensor]
    window_index: Tensor
    reverse_index: Tensor
    full_lengths: list[int]
    window_lengths: list[int]

    @classmethod
    def _from_grid(cls, visual: nn.Module, grid_thw: list[list[int]]) -> Self:
        grid = torch.tensor(grid_thw, device=visual.device)
        unit = visual.spatial_merge_unit
        rotary_pos_emb = visual.rot_pos_emb(grid)
        window_index, cu_window_seqlens = visual.get_window_index(grid)
        seq_len = rotary_pos_emb.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // unit, unit, -1)[window_index].reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        window_index = window_index.to(emb.device)
        return cls(
            grid_thw=grid_thw,
            device=visual.device,
            dtype=visual.dtype,
            position_embeddings=(emb.cos(), emb.sin()),
            window_index=window_index,
            reverse_index=torch.argsort(window_index),
            full_lengths=[h * w for t, h, w in grid_thw for _ in range(t)],
            # The stock forward drops the repeated boundaries with unique_consecutive.
            window_lengths=[b - a for a, b in pairwise(cu_window_seqlens) if b > a],
        )


def qwen2_5_vision_forward(visual: nn.Module, pixel_values: Tensor, grid: Qwen2_5VisionGrid) -> Tensor:
    """Return what `visual(pixel_values, grid_thw).pooler_output` returns, using precomputed `grid`."""
    hidden_states = visual.patch_embed(pixel_values)
    seq_len, unit = hidden_states.shape[0], visual.spatial_merge_unit
    hidden_states = hidden_states.reshape(seq_len // unit, unit, -1)[grid.window_index].reshape(seq_len, -1)
    full = (cumulative_lengths(grid.full_lengths, hidden_states.device), grid.full_lengths)
    window = (cumulative_lengths(grid.window_lengths, hidden_states.device), grid.window_lengths)
    for layer_num, block in enumerate(visual.blocks):
        cu_seqlens, lengths = full if layer_num in visual.fullatt_block_indexes else window
        hidden_states = vision_block_forward(
            block,
            hidden_states,
            grid.position_embeddings,
            cu_seqlens,
            lengths,
            apply_rotary_pos_emb_vision,
            eager_attention_forward,
        )
    return visual.merger(hidden_states)[grid.reverse_index]
