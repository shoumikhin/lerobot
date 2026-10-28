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

"""Run a transformers Qwen3-VL model on a fixed image grid.

The Qwen3-VL vision tower reads the image grid (`image_grid_thw`) back to Python with `.tolist()` to build
its position embeddings and attention chunks, so `torch.export` cannot trace it. When every batch uses the
same camera grid, what the tower derives from the grid can be computed once in eager mode
(`Qwen3VisionGrid.build`) and reused by the forward below, which runs the stock submodules and gives the
same numbers as the stock forward. Attention runs once per image, like the stock non-flash path, so a
flash-attention backend is called per image instead of on one packed sequence.

`Qwen3VisionGrid.build` calls the tower's `rot_pos_emb`, which transformers removed in 5.17, while LeRobot
pins it below 5.6. From 5.9 on, the tower accepts these precomputed tensors itself
(`transformers.vision_utils`), which can replace this module once the pin moves.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLVisionAttention,
        Qwen3VLVisionModel,
        apply_rotary_pos_emb_vision,
        eager_attention_forward,
    )
    from transformers.utils import torch_compilable_check
else:
    ALL_ATTENTION_FUNCTIONS = None
    Qwen3VLModel = None
    Qwen3VLVisionAttention = None
    Qwen3VLVisionModel = None
    apply_rotary_pos_emb_vision = None
    eager_attention_forward = None
    torch_compilable_check = None


def _bilinear_indices_and_weights(
    grid_thw: list[list[int]], num_grid_per_side: int, merge_size: int
) -> tuple[Tensor, Tensor]:
    """The weight-independent part of `Qwen3VLVisionModel.fast_pos_embed_interpolate`.

    Returns the four `pos_embed` rows each patch interpolates and their weights, both `(4, patches)`, in the
    tower's patch order (frames repeated, then grouped by merge window).
    """
    side = num_grid_per_side
    indices, weights = [], []
    for t, h, w in grid_thw:
        h_idxs = torch.linspace(0, side - 1, h)
        w_idxs = torch.linspace(0, side - 1, w)
        h_floor, w_floor = h_idxs.int(), w_idxs.int()
        h_ceil, w_ceil = (h_floor + 1).clip(max=side - 1), (w_floor + 1).clip(max=side - 1)
        dh, dw = h_idxs - h_floor, w_idxs - w_floor
        index = torch.stack(
            [(a[:, None] * side + b[None]).flatten() for a in (h_floor, h_ceil) for b in (w_floor, w_ceil)]
        )
        weight = torch.stack([(a[:, None] * b[None]).flatten() for a in (1 - dh, dh) for b in (1 - dw, dw)])
        order = (
            torch.arange(h * w)
            .repeat(t)
            .view(t, h // merge_size, merge_size, w // merge_size, merge_size)
            .permute(0, 1, 3, 2, 4)
            .flatten()
        )
        indices.append(index[:, order].long())
        weights.append(weight[:, order])
    return torch.cat(indices, dim=1), torch.cat(weights, dim=1)


@dataclass
class Qwen3VisionGrid:
    """Everything a Qwen3-VL vision tower derives from one image grid, computed in eager mode.

    The learned position embeddings are not cached, only which rows they interpolate, so training and
    weight updates always use the current `pos_embed`.
    """

    grid_thw: list[list[int]]
    device: torch.device
    dtype: torch.dtype
    pos_embed_indices: Tensor
    pos_embed_weights: Tensor
    position_embeddings: tuple[Tensor, Tensor]
    chunk_lengths: list[int]

    @classmethod
    def build(cls, visual: "Qwen3VLVisionModel", grid_thw: Tensor) -> "Qwen3VisionGrid":
        """Compute the grid metadata exactly as `Qwen3VLVisionModel.forward` does."""
        grid = grid_thw.tolist()
        indices, weights = _bilinear_indices_and_weights(
            grid, visual.num_grid_per_side, visual.spatial_merge_size
        )
        rotary_pos_emb = visual.rot_pos_emb(grid_thw).reshape(indices.shape[1], -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        device = visual.pos_embed.weight.device
        return cls(
            grid_thw=grid,
            device=visual.device,
            dtype=visual.dtype,
            pos_embed_indices=indices.to(device),
            pos_embed_weights=weights.to(device),
            position_embeddings=(emb.cos(), emb.sin()),
            chunk_lengths=[h * w for t, h, w in grid for _ in range(t)],
        )

    @classmethod
    def reuse_or_build(
        cls, cache: "Qwen3VisionGrid | None", visual: "Qwen3VLVisionModel", grid_thw: Tensor
    ) -> "Qwen3VisionGrid":
        """Return `cache` if it was built for `grid_thw` and the tower's current device and dtype.

        Otherwise build a new grid, which reads `grid_thw` on the host, so run one eager call before
        `torch.export`. Reusing reads nothing: the grid values are compared with `torch_compilable_check`,
        which raises in eager and is a runtime assert in an exported program.
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


def _vision_attention(attn: "Qwen3VLVisionAttention", hidden_states: Tensor, grid: Qwen3VisionGrid) -> Tensor:
    # `Qwen3VLVisionAttention.forward` without the flash-attention branch, split by Python lengths.
    seq_length = hidden_states.shape[0]
    query, key, value = (
        attn.qkv(hidden_states).reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    query, key = apply_rotary_pos_emb_vision(query, key, *grid.position_embeddings)
    query, key, value = (x.transpose(0, 1).unsqueeze(0) for x in (query, key, value))
    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        attn.config._attn_implementation, eager_attention_forward
    )
    outputs = [
        attention_interface(
            attn, q, k, v, attention_mask=None, scaling=attn.scaling, dropout=0.0, is_causal=False
        )[0]
        for q, k, v in zip(*(x.split(grid.chunk_lengths, dim=2) for x in (query, key, value)), strict=True)
    ]
    return attn.proj(torch.cat(outputs, dim=1).reshape(seq_length, -1).contiguous())


def qwen3_vision_forward(
    visual: "Qwen3VLVisionModel", pixel_values: Tensor, grid: Qwen3VisionGrid
) -> tuple[Tensor, list[Tensor]]:
    """`Qwen3VLVisionModel.forward` on a precomputed grid: merged image embeddings and DeepStack features."""
    corners = (
        visual.pos_embed(grid.pos_embed_indices)
        * grid.pos_embed_weights.to(visual.pos_embed.weight.dtype)[..., None]
    )
    pos_embeds = corners[0] + corners[1] + corners[2] + corners[3]
    hidden_states = visual.patch_embed(pixel_values.type(visual.dtype)) + pos_embeds
    deepstack_features = []
    for layer_num, block in enumerate(visual.blocks):
        hidden_states = hidden_states + _vision_attention(block.attn, block.norm1(hidden_states), grid)
        hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))
        if layer_num in visual.deepstack_visual_indexes:
            merger = visual.deepstack_merger_list[visual.deepstack_visual_indexes.index(layer_num)]
            deepstack_features.append(merger(hidden_states))
    return visual.merger(hidden_states), deepstack_features


def qwen3_vl_forward(
    model: "Qwen3VLModel",
    inputs_embeds: Tensor,
    attention_mask: Tensor | None,
    pixel_values: Tensor,
    grid: Qwen3VisionGrid,
    image_mask: Tensor,
    position_ids: Tensor | None = None,
) -> Tensor:
    """`Qwen3VLModel.forward` for images and no cache, on a precomputed grid.

    `image_mask` is True at the image placeholder tokens. `position_ids` are the 3D positions from
    `qwen3_image_rope_index`; when None, positions are 0, 1, 2, ... as in the stock forward without
    `input_ids`. Returns the last hidden state.
    """
    image_embeds, deepstack_features = qwen3_vision_forward(model.visual, pixel_values, grid)
    torch_compilable_check(
        image_mask.sum() == image_embeds.shape[0], "Image features and image tokens do not match."
    )
    inputs_embeds = inputs_embeds.masked_scatter(
        image_mask[..., None].expand_as(inputs_embeds), image_embeds.to(inputs_embeds.dtype)
    )
    return model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_pos_masks=image_mask,
        deepstack_visual_embeds=deepstack_features,
    ).last_hidden_state


def qwen3_image_rope_index(
    input_ids: Tensor,
    attention_mask: Tensor,
    image_grid_thw: list[list[int]],
    spatial_merge_size: int,
    image_token_id: int,
) -> Tensor:
    """The position ids of `Qwen3VLModel.get_rope_index` for images on a fixed grid, with tensor ops only.

    `image_grid_thw` lists the (t, h, w) grid of every image block in `input_ids`, in order, as Python
    ints, including blocks hidden by `attention_mask`. Returns the 3D position ids `(3, batch, seq)`,
    with 0 at padding.
    """
    device = input_ids.device
    valid = attention_mask.bool()
    is_image = input_ids == image_token_id

    # Per image token, in batch order: its (t, h, w) offset inside the image, and how far the image
    # moves the text position, put on the image's last token so earlier tokens keep the image's start.
    offsets, advances = [], []
    for t, h, w in image_grid_thw:
        h, w = h // spatial_merge_size, w // spatial_merge_size
        rows = torch.arange(h, device=device).repeat_interleave(w * t)
        cols = torch.arange(w, device=device).repeat(h * t)
        offsets.append(torch.stack([torch.zeros_like(rows), rows, cols]))
        advance = torch.zeros(t * h * w, dtype=torch.long, device=device)
        advance[-1] = max(h, w)
        advances.append(advance)
    offsets = torch.cat(offsets, dim=1)
    advances = torch.cat(advances)

    steps = torch.ones_like(input_ids).masked_scatter(is_image, advances) * valid
    start = steps.cumsum(dim=1) - steps
    position_ids = start.unsqueeze(0).expand(3, -1, -1)
    image_offsets = torch.zeros_like(position_ids).masked_scatter(is_image.expand_as(position_ids), offsets)
    return (position_ids + image_offsets).masked_fill(~valid, 0)
