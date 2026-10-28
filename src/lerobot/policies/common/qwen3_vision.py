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

"""Run a stock transformers Qwen3-VL model on a fixed image grid.

The transformers Qwen3-VL vision tower reads the image grid (`image_grid_thw`) back to Python with
`.tolist()` to build its position embeddings and attention chunks, so `torch.export` cannot trace it.
Its text model adds the DeepStack features through boolean-mask indexing, which leaves a shape that
depends on tensor values in the exported graph. When every batch uses the same camera grid, everything
derived from the grid can be computed once in eager mode (`Qwen3VisionGrid.build`), and the forward
below reuses the stock submodules with that metadata and keeps every shape static. The numbers are the
same as the stock forward.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.masking_utils import create_causal_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLVisionAttention,
        Qwen3VLVisionModel,
        apply_rotary_pos_emb_vision,
        eager_attention_forward,
    )
else:
    create_causal_mask = None
    ALL_ATTENTION_FUNCTIONS = None
    Qwen3VLModel = None
    Qwen3VLVisionAttention = None
    Qwen3VLVisionModel = None
    apply_rotary_pos_emb_vision = None
    eager_attention_forward = None


@dataclass
class Qwen3VisionGrid:
    """Everything a Qwen3-VL vision tower derives from one image grid, computed in eager mode."""

    grid_thw: list[list[int]]
    pos_embeds: Tensor
    position_embeddings: tuple[Tensor, Tensor]
    chunk_lengths: list[int]

    @classmethod
    def build(cls, visual: "Qwen3VLVisionModel", grid_thw: Tensor) -> "Qwen3VisionGrid":
        """Compute the grid metadata exactly as `Qwen3VLVisionModel.forward` does."""
        pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
        rotary_pos_emb = visual.rot_pos_emb(grid_thw).reshape(pos_embeds.shape[0], -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        grid = grid_thw.tolist()
        chunk_lengths = [h * w for t, h, w in grid for _ in range(t)]
        return cls(grid, pos_embeds, (emb.cos(), emb.sin()), chunk_lengths)

    @property
    def num_patches(self) -> int:
        return self.pos_embeds.shape[0]


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
    hidden_states = visual.patch_embed(pixel_values.type(visual.dtype)) + grid.pos_embeds
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
    image_mask = image_mask[..., None].expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds.dtype))

    text = model.language_model
    batch_size, seq_length = inputs_embeds.shape[:2]
    if position_ids is None:
        position_ids = torch.arange(seq_length, device=inputs_embeds.device).view(1, 1, -1)
        position_ids = position_ids.expand(4, batch_size, -1)
    text_position_ids = None
    if position_ids.shape[0] == 4:
        text_position_ids, position_ids = position_ids[0], position_ids[1:]
    causal_mask = create_causal_mask(
        config=text.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    hidden_states = inputs_embeds
    position_embeddings = text.rotary_emb(hidden_states, position_ids)
    for layer_idx, decoder_layer in enumerate(text.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
        if layer_idx < len(deepstack_features):
            # DeepStack adds visual features at the image tokens; a scatter keeps the shapes static.
            features = deepstack_features[layer_idx].to(hidden_states.dtype)
            hidden_states = hidden_states + torch.zeros_like(hidden_states).masked_scatter(
                image_mask, features
            )
    return text.norm(hidden_states)


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
        rows = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
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
