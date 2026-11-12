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

"""Run a transformers Qwen3-VL model on a fixed image grid (see `qwen_vision.py`).

`Qwen3VisionGrid` derives the rope positions and the position-embedding interpolation of a grid with
transformers' `vision_utils`, as the stock forward does.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import torch
from torch import Tensor

from lerobot.utils.import_utils import _transformers_available

from .input_checks import check_input
from .qwen_vision import FixedImageGrid, replay_vision_block

if TYPE_CHECKING or _transformers_available:
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLModel,
        Qwen3VLTextModel,
        Qwen3VLVisionModel,
        apply_rotary_pos_emb_vision,
        eager_attention_forward,
    )
    from transformers.vision_utils import (
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )
else:
    create_causal_mask = None
    Qwen3VLModel = None
    Qwen3VLTextModel = None
    Qwen3VLVisionModel = None
    apply_rotary_pos_emb_vision = None
    eager_attention_forward = None
    get_vision_interpolation_indices_and_weights = None
    get_vision_position_ids = None


@dataclass
class Qwen3VisionGrid(FixedImageGrid):
    """What a Qwen3-VL vision tower derives from one image grid, as its stock forward computes it.

    The learned position embeddings are not cached, only which rows they interpolate, so training and
    weight updates always use the current `pos_embed`.
    """

    pos_embed_indices: Tensor
    pos_embed_weights: Tensor
    position_embeddings: tuple[Tensor, Tensor]
    chunk_lengths: list[int]

    @classmethod
    def _from_grid(cls, visual: "Qwen3VLVisionModel", grid_thw: list[list[int]]) -> Self:
        grid = torch.tensor(grid_thw, device=visual.device)
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid,
            num_grid_per_side=visual.num_grid_per_side,
            mode=visual.interpolation_mode,
            align_corners=visual.interpolation_align_corners,
            spatial_merge_size=visual.config.spatial_merge_size,
        )
        position_ids = get_vision_position_ids(grid, visual.spatial_merge_size)
        return cls(
            grid_thw=grid_thw,
            device=visual.device,
            dtype=visual.dtype,
            pos_embed_indices=indices,
            pos_embed_weights=weights,
            position_embeddings=visual.rotary_pos_emb(visual.pos_embed.weight, position_ids),
            chunk_lengths=[h * w for t, h, w in grid_thw for _ in range(t)],
        )


def qwen3_vision_forward(
    visual: "Qwen3VLVisionModel", pixel_values: Tensor, grid_thw: Tensor, grid: Qwen3VisionGrid
) -> tuple[Tensor, list[Tensor]]:
    """The merged image embeddings and DeepStack features of `visual(pixel_values, grid_thw)`.

    They are computed from `grid`, `Qwen3VisionGrid.build(visual, grid_thw)`, so that `torch.export` can trace
    them. A grid built for another image grid is refused.
    """
    grid.check(visual, grid_thw)
    hidden_states = visual.patch_embed(pixel_values)
    pos_embeds = (visual.pos_embed(grid.pos_embed_indices) * grid.pos_embed_weights[:, :, None]).sum(1)
    hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
    deepstack_features = []
    for layer_num, block in enumerate(visual.blocks):
        hidden_states = replay_vision_block(
            block,
            hidden_states,
            grid.position_embeddings,
            grid.chunk_lengths,
            apply_rotary_pos_emb_vision,
            eager_attention_forward,
        )
        if layer_num in visual.deepstack_visual_indexes:
            merger = visual.deepstack_merger_list[visual.deepstack_visual_indexes.index(layer_num)]
            deepstack_features.append(merger(hidden_states))
    return visual.merger(hidden_states), deepstack_features


def qwen3_vl_image_embeds(
    model: "Qwen3VLModel",
    inputs_embeds: Tensor,
    pixel_values: Tensor,
    image_grid_thw: Tensor,
    grid: Qwen3VisionGrid,
    image_mask: Tensor,
) -> tuple[Tensor, list[Tensor]]:
    """`inputs_embeds` with the image features at the image tokens, and the DeepStack features, as
    `qwen3_vl_forward` hands them to the text model. `image_mask` is True at the image placeholder tokens."""
    image_embeds, deepstack_features = qwen3_vision_forward(model.visual, pixel_values, image_grid_thw, grid)
    check_input(image_mask.sum() == image_embeds.shape[0], "Image features and image tokens do not match.")
    inputs_embeds = inputs_embeds.masked_scatter(
        image_mask[..., None].expand_as(inputs_embeds), image_embeds.to(inputs_embeds.dtype)
    )
    return inputs_embeds, deepstack_features


def qwen3_vl_forward(
    model: "Qwen3VLModel",
    inputs_embeds: Tensor,
    attention_mask: Tensor | None,
    pixel_values: Tensor,
    image_grid_thw: Tensor,
    grid: Qwen3VisionGrid,
    image_mask: Tensor,
    position_ids: Tensor | None = None,
) -> Tensor:
    """`Qwen3VLModel.forward` for images and no cache, on `grid` (see `qwen3_vision_forward`), traceable by
    `torch.export` while the text model uses SDPA or eager attention.

    `image_mask` is True at the image placeholder tokens. `position_ids` are 3 rows of rope positions, as
    `qwen_image_rope_index` gives them, or 4 rows whose first row holds the text positions (the text model
    splits it off); when None, positions are 0, 1, 2, ... as in the stock forward without `input_ids`.
    Once a call with `input_ids` (or `generate`) has cached rope deltas on the model, the stock forward
    adds them to those positions, so a call without `position_ids` is refused. Returns the last hidden
    state.
    """
    if position_ids is None and model.rope_deltas is not None:
        raise ValueError(
            "The model holds rope deltas from a call with input_ids, which the stock forward would add to the "
            "positions; pass position_ids."
        )
    inputs_embeds, deepstack_features = qwen3_vl_image_embeds(
        model, inputs_embeds, pixel_values, image_grid_thw, grid, image_mask
    )
    return model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        visual_pos_masks=image_mask,
        deepstack_visual_embeds=deepstack_features,
    ).last_hidden_state


def qwen3_text_layers(
    model: "Qwen3VLTextModel",
    hidden_states: Tensor,
    attention_mask: Tensor | None,
    image_mask: Tensor,
    deepstack_features: list[Tensor],
    layers: range,
    position_ids: Tensor | None = None,
) -> Tensor:
    """The hidden states after `layers` of `model.forward`, the text model of a Qwen3-VL model, without its final
    norm, so that a chain of programs can run the text model a few layers at a time.

    `hidden_states` are the inputs of the first layer in `layers`. `deepstack_features` are the DeepStack features
    of `qwen3_vl_image_embeds`, of which each layer adds its own; a range past them can pass none. `position_ids`
    are as in `qwen3_vl_forward`, None for 0, 1, 2, ...
    """
    if position_ids is None:
        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        position_ids = position_ids.view(1, 1, -1).expand(4, hidden_states.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
    text_position_ids = None
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids, position_ids = position_ids[0], position_ids[1:]
    mask = create_causal_mask(
        config=model.config,
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    position_embeddings = model.rotary_emb(hidden_states, position_ids)
    for index in layers:
        hidden_states = model.layers[index](
            hidden_states,
            attention_mask=mask,
            position_ids=text_position_ids,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
        if index < len(deepstack_features):
            hidden_states = model._deepstack_process(hidden_states, image_mask, deepstack_features[index])
    return hidden_states
