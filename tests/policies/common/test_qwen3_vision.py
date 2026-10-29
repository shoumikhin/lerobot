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

"""The fixed-grid Qwen3-VL helpers must give exactly what the stock transformers model gives."""

from copy import deepcopy

import pytest
import torch

pytest.importorskip("transformers")

from transformers import Qwen3VLModel  # noqa: E402

from lerobot.policies.common.qwen3_vision import (  # noqa: E402
    Qwen3VisionGrid,
    qwen3_vision_forward,
    qwen3_vl_forward,
)
from lerobot.policies.common.qwen_vision import qwen_image_rope_index  # noqa: E402
from tests.policies.common.fixed_grid_checks import (  # noqa: E402
    count_block_calls_with_checkpointing,
    requires_gloo,
    run_with_sharded_blocks,
)
from tests.policies.common.qwen3_vl_helpers import (  # noqa: E402
    IMAGE_GRIDS,
    IMAGE_TOKEN_ID,
    image_prompts,
    tiny_qwen3_vl_config,
)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Qwen3VLModel(tiny_qwen3_vl_config()).eval()


def _inputs(model):
    prompts = image_prompts(model.config)
    return tuple(prompts[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"))


def test_qwen3_vl_forward_matches_stock_model(model):
    input_ids, attention_mask, pixel_values, grid_thw = _inputs(model)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    grid = Qwen3VisionGrid.build(model.visual, grid_thw)
    image_mask = input_ids == IMAGE_TOKEN_ID
    with torch.no_grad():
        expected = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
        ).last_hidden_state
        actual = qwen3_vl_forward(model, inputs_embeds, attention_mask, pixel_values, grid, image_mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    class Wrapper(torch.nn.Module):
        def forward(self, inputs_embeds, attention_mask, pixel_values, image_mask):
            return qwen3_vl_forward(model, inputs_embeds, attention_mask, pixel_values, grid, image_mask)

    with torch.no_grad():
        program = torch.export.export(Wrapper(), (inputs_embeds, attention_mask, pixel_values, image_mask))
        torch.testing.assert_close(
            program.module()(inputs_embeds, attention_mask, pixel_values, image_mask), expected
        )


def test_image_rope_index_matches_get_rope_index(model):
    input_ids, attention_mask, pixel_values, grid_thw = _inputs(model)
    expected_ids, _ = model.get_rope_index(
        input_ids,
        mm_token_type_ids=(input_ids == IMAGE_TOKEN_ID).int(),
        image_grid_thw=grid_thw,
        attention_mask=attention_mask,
    )
    position_ids = qwen_image_rope_index(
        input_ids, attention_mask, grid_thw.tolist(), model.visual.spatial_merge_size, IMAGE_TOKEN_ID
    )
    assert torch.equal(position_ids, expected_ids)

    # An image block hidden by the mask (a missing camera) keeps its grid entry here; get_rope_index skips it.
    merge = model.visual.spatial_merge_size
    hidden_mask = attention_mask.clone()
    block_length = IMAGE_GRIDS[0][1] * IMAGE_GRIDS[0][2] // merge**2
    first_block = (input_ids[0] == IMAGE_TOKEN_ID).nonzero()[:block_length, 0]
    hidden_mask[0, first_block] = 0
    expected_hidden, _ = model.get_rope_index(
        input_ids,
        mm_token_type_ids=(input_ids == IMAGE_TOKEN_ID).int(),
        image_grid_thw=grid_thw[1:],
        attention_mask=hidden_mask,
    )
    actual_hidden = qwen_image_rope_index(input_ids, hidden_mask, grid_thw.tolist(), merge, IMAGE_TOKEN_ID)
    assert torch.equal(actual_hidden, expected_hidden)

    grid = Qwen3VisionGrid.build(model.visual, grid_thw)
    inputs_embeds = model.get_input_embeddings()(input_ids)
    with torch.no_grad():
        model.rope_deltas = None
        expected = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            mm_token_type_ids=(input_ids == IMAGE_TOKEN_ID).int(),
        ).last_hidden_state
        actual = qwen3_vl_forward(
            model,
            inputs_embeds,
            attention_mask,
            pixel_values,
            grid,
            input_ids == IMAGE_TOKEN_ID,
            position_ids=position_ids,
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cached_grid_keeps_learned_positions_trainable(model):
    visual = deepcopy(model.visual)
    _, _, pixel_values, grid_thw = _inputs(model)
    with torch.inference_mode():  # as when a prediction builds the cache before training
        grid = Qwen3VisionGrid.build(visual, grid_thw)
    for _ in range(2):  # A second backward fails if the grid holds an autograd graph.
        visual.zero_grad()
        qwen3_vision_forward(visual, pixel_values, grid)[0].square().sum().backward()
    cached_grad = visual.pos_embed.weight.grad.clone()
    visual.zero_grad()
    visual(pixel_values, grid_thw=grid_thw).pooler_output.square().sum().backward()
    torch.testing.assert_close(cached_grad, visual.pos_embed.weight.grad)

    with torch.no_grad():
        visual.pos_embed.weight.add_(1.0)
        torch.testing.assert_close(
            qwen3_vision_forward(visual, pixel_values, grid)[0],
            visual(pixel_values, grid_thw=grid_thw).pooler_output,
            rtol=0,
            atol=0,
        )


def test_reused_grid_follows_dtype_changes(model):
    visual = deepcopy(model.visual)
    _, _, pixel_values, grid_thw = _inputs(model)
    grid = Qwen3VisionGrid.reuse_or_build(None, visual, grid_thw)
    visual.to(torch.bfloat16)
    grid = Qwen3VisionGrid.reuse_or_build(grid, visual, grid_thw)
    with torch.no_grad():
        torch.testing.assert_close(
            qwen3_vision_forward(visual, pixel_values, grid)[0],
            visual(pixel_values.bfloat16(), grid_thw=grid_thw).pooler_output,
            rtol=0,
            atol=0,
        )


def test_gradient_checkpointing_reruns_every_block(model):
    visual = deepcopy(model.visual)
    _, _, pixel_values, grid_thw = _inputs(model)
    grid = Qwen3VisionGrid.build(visual, grid_thw)
    calls = count_block_calls_with_checkpointing(
        visual, lambda: qwen3_vision_forward(visual, pixel_values, grid)[0]
    )
    assert calls == 2 * len(visual.blocks)


@requires_gloo
def test_vision_forward_runs_with_fsdp2_sharded_blocks(model, tmp_path):
    visual = deepcopy(model.visual).train()
    _, _, pixel_values, grid_thw = _inputs(model)
    expected = visual(pixel_values, grid_thw=grid_thw).pooler_output.detach()

    def forward(visual):
        grid = Qwen3VisionGrid.reuse_or_build(None, visual, grid_thw)
        return qwen3_vision_forward(visual, pixel_values, grid)[0]

    actual = run_with_sharded_blocks(tmp_path, deepcopy(visual), forward)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
