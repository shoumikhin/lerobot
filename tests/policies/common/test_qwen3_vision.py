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

import pytest
import torch

pytest.importorskip("transformers")

from transformers import Qwen3VLConfig, Qwen3VLModel  # noqa: E402

from lerobot.policies.common.qwen3_vision import (  # noqa: E402
    Qwen3VisionGrid,
    qwen3_image_rope_index,
    qwen3_vl_forward,
)

IMAGE_TOKEN_ID = 7
IMAGE_GRIDS = [[1, 4, 4], [1, 4, 6]]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = Qwen3VLConfig(
        vision_config={
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "patch_size": 4,
            "out_hidden_size": 64,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [4, 2, 2],
                "mrope_interleaved": True,
            },
        },
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=IMAGE_TOKEN_ID + 1,
    )
    return Qwen3VLModel(config).eval()


def _inputs(model):
    """Two left-padded prompts of different lengths: text, image, text, image, text."""
    merge = model.config.vision_config.spatial_merge_size
    rows = []
    for prefix in ([1, 2], [1, 2, 3, 4, 5]):
        ids = list(prefix)
        for _, h, w in IMAGE_GRIDS:
            ids += [IMAGE_TOKEN_ID] * (h * w // merge**2) + [3]
        rows.append(ids + [4, 5])
    length = max(len(r) for r in rows)
    input_ids = torch.tensor([[0] * (length - len(r)) + r for r in rows])
    attention_mask = torch.tensor([[0] * (length - len(r)) + [1] * len(r) for r in rows])
    grid_thw = torch.tensor(IMAGE_GRIDS * len(rows))
    vision = model.config.vision_config
    patch_dim = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    pixel_values = torch.randn(int(grid_thw.prod(-1).sum()), patch_dim)
    return input_ids, attention_mask, pixel_values, grid_thw


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


def test_qwen3_image_rope_index_matches_get_rope_index(model):
    input_ids, attention_mask, pixel_values, grid_thw = _inputs(model)
    expected_ids, _ = model.get_rope_index(
        input_ids,
        mm_token_type_ids=(input_ids == IMAGE_TOKEN_ID).int(),
        image_grid_thw=grid_thw,
        attention_mask=attention_mask,
    )
    position_ids = qwen3_image_rope_index(
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
    actual_hidden = qwen3_image_rope_index(input_ids, hidden_mask, grid_thw.tolist(), merge, IMAGE_TOKEN_ID)
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
