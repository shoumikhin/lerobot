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

"""The fixed-grid Qwen2.5-VL helpers must give exactly what the stock transformers code gives."""

from copy import deepcopy

import pytest
import torch

pytest.importorskip("transformers")

import transformers  # noqa: E402
from packaging.version import Version  # noqa: E402
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig  # noqa: E402
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)

from lerobot.policies.common.qwen2_5_vision import Qwen2_5VisionGrid, qwen2_5_vision_forward  # noqa: E402
from lerobot.policies.common.qwen_vision import qwen_image_rope_index  # noqa: E402
from tests.policies.common.fixed_grid_checks import (  # noqa: E402
    count_block_calls_with_checkpointing,
    requires_gloo,
    run_with_sharded_blocks,
)

IMAGE_TOKEN_ID = 99
GRID = [[1, 8, 12]]


def _tiny_config() -> Qwen2_5_VLConfig:
    config = Qwen2_5_VLConfig(
        vision_config={
            "depth": 4,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 48,
            "fullatt_block_indexes": [1, 3],
            "window_size": 112,
        },
        text_config={"hidden_size": 48, "num_hidden_layers": 1, "num_attention_heads": 2, "vocab_size": 128},
        image_token_id=IMAGE_TOKEN_ID,
    )
    config.vision_config._attn_implementation = "sdpa"
    return config


def _tiny_visual():
    torch.manual_seed(0)
    return Qwen2_5_VisionTransformerPretrainedModel._from_config(_tiny_config().vision_config).eval()


def _pixels(grid_thw: torch.Tensor) -> torch.Tensor:
    return torch.randn(int(grid_thw.prod(-1).sum()), 3 * 2 * 14 * 14)


class FixedGridVision(torch.nn.Module):
    def __init__(self, visual, grid):
        super().__init__()
        self.visual = visual
        self.grid = grid

    def forward(self, pixel_values, grid_thw):
        grid = Qwen2_5VisionGrid.reuse_or_build(self.grid, self.visual, grid_thw)
        return qwen2_5_vision_forward(self.visual, pixel_values, grid)


# Two images of 18x18 patches (as 224 px cameras give) have uneven attention windows; the
# two-frame grid checks the temporal axis.
@pytest.mark.parametrize("grids", [[[1, 18, 18], [1, 18, 18]], [[2, 8, 12], [1, 16, 22]]])
def test_vision_forward_matches_stock_and_exports(grids):
    visual = _tiny_visual()
    grid_thw = torch.tensor(grids)
    pixel_values = _pixels(grid_thw)
    with torch.no_grad():
        expected = visual(pixel_values, grid_thw=grid_thw).pooler_output
        model = FixedGridVision(visual, Qwen2_5VisionGrid.build(visual, grid_thw))
        exported = torch.export.export(model, (pixel_values, grid_thw))
        torch.testing.assert_close(model(pixel_values, grid_thw), expected, rtol=0, atol=0)
        torch.testing.assert_close(exported.module()(pixel_values, grid_thw), expected, rtol=0, atol=0)


def test_export_without_a_cached_grid_asks_for_an_eager_call():
    visual = _tiny_visual()
    grid_thw = torch.tensor(GRID)
    with pytest.raises(RuntimeError, match="Run one eager call"):
        torch.export.export(FixedGridVision(visual, None), (_pixels(grid_thw), grid_thw))


def test_exported_program_rejects_a_changed_grid():
    visual = _tiny_visual()
    grid_thw = torch.tensor(GRID)
    model = FixedGridVision(visual, Qwen2_5VisionGrid.build(visual, grid_thw))
    with pytest.raises(ValueError, match="image grid differs"):
        model(_pixels(grid_thw), torch.tensor([[1, 12, 8]]))
    with torch.no_grad():
        program = torch.export.export(model, (_pixels(grid_thw), grid_thw)).module()
    program(_pixels(grid_thw), grid_thw)
    with pytest.raises(RuntimeError):
        program(_pixels(grid_thw), torch.tensor([[1, 12, 8]]))


@pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.5.4"),
    reason="transformers before 5.5.4 also scales the temporal position of still images",
)
def test_image_rope_index_matches_stock():
    model = Qwen2_5_VLModel._from_config(_tiny_config())
    grids = [[1, 4, 6], [1, 4, 4], [1, 6, 2]]
    text = lambda n: list(range(1, n + 1))  # noqa: E731
    image = lambda t, h, w: [IMAGE_TOKEN_ID] * (t * h * w // 4)  # noqa: E731
    rows = [
        text(3) + image(*grids[0]) + text(2) + image(*grids[1]) + text(4),
        text(1) + image(*grids[2]) + text(2),
    ]
    length = max(map(len, rows)) + 1
    input_ids = torch.zeros(len(rows), length, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, row in enumerate(rows):  # right padding for the first prompt, left padding for the second
        span = slice(0, len(row)) if i == 0 else slice(length - len(row), length)
        input_ids[i, span] = torch.tensor(row)
        attention_mask[i, span] = 1

    expected, _ = model.get_rope_index(
        input_ids,
        (input_ids == IMAGE_TOKEN_ID).int(),
        image_grid_thw=torch.tensor(grids),
        attention_mask=attention_mask,
    )
    position_ids = qwen_image_rope_index(input_ids, attention_mask, grids, 2, IMAGE_TOKEN_ID)

    assert torch.equal(position_ids, expected)


def test_reused_grid_follows_dtype_changes():
    visual = _tiny_visual()
    grid_thw = torch.tensor(GRID)
    pixel_values = _pixels(grid_thw)
    grid = Qwen2_5VisionGrid.reuse_or_build(None, visual, grid_thw)
    visual.to(torch.bfloat16)
    grid = Qwen2_5VisionGrid.reuse_or_build(grid, visual, grid_thw)
    pixel_values = pixel_values.bfloat16()
    with torch.no_grad():
        torch.testing.assert_close(
            qwen2_5_vision_forward(visual, pixel_values, grid),
            visual(pixel_values, grid_thw=grid_thw).pooler_output,
            rtol=0,
            atol=0,
        )


def test_grid_built_in_inference_mode_can_train():
    visual = _tiny_visual()
    grid_thw = torch.tensor(GRID)
    pixel_values = _pixels(grid_thw)
    with torch.inference_mode():
        grid = Qwen2_5VisionGrid.reuse_or_build(None, visual, grid_thw)
        qwen2_5_vision_forward(visual, pixel_values, grid)
    visual.train()
    qwen2_5_vision_forward(visual, pixel_values, grid).square().sum().backward()
    cached_grad = visual.patch_embed.proj.weight.grad.clone()
    visual.zero_grad()
    visual(pixel_values, grid_thw=grid_thw).pooler_output.square().sum().backward()
    torch.testing.assert_close(cached_grad, visual.patch_embed.proj.weight.grad, rtol=0, atol=0)


def test_gradient_checkpointing_reruns_every_block():
    visual = _tiny_visual()
    grid_thw = torch.tensor(GRID)
    pixel_values = _pixels(grid_thw)
    grid = Qwen2_5VisionGrid.build(visual, grid_thw)
    calls = count_block_calls_with_checkpointing(
        visual, lambda: qwen2_5_vision_forward(visual, pixel_values, grid)
    )
    assert calls == 2 * len(visual.blocks)


@requires_gloo
def test_vision_forward_runs_with_fsdp2_sharded_blocks(tmp_path):
    visual = _tiny_visual().train()
    grid_thw = torch.tensor(GRID)
    pixel_values = _pixels(grid_thw)
    expected = visual(pixel_values, grid_thw=grid_thw).pooler_output.detach()

    def forward(visual):
        grid = Qwen2_5VisionGrid.reuse_or_build(None, visual, grid_thw)
        return qwen2_5_vision_forward(visual, pixel_values, grid)

    actual = run_with_sharded_blocks(tmp_path, deepcopy(visual), forward)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
