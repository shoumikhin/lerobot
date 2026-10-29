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

"""pi0.5's denoising step, given the time conditioning instead of computing it with the time MLP."""

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d  # noqa: E402
from lerobot.policies.pi05 import PI05Config, modeling_pi05  # noqa: E402
from tests.policies.pi0_pi05.utils.meta_load import use_tiny_backbone  # noqa: E402


class Forbidden(nn.Module):
    def forward(self, x):
        raise AssertionError("the time MLP ran")


def test_denoise_step_takes_the_time_conditioning(monkeypatch):
    use_tiny_backbone(monkeypatch, modeling_pi05)
    torch.manual_seed(0)
    config = PI05Config(
        image_resolution=(28, 28), device="cpu", dtype=torch.float32, chunk_size=4, n_action_steps=4
    )
    model = modeling_pi05.PI05Pytorch(config).eval()
    width = model.paligemma_with_expert.paligemma.config.text_config.hidden_size
    prefix = torch.randn(1, 5, width)
    pad_masks = torch.ones(1, 5, dtype=torch.bool)
    masks = prepare_attention_masks_4d(make_att_2d_masks(pad_masks, torch.zeros_like(pad_masks)))
    with torch.no_grad():
        _, cache = model.paligemma_with_expert.forward(
            attention_mask=masks,
            position_ids=torch.arange(5)[None],
            inputs_embeds=[prefix, None],
            use_cache=True,
        )
        x_t = torch.randn(1, config.chunk_size, config.max_action_dim)
        timestep = torch.tensor([0.7])
        *_, cond = model.embed_suffix(x_t, timestep)
        expected = model.denoise_step(pad_masks, cache, x_t, timestep)
        model.time_mlp_in = Forbidden()
        actual = model.denoise_step(pad_masks, cache, x_t, timestep, cond)
    assert torch.equal(actual, expected)
