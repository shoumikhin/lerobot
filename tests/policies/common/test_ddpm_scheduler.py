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

import pytest
import torch

from lerobot.policies.common.ddpm_scheduler import ExportableDDPMScheduler

diffusers = pytest.importorskip("diffusers")


class SamplingLoop(torch.nn.Module):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler
        self.model = torch.nn.Linear(4, 4)

    def forward(self, sample):
        for t in self.scheduler.timesteps.tolist():
            sample = self.scheduler.step(self.model(sample), t, sample).prev_sample
        return sample


@pytest.mark.parametrize("num_inference_steps", [1, 10])
@pytest.mark.parametrize("timestep_spacing", ["leading", "trailing", "linspace"])
def test_exported_sampling_matches_stock_scheduler(timestep_spacing, num_inference_steps):
    stock = diffusers.DDPMScheduler(num_train_timesteps=100, timestep_spacing=timestep_spacing)
    exportable = ExportableDDPMScheduler(**stock.config)
    stock.set_timesteps(num_inference_steps)
    exportable.set_timesteps(num_inference_steps)
    loop = SamplingLoop(stock)
    sample = torch.randn(2, 4)

    torch.manual_seed(0)
    expected = loop(sample)
    loop.scheduler = exportable
    # Non-strict, as torch.export defaults to on current PyTorch: strict tracing cannot read the timesteps as ints.
    exported = torch.export.export(loop, (sample,), strict=False).module()
    torch.manual_seed(0)
    assert torch.equal(exported(sample), expected)
