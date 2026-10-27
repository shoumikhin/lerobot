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

pytest.importorskip("diffusers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel  # noqa: E402
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig  # noqa: E402
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import DiffusionObjective  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402


class Sampler(torch.nn.Module):
    def __init__(self, model: DiffusionModel):
        super().__init__()
        self.model = model

    def forward(self, global_cond, noise):
        return self.model.conditional_sample(noise.shape[0], global_cond=global_cond, noise=noise)


@pytest.mark.parametrize("scheduler", ["DDPM", "DDIM"])
def test_exported_diffusion_sampling_matches_eager(scheduler):
    config = DiffusionConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        horizon=8,
        n_action_steps=4,
        down_dims=(16, 32),
        diffusion_step_embed_dim=16,
        n_groups=4,
        noise_scheduler_type=scheduler,
        num_inference_steps=5,
    )
    model = DiffusionModel(config).eval()
    model.num_inference_steps = 10
    assert len(model.noise_scheduler.timesteps) == 10
    with pytest.raises(ValueError, match="at least 1"):
        model.num_inference_steps = 0
    sampler = Sampler(model)
    global_cond = torch.randn(2, 4 * config.n_obs_steps)
    noise = torch.randn(2, config.horizon, 2)

    torch.manual_seed(0)
    expected = sampler(global_cond, noise)
    # Strict tracing cannot read the timesteps as ints.
    exported = torch.export.export(sampler, (global_cond, noise), strict=False).module()
    torch.manual_seed(0)
    assert torch.equal(exported(global_cond, noise), expected)


class Denoiser(torch.nn.Module):  # a stand-in for the DiT
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)

    def forward(self, sample, timestep, conditioning_vec):
        return self.proj(sample) * timestep[:, None, None] / 100 + conditioning_vec[:, None]


@pytest.mark.parametrize("scheduler", ["DDPM", "DDIM"])
def test_exported_multi_task_dit_sampling_matches_eager(scheduler):
    config = MultiTaskDiTConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        objective="diffusion",
        noise_scheduler_type=scheduler,
        num_train_timesteps=100,
        num_inference_steps=10,
    )
    objective = DiffusionObjective(config, action_dim=2, horizon=8)
    with pytest.raises(ValueError, match="at least 1"):
        objective.num_inference_steps = 0

    class ObjectiveSampler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.denoiser = Denoiser()

        def forward(self, conditioning_vec, noise):
            return objective.conditional_sample(self.denoiser, noise.shape[0], conditioning_vec, noise=noise)

    sampler = ObjectiveSampler()
    conditioning_vec, noise = torch.randn(2, 2), torch.randn(2, 8, 2)
    torch.manual_seed(0)
    expected = sampler(conditioning_vec, noise)
    exported = torch.export.export(sampler, (conditioning_vec, noise), strict=False).module()
    torch.manual_seed(0)
    assert torch.equal(exported(conditioning_vec, noise), expected)
