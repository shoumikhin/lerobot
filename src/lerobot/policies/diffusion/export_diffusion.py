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

"""Export Diffusion Policy as an encoder run once, a U-Net step run once per denoising step, and the
actions."""

from __future__ import annotations

from torch import Tensor, nn

from ..common.export_denoising import DenoisingRecipe, Schedule, scheduler_schedule
from .modeling_diffusion import DiffusionModel


class DiffusionExportRecipe(DenoisingRecipe):
    """Diffusion Policy: its last `n_obs_steps` frames in, its DDPM or DDIM sampling in a loop of programs.

    See `lerobot.policies.common.export_denoising` for the programs.
    """

    def _encoder(self) -> nn.Module:
        return _GlobalConditioning(self.policy.diffusion)

    def _network(self) -> nn.Module:
        return self.policy.diffusion.unet

    def _schedule(self) -> Schedule:
        diffusion = self.policy.diffusion
        # As `DiffusionModel.conditional_sample` does before it samples.
        diffusion.noise_scheduler.set_timesteps(diffusion.num_inference_steps)
        return scheduler_schedule(diffusion.noise_scheduler)


class _GlobalConditioning(nn.Module):
    def __init__(self, model: DiffusionModel):
        super().__init__()
        self.model = model

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        # The policy's own encoder, which it exposes under no public name.
        return self.model._prepare_global_conditioning(batch)
