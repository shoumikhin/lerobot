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

"""Export Multi-Task DiT as an encoder run once, a transformer step run once per denoising step, and the
actions."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ..common.export_denoising import EULER, DenoisingRecipe, Schedule, scheduler_schedule
from .modeling_multi_task_dit import ObservationEncoder


class MultiTaskDiTExportRecipe(DenoisingRecipe):
    """Multi-Task DiT: its last `n_obs_steps` frames in, its DDPM, DDIM or Euler sampling in a loop of programs.

    See `lerobot.policies.common.export_denoising` for the programs. Like the default recipe, the bundle runs
    the task it was exported for: the CLIP text encoder's features for that task are computed once and held in
    the encoder program, in place of the text encoder.
    """

    def _encoder(self) -> nn.Module:
        """The policy's observation encoder, with its text encoder replaced by the exported task's features.

        The policy keeps its own text encoder, so the parts are checked against the policy as it is.
        """
        encoder = self.policy.observation_encoder
        first_tick = {name: value[0] for name, value in self._case.frame.items()}
        batch = self.preprocessor(self._observation(first_tick, self.task))
        with torch.no_grad():
            features = encoder.text_encoder(batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK])
        # A copy of the module tree that shares every weight, with the features in place of the text encoder.
        memo = {id(tensor): tensor for tensor in (*encoder.parameters(), *encoder.buffers())}
        memo[id(encoder.text_encoder)] = _TaskFeatures(features)
        return _Conditioning(copy.deepcopy(encoder, memo))

    def _network(self) -> nn.Module:
        return self.policy.noise_predictor

    def _schedule(self) -> Schedule:
        config = self.policy.config
        if config.is_diffusion:
            objective = self.policy.objective
            # As `DiffusionObjective.conditional_sample` does before it samples.
            objective.noise_scheduler.set_timesteps(objective.num_inference_steps)
            return scheduler_schedule(objective.noise_scheduler)
        if config.integration_method != "euler":
            raise ValueError(f"Only Euler integration can be exported, not {config.integration_method!r}.")
        # The time grid of `FlowMatchingObjective.conditional_sample`, on the device the policy computes it on.
        grid = torch.linspace(0, 1, config.num_integration_steps + 1, device=self._device)
        times = [grid[i].item() for i in range(len(grid))]
        return Schedule(
            EULER,
            [np.array([t], np.float32) for t in times[:-1]],
            [np.array([(grid[i + 1] - grid[i]).item()], np.float32) for i in range(len(times) - 1)],
        )


class _TaskFeatures(nn.Module):
    """The text encoder's features for the one task the bundle runs, whatever tokens it is given."""

    def __init__(self, features: Tensor):
        super().__init__()
        self.register_buffer("features", features.detach().clone())

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.features


class _Conditioning(nn.Module):
    def __init__(self, encoder: ObservationEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        return self.encoder.encode(batch)
