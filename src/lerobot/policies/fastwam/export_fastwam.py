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

"""The FastWAM export recipe: the default program, holding the task FastWAM reads and the noise it seeds.

FastWAM writes the task into a prompt that it encodes itself, so its preprocessor's output does not show the
task. It also draws its starting noise from a generator it seeds with `inference_seed` each time it predicts a
chunk, so every chunk starts from the same noise. The program holds both, so a bundle runs the task it was
exported for and starts every chunk from that noise, as `lerobot-rollout` does.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from lerobot.export.recipe import ActionChunkRecipe

from .modeling_fastwam import FastWAMPolicy


class FastWAMExportRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] for FastWAM, which reads its task and seeds its own noise."""

    policy: FastWAMPolicy

    def _reads_task(self, frame: dict[str, np.ndarray]) -> bool:
        # FastWAM encodes its prompt inside the policy, so the preprocessor's output does not depend on the task.
        return True

    def _seeded_noise(self) -> Tensor | None:
        config = self.policy.config
        if config.inference_seed is None:
            return None
        # The noise FastWAM draws when it is given none.
        generator = torch.Generator(device=config.rand_device).manual_seed(config.inference_seed)
        shape = (1, config.action_horizon, self.policy.model.action_expert.action_dim)
        noise = torch.randn(shape, generator=generator, device=config.rand_device, dtype=torch.float32)
        return noise.to(self._device)
