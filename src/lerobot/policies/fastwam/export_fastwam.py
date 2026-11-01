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

"""The FastWAM export recipe: the default program, holding the task FastWAM reads and any noise it seeds.

FastWAM writes the task into a prompt that it encodes itself, so its preprocessor's output does not show the
task. The program holds that task, so a bundle runs the task it was exported for. With `inference_seed` set,
FastWAM draws its starting noise from a generator it seeds each time it predicts a chunk, so every chunk
starts from the same noise; the program then holds that noise, as `lerobot-rollout` uses it. Without it, the
program takes the noise as an input, as the default program does.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.export.recipe import ActionChunkRecipe

from .modeling_fastwam import FastWAMPolicy, _prompt_from_batch


class FastWAMExportRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] for FastWAM, which reads its task and, with `inference_seed`, seeds its noise."""

    policy: FastWAMPolicy

    def predict_chunk(self, policy: FastWAMPolicy, batch: dict[str, Any], noise: Tensor | None) -> Tensor:
        """The chunk FastWAM predicts from its task's text context, which torch.export cannot encode."""
        # The eager runs before export encode and cache the context, so the program holds it as a constant.
        context, context_mask = policy._encode_prompt_cached(_prompt_from_batch(batch, policy.config))
        batch = {**batch, "context": context, "context_mask": context_mask}
        return super().predict_chunk(policy, batch, noise)

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
