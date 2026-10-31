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

"""The FLUX3 export recipe: the default program, holding the two starting noises FLUX3 seeds on every call.

FLUX3 starts each chunk from two noises, its video latents and its actions, drawn from a generator it seeds with
`inference_seed` each time, so every chunk starts from the same noise. The program holds the two flattened and
joined, video first as the policy draws them, and splits them back. It takes no noise input, so a bundle starts
every chunk from that same noise too.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.export.recipe import ActionChunkRecipe
from lerobot.policies import PreTrainedPolicy
from lerobot.processor import policy_output_to_transition, transition_to_policy_action

from .f3.packing import LATENT_CHANNELS


class Flux3ExportRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] with FLUX3's seeded video and action noise held in its program."""

    @staticmethod
    def predict_chunk(policy: PreTrainedPolicy, batch: dict[str, Any], noise: Tensor | None) -> Tensor:
        """The chunk FLUX3 predicts, with `noise` split into its starting video latents and actions."""
        kwargs: dict[str, Tensor] = {}
        if noise is not None:
            video_shape, action_shape = _noise_shapes(policy)
            video, action = noise.split([math.prod(video_shape), math.prod(action_shape)], dim=-1)
            kwargs = {"video_noise": video.reshape(video_shape), "noise": action.reshape(action_shape)}
        output = policy.predict_action_chunk(batch, **kwargs)
        return transition_to_policy_action(policy_output_to_transition(output))

    def _seeded_noise(self) -> Tensor:
        # The two noises FLUX3 draws when it is given none, in its order.
        generator = torch.Generator().manual_seed(self.policy.config.inference_seed)
        noise = [torch.randn(shape, generator=generator).flatten() for shape in _noise_shapes(self.policy)]
        return torch.cat(noise)[None].to(self._device)

    def _reads_task(self, frame: dict[str, np.ndarray]) -> bool:
        # FLUX3 encodes its task inside the policy, so the preprocessor's output does not depend on it.
        return True


def _noise_shapes(policy: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """One sample's starting video latents and actions, shaped as `Flux3Policy.predict_action_chunk` takes them."""
    config = policy.config
    video = (1, LATENT_CHANNELS, policy.packer.predicted_latent_frames(config), *config.latent_hw)
    return video, (1, config.action_dim, config.chunk_size)
