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

"""The FLUX3 export recipe: the default program, with FLUX3's two starting noises in its one noise input.

FLUX3 starts each chunk from two noises, its video latents and its actions, and an exported program takes both
as inputs. A bundle gives its program one noise, so the program takes the two flattened and joined, video first
as the policy draws them, and splits them back.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from torch import Tensor

from lerobot.export.recipe import ActionChunkRecipe
from lerobot.policies import PreTrainedPolicy
from lerobot.processor import policy_output_to_transition, transition_to_policy_action

from .f3.packing import LATENT_CHANNELS


class Flux3ExportRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] with FLUX3's video and action noise joined in the one noise input."""

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

    def _sampled_noise_shape(self, frame: dict[str, np.ndarray]) -> tuple[int, ...]:
        return (sum(math.prod(shape) for shape in _noise_shapes(self.policy)),)


def _noise_shapes(policy: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """One sample's starting video latents and actions, shaped as `Flux3Policy.predict_action_chunk` takes them."""
    config = policy.config
    video = (1, LATENT_CHANNELS, policy.packer.predicted_latent_frames(config), *config.latent_hw)
    return video, (1, config.action_dim, config.chunk_size)
