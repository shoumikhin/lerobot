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

"""The EVO1 export recipe: the default one program, with EVO1's uniform starting noise made from normal noise.

EVO1 starts its flow from noise drawn uniformly from [-1, 1), while a bundle draws standard normal noise. For
a standard normal `z`, `erf(z / sqrt(2))` is uniform on (-1, 1), so the program maps its `noise` input that
way and samples from the same distribution as the PyTorch policy.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.export.recipe import ActionChunkRecipe
from lerobot.policies import PreTrainedPolicy


class Evo1ExportRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] for EVO1, whose program takes standard normal noise."""

    @staticmethod
    def predict_chunk(policy: PreTrainedPolicy, batch: dict[str, Any], noise: Tensor | None) -> Tensor:
        """The chunk EVO1 predicts, starting from `erf(noise / sqrt(2))` when `noise` is given."""
        if noise is not None:
            noise = torch.special.erf(noise / math.sqrt(2))
        return policy.predict_action_chunk(batch, noise=noise)

    def _sampled_noise_shape(self, frame: dict[str, np.ndarray]) -> tuple[int, ...] | None:
        return (self.policy.config.chunk_size, self.policy.config.max_action_dim)
