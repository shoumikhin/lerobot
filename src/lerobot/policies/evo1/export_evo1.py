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

"""The EVO1 export recipe: the default program, with EVO1's uniform starting noise made from normal noise.

EVO1 starts its flow from noise drawn uniformly from [-1, 1), while a bundle draws standard normal noise. For a
standard normal `z`, `erf(z / sqrt(2))` is uniform on (-1, 1), so the program maps its `noise` input that way and
samples from the same distribution as the PyTorch policy.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from lerobot.export.backends.executorch_tensorrt import ExecuTorchTensorRTConfig
from lerobot.export.recipe import ActionChunkRecipe
from lerobot.policies import PreTrainedPolicy

if TYPE_CHECKING:
    from lerobot.export.configs import ExportConfig
    from lerobot.processor import PolicyProcessorPipeline


class Evo1ExportRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] for EVO1, whose program takes standard normal noise."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        cfg: ExportConfig,
    ) -> None:
        """The recipe, with the vision-language model's eager attention on SDPA for TensorRT."""
        vlm = policy.model.embedder.model
        if isinstance(cfg.backend, ExecuTorchTensorRTConfig) and vlm.config._attn_implementation == "eager":
            # TensorRT computes the eager form wrong on Jetson Thor once the fixed prompt makes its mask constant
            # (NVIDIA/TensorRT#4956).
            vlm.set_attn_implementation("sdpa")
        super().__init__(policy, preprocessor, postprocessor, cfg)

    def predict_chunk(self, policy: PreTrainedPolicy, batch: dict[str, Any], noise: Tensor | None) -> Tensor:
        """The chunk EVO1 predicts, starting from `erf(noise / sqrt(2))` when `noise` is given."""
        if noise is not None:
            noise = torch.special.erf(noise / math.sqrt(2))
        return super().predict_chunk(policy, batch, noise)

    def _sampled_noise_shape(self, frame: dict[str, np.ndarray]) -> tuple[int, ...]:
        return (self.policy.config.chunk_size, self.policy.config.max_action_dim)

    def _reads_task(self, frame: dict[str, np.ndarray]) -> bool:
        # EVO1 tokenizes its task inside the policy, so the preprocessor's output does not depend on it.
        return True
