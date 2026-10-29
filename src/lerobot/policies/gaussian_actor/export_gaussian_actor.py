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

"""The Gaussian actor export recipe: one action per run, sampled from noise the bundle draws.

`lerobot-rollout` calls the actor's `select_action` every control tick. It samples a fresh action from a
tanh-squashed Gaussian with one standard normal draw. The program does the same, with that draw as its `noise`
input, so a bundle samples actions the way the PyTorch policy does, and its test case pins the draw.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from torch import Tensor
from torch.overrides import TorchFunctionMode

from lerobot.export.recipe import _RANDOM_NORMAL, ActionChunkRecipe
from lerobot.policies import PreTrainedPolicy


class GaussianActorExportRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] for an actor that returns one action per call instead of a chunk."""

    @staticmethod
    def predict_chunk(policy: PreTrainedPolicy, batch: dict[str, Any], noise: Tensor | None) -> Tensor:
        """The one action `select_action` samples, as a chunk of one, with `noise` as its normal draw."""
        if noise is None:
            return policy.select_action(batch)[:, None]
        with _ReplaceNormalDraw(noise):
            return policy.select_action(batch)[:, None]

    def _execution_steps(self) -> int:
        return 1

    def _sampled_noise_shape(self, frame: dict[str, np.ndarray]) -> tuple[int, ...] | None:
        return self._recorded_noise_shape(frame)


class _ReplaceNormalDraw(TorchFunctionMode):
    """Makes the one random normal tensor drawn while it is active `noise`, and refuses a second draw."""

    def __init__(self, noise: Tensor) -> None:
        super().__init__()
        self.noise: Tensor | None = noise

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func not in _RANDOM_NORMAL:
            return func(*args, **kwargs)
        if self.noise is None:
            raise RuntimeError("The policy drew a second random normal tensor; only one is replaced.")
        drawn = func(*args, **kwargs)
        if drawn.shape != self.noise.shape:
            raise RuntimeError(
                f"The policy drew a normal tensor shaped {drawn.shape}, not {self.noise.shape}."
            )
        noise, self.noise = self.noise.to(drawn.dtype), None
        if func is Tensor.normal_:
            return drawn.copy_(noise)
        return noise
