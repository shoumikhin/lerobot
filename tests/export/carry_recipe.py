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

"""A recipe whose program carries the sum of its noises from chunk to chunk. Importing this module makes it ACT's
recipe, as an `export_act.py` would.

ACT starts from a noise added to its actions, as in `noise_recipe`, but from the sum of the noises of all the
chunks since the last reset. So the bundle's check, which plays a test case of three chunks, passes only if the
engine passes the sum from each chunk to the next, starting from zeros.
"""

import sys
from collections.abc import Iterator

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.export.recipe import ActionChunkRecipe, Part, RunPlan, TestCase, random_frame
from lerobot.policies import PreTrainedPolicy

CHUNKS = 3
TOTAL = "noise_total"


class _Accumulate(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *inputs: Tensor) -> tuple[Tensor, Tensor]:
        *frame, noise, total = inputs
        total = total + noise
        return self.module(*frame, total), total


class ACTCarryRecipe(ActionChunkRecipe):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        frames = [
            self._case.frame,
            *(random_frame(self._features, self._generator) for _ in range(CHUNKS - 1)),
        ]
        noises = [self._case.noise, *(self._random_noise() for _ in range(CHUNKS - 1))]
        total = np.zeros_like(noises[0])
        for noise in noises:
            total = total + noise
        chunks = {name: np.stack([frame[name] for frame in frames]) for name in self._features}
        actions = self._reference(frames[-1], total)
        self._chunks = TestCase(chunks, np.stack(noises), actions, self._case.action_scale)

    @staticmethod
    def predict_chunk(policy: PreTrainedPolicy, batch: dict, noise: Tensor | None) -> Tensor:
        actions = ActionChunkRecipe.predict_chunk(policy, batch, None)
        return actions if noise is None else actions + noise

    def _sampled_noise_shape(self, frame: dict[str, np.ndarray]) -> tuple[int, ...]:
        return (self.policy.config.chunk_size, 1)

    def parts(self) -> Iterator[Part]:
        (part,) = super().parts()
        inputs = (*part.example_inputs, torch.zeros_like(part.example_inputs[-1]))
        yield Part(part.name, _Accumulate(part.module), inputs, [*part.inputs, TOTAL], [*part.outputs, TOTAL])

    def test_case(self) -> TestCase:
        return self._chunks

    def plan(self) -> RunPlan:
        return RunPlan(carry={TOTAL: self._noise_shape})


ACTExportRecipe = ACTCarryRecipe
sys.modules["lerobot.policies.act.export_act"] = sys.modules[__name__]
