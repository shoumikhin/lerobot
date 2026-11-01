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

"""A recipe whose first chunk after a reset runs a program of its own. Importing this module makes it ACT's
recipe, as an `export_act.py` would.

ACT starts from a noise added to its actions, as in `carry_recipe`, from the sum of the noises of the chunks since
the last reset. Here the first chunk's program starts that sum, so it has no value before, and later chunks add to
it. The test case plays three chunks from a reset tick by tick, so the bundle's check passes only if the engine runs
the first chunk's program first and then passes the sum from chunk to chunk.
"""

import sys
from collections.abc import Iterator

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.bundle.manifest import NOISE, ProgramRun
from lerobot.export.recipe import ActionChunkRecipe, Part, RunPlan, TestCase, random_frame
from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION

CHUNKS = 3
TOTAL = "noise_total"


class _Begin(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *inputs: Tensor) -> tuple[Tensor, Tensor]:
        *frame, noise = inputs
        total = noise.clone()
        return self.module(*frame, total), total


class _Accumulate(_Begin):
    def forward(self, *inputs: Tensor) -> tuple[Tensor, Tensor]:
        *frame, noise, total = inputs
        total = total + noise
        return self.module(*frame, total), total


class ACTFirstRunsRecipe(ActionChunkRecipe):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        ticks = CHUNKS * self._steps
        frames = [
            self._case.frame,
            *(random_frame(self._features, self._generator) for _ in range(ticks - 1)),
        ]
        noises = [self._case.noise, *(self._random_noise() for _ in range(CHUNKS - 1))]
        total = np.zeros_like(noises[0])
        actions = []
        for chunk, noise in enumerate(noises):
            total = total + noise
            actions.append(self._reference(frames[chunk * self._steps], total))
        stacked = {name: np.stack([frame[name] for frame in frames]) for name in self._features}
        self._ticks = TestCase(stacked, np.stack(noises), np.concatenate(actions), self._case.action_scale)

    @staticmethod
    def predict_chunk(policy: PreTrainedPolicy, batch: dict, noise: Tensor | None) -> Tensor:
        actions = ActionChunkRecipe.predict_chunk(policy, batch, None)
        return actions if noise is None else actions + noise

    def _sampled_noise_shape(self, frame: dict[str, np.ndarray]) -> tuple[int, ...]:
        return (self.policy.config.chunk_size, 1)

    def parts(self) -> Iterator[Part]:
        (part,) = super().parts()
        yield Part("begin", _Begin(part.module), part.example_inputs, part.inputs, [ACTION, TOTAL])
        inputs = (*part.example_inputs, torch.zeros_like(part.example_inputs[-1]))
        yield Part("model", _Accumulate(part.module), inputs, [*part.inputs, TOTAL], [ACTION, TOTAL])

    def test_case(self) -> TestCase:
        return self._ticks

    def plan(self) -> RunPlan:
        frame = list(self._features)
        return RunPlan(
            runs=[ProgramRun("model", "model", [*frame, NOISE, TOTAL], [ACTION, TOTAL])],
            first_runs=[[ProgramRun("begin", "begin", [*frame, NOISE], [ACTION, TOTAL])]],
            carry={TOTAL: None},
        )


ACTExportRecipe = ACTFirstRunsRecipe
sys.modules["lerobot.policies.act.export_act"] = sys.modules[__name__]
