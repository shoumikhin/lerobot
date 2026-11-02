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

"""A recipe whose program reads two frames of history. Importing this module makes it ACT's recipe.

The program runs ACT on the older of its two frames, so the bundle's check, which replays a test case of two
ticks, passes only if the engine gives the programs their frames oldest first.
"""

import sys
from collections.abc import Iterator

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.export.recipe import ActionChunkRecipe, Part, RunPlan, TestCase, random_frame

TICKS = 2


class _Older(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *frames: Tensor) -> Tensor:
        return self.module(*(frame[0] for frame in frames))


class ACTHistoryRecipe(ActionChunkRecipe):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        frames = [self._case.frame, random_frame(self._features, self._generator)]
        ticks = {name: np.stack([frame[name] for frame in frames]) for name in self._features}
        self._ticks = TestCase(ticks, None, self._case.actions, self._case.action_scale)

    def parts(self) -> Iterator[Part]:
        (part,) = super().parts()
        inputs = tuple(torch.from_numpy(x) for x in self._ticks.frame.values())
        yield Part(part.name, _Older(part.module), inputs, part.inputs, part.outputs)

    def test_case(self) -> TestCase:
        return self._ticks

    def plan(self) -> RunPlan:
        return RunPlan(history=dict.fromkeys(self._features, TICKS))


ACTExportRecipe = ACTHistoryRecipe
sys.modules["lerobot.policies.act.export_act"] = sys.modules[__name__]
