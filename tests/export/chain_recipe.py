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

"""A recipe that runs ACT as a chain. Importing this module makes it ACT's recipe, as an `export_act.py` would.

`prepare` converts the frame and runs the preprocessor, `model` runs ACT and the postprocessor, and `scale`
runs twice, multiplying by `FACTOR` and then by its inverse, both read from the bundle's constants. So the
chain plays the policy's actions through a repeated program.
"""

import sys
from collections.abc import Iterator

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.bundle.manifest import ProgramRun
from lerobot.export.recipe import ActionChunkRecipe, Part, RunPlan, to_policy_input
from lerobot.utils.constants import ACTION


class _Prepare(nn.Module):
    def __init__(self, preprocessor, frame_names: list[str], visual: set[str]):
        super().__init__()
        self.preprocessor = preprocessor
        self.frame_names = frame_names
        self.visual = visual

    def forward(self, *frame: Tensor) -> tuple[Tensor, ...]:
        observation = {
            name: to_policy_input(x, name in self.visual)
            for name, x in zip(self.frame_names, frame, strict=True)
        }
        batch = self.preprocessor({**observation, "task": "", "robot_type": ""})
        return tuple(batch[name].contiguous() for name in self.frame_names)


class _Model(nn.Module):
    def __init__(self, policy, postprocessor, frame_names: list[str], execution_steps: int):
        super().__init__()
        self.policy = policy
        self.postprocessor = postprocessor
        self.frame_names = frame_names
        self.execution_steps = execution_steps

    def forward(self, *prepared: Tensor) -> Tensor:
        actions = self.policy.predict_action_chunk(dict(zip(self.frame_names, prepared, strict=True)))
        return self.postprocessor(actions[:, : self.execution_steps])[0].float()


class _Scale(nn.Module):
    def forward(self, x: Tensor, factor: Tensor) -> Tensor:
        return x * factor


class ACTChainRecipe(ActionChunkRecipe):
    FACTOR = 2.0

    def parts(self) -> Iterator[Part]:
        self._runs: list[ProgramRun] = []
        frame_names = list(self._features)
        prepared_names = [f"prepared.{name}" for name in frame_names]
        prepare = _Prepare(self.preprocessor, frame_names, self._visual)
        inputs = self._inputs(self._case.frame, None)
        with torch.no_grad():
            prepared = prepare(*inputs)
        self._runs.append(ProgramRun("prepare", "prepare", frame_names, prepared_names))
        yield Part("prepare", prepare, inputs, frame_names, prepared_names)
        model = _Model(self.policy, self.postprocessor, frame_names, self._steps)
        with torch.no_grad():
            actions = model(*prepared)
        self._runs.append(ProgramRun("model", "model", prepared_names, ["unscaled"]))
        yield Part("model", model, prepared, prepared_names, ["unscaled"])
        self._runs += [
            ProgramRun("scale_0", "scale", ["unscaled", "factor_0"], ["scaled_0"]),
            ProgramRun("scale_1", "scale", ["scaled_0", "factor_1"], [ACTION]),
        ]
        yield Part(
            "scale",
            _Scale(),
            (actions, torch.tensor([self.FACTOR], device=actions.device)),
            ["x", "factor"],
            ["scaled"],
        )

    def plan(self) -> RunPlan:
        return RunPlan(
            runs=self._runs,
            constants={
                "factor_0": np.array([self.FACTOR], np.float32),
                "factor_1": np.array([1 / self.FACTOR], np.float32),
            },
        )


ACTExportRecipe = ACTChainRecipe
sys.modules["lerobot.policies.act.export_act"] = sys.modules[__name__]
