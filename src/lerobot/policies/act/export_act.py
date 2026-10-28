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

"""PROBE ONLY: ACT's chunk as a chain of two programs, to show that a recipe alone can split a policy.

`prepare` converts the frame and runs the preprocessor; `model` runs ACT and the postprocessor. The backends,
the runtime loaders, the engine and the rollout are unchanged.
"""

from collections.abc import Iterator

import torch
from torch import Tensor, nn

from lerobot.export.recipe import ActionChunkRecipe, Part
from lerobot.utils.constants import ACTION


class _Prepare(nn.Module):
    def __init__(self, preprocessor, frame_names: list[str], visual: set[str]):
        super().__init__()
        self.preprocessor = preprocessor
        self.frame_names = frame_names
        self.visual = visual

    def forward(self, *frame: Tensor) -> tuple[Tensor, ...]:
        observation = {
            name: ((x.float() / 255).permute(2, 0, 1).contiguous() if name in self.visual else x).unsqueeze(0)
            for name, x in zip(self.frame_names, frame, strict=True)
        }
        batch = self.preprocessor({**observation, "task": "", "robot_type": ""})
        # A program reads its inputs in the memory layout it was compiled for, so values passed between
        # programs are contiguous.
        return tuple(batch[name].contiguous() for name in self.frame_names)


class _Model(nn.Module):
    def __init__(self, policy, postprocessor, frame_names: list[str], steps: int):
        super().__init__()
        self.policy = policy
        self.postprocessor = postprocessor
        self.frame_names = frame_names
        self.steps = steps

    def forward(self, *prepared: Tensor) -> Tensor:
        actions = self.policy.predict_action_chunk(dict(zip(self.frame_names, prepared, strict=True)))
        return self.postprocessor(actions[:, : self.steps])[0].float()


class ACTExportRecipe(ActionChunkRecipe):
    """ACT as two programs connected by name: `prepare` outputs `prepared.<key>`, which `model` reads."""

    def parts(self) -> Iterator[Part]:
        frame_names = list(self._frame_shapes)
        prepared_names = [f"prepared.{name}" for name in frame_names]
        prepare = _Prepare(self.preprocessor, frame_names, self._visual)
        inputs = self._inputs(self._case.frame, None)
        with torch.no_grad():
            prepared = prepare(*inputs)
        yield Part("prepare", prepare, inputs, frame_names, prepared_names)
        model = _Model(self.policy, self.postprocessor, frame_names, self._steps)
        yield Part("model", model, prepared, prepared_names, [ACTION])
