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

"""Export WALL-X as one program: its Qwen2.5-VL prompt made once, its pixels and state made on every frame.

WALL-X's tokenizer step expands the image placeholders of the prompt with Python string edits and tokenizes
it, which the program cannot run. The recipe wraps that step in [`FixedPromptStep`]. Its per-frame inputs are
the pixel values, from the step's own resize and its processor's image processor, and the padded state with
its mask. The vision tower reads the patch grid prepared by an eager call inside `prepare_image_grid()`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import torch

from lerobot.export.recipe import ActionChunkRecipe, Part
from lerobot.processor import EnvTransition, PolicyProcessorPipeline, TransitionKey
from lerobot.utils.constants import OBS_STATE

from ..common.fixed_prompt import FixedPromptStep
from ..common.qwen_vision import prepare_image_grid
from .processor_wall_x import WallXTokenizerStep
from .utils import prepare_wall_x_image_inputs


class WallXExportRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] with WALL-X's tokenizer step made traceable, the vision tower on a fixed grid."""

    def program_preprocessor(self) -> PolicyProcessorPipeline:
        """The saved preprocessor with its tokenizer step wrapped in [`FixedPromptStep`]."""
        steps = [
            FixedPromptStep(step, _FrameInputs(step)) if isinstance(step, WallXTokenizerStep) else step
            for step in self.preprocessor.steps
        ]
        return replace(self.preprocessor, steps=steps)

    def parts(self) -> Iterator[Part]:
        """The default recipe's part, checked inside `prepare_image_grid()` so the checks prepare the grid."""
        with prepare_image_grid():
            (part,) = super().parts()
        yield part


class _FrameInputs:
    """The pixel values and state tensors WALL-X's tokenizer step computes from a frame."""

    def __init__(self, step: WallXTokenizerStep) -> None:
        self.step = step

    def __call__(self, transition: EnvTransition) -> dict[str, torch.Tensor]:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation[OBS_STATE]
        image_keys = [key for key in self.step.image_keys if key in observation]
        images, _ = prepare_wall_x_image_inputs(observation, image_keys)
        processor, _ = self.step._get_processors()
        pixels = processor.image_processor(images=images, return_tensors="pt", device=state.device)
        proprioception, agent_pos_mask = self.step.state_inputs(state)
        return {
            "pixel_values": pixels["pixel_values"],
            "proprioception": proprioception,
            "agent_pos_mask": agent_pos_mask,
        }
