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
its mask. The vision tower runs on the image grid of the export's frames ([`VisionGridRecipe`]).
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from lerobot.export.recipe import VisionGridRecipe
from lerobot.policies import PreTrainedPolicy
from lerobot.processor import EnvTransition, PolicyProcessorPipeline, TransitionKey
from lerobot.utils.constants import OBS_STATE

from ..common.fixed_prompt import FixedPromptStep
from ..common.qwen2_5_vision import Qwen2_5VisionGrid
from .processor_wall_x import WallXTokenizerStep
from .utils import prepare_wall_x_image_inputs


class WallXExportRecipe(VisionGridRecipe):
    """[`VisionGridRecipe`] with WALL-X's tokenizer step made traceable, the vision tower on a fixed grid."""

    # In bfloat16, eager PyTorch alone moved the test case's actions by up to 0.12 on a DGX Spark and 0.09 on an
    # A100, and correct bundles differed by up to 0.22 on the Spark and 0.125 on the A100. Zeroed actions or a
    # skipped layer moved them by 1.1, and swapped keys and values by 8.6: 0.5 leaves at least twice the margin on
    # both sides.
    bfloat16_tolerance = 0.5
    grid_type = Qwen2_5VisionGrid

    def program_preprocessor(self) -> PolicyProcessorPipeline:
        """The saved preprocessor with its tokenizer step wrapped in [`FixedPromptStep`]."""
        steps = [
            FixedPromptStep(step, _FrameInputs(step)) if isinstance(step, WallXTokenizerStep) else step
            for step in self.preprocessor.steps
        ]
        return replace(self.preprocessor, steps=steps)

    def vision_tower(self, policy: PreTrainedPolicy) -> nn.Module:
        """The policy's Qwen vision tower."""
        return policy.model.visual


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
