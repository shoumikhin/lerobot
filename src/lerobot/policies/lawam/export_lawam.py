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

"""Export LaWAM as one program: its Qwen prompt made once at export, its pixels made on every frame.

LaWAM's preprocessor builds the Qwen3-VL inputs with the transformers processor, which the program cannot
run. The recipe wraps that step in [`FixedPromptStep`], with the pixel values computed by the saved
processor's own image processor as the only per-frame input. The vision tower reads the patch grid prepared
by an eager call inside `prepare_image_grid()`.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from lerobot.export.recipe import ActionChunkRecipe
from lerobot.processor import EnvTransition, PolicyProcessorPipeline, TransitionKey

from ..common.fixed_prompt import FixedPromptStep
from .processor_lawam import LaWAMQwenInputsProcessorStep


class LaWAMExportRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] with LaWAM's Qwen step made traceable and the vision tower on a prepared grid."""

    def program_preprocessor(self) -> PolicyProcessorPipeline:
        """The saved preprocessor with its Qwen step wrapped in [`FixedPromptStep`]."""
        steps = [
            FixedPromptStep(step, _PixelValues(step))
            if isinstance(step, LaWAMQwenInputsProcessorStep)
            else step
            for step in self.preprocessor.steps
        ]
        return replace(self.preprocessor, steps=steps)


class _PixelValues:
    """The pixel values LaWAM's Qwen step computes, from its processor's image processor alone."""

    def __init__(self, step: LaWAMQwenInputsProcessorStep) -> None:
        self.step = step

    def __call__(self, transition: EnvTransition) -> dict[str, torch.Tensor]:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        image_views, wrist_image_views = self.step.camera_views(observation)
        images = [
            image
            for primary, wrist in zip(image_views, wrist_image_views, strict=True)
            for image in (*primary, *wrist)
        ]
        processor, _ = self.step._ensure_processor()
        processed = processor.image_processor(
            images=images, do_rescale=False, device=images[0].device, return_tensors="pt"
        )
        return {"pixel_values": processed["pixel_values"]}
