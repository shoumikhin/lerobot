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

"""Export VLA-JEPA as one program: its Qwen3-VL prompt made once at export, its pixels made on every frame.

VLA-JEPA makes its Qwen3-VL inputs inside `predict_action_chunk`, with the transformers processor, which
renders a chat template and tokenizes it in Python, so a program cannot run it. The recipe makes those inputs
in a step it adds to the preprocessor, with the policy's own `qwen_inputs`, and the program runs that step in
[`FixedPromptStep`], with the pixel values, computed by the processor's own image processor, as the only
per-frame input. The vision tower reads the patch grid prepared by an eager call inside `prepare_image_grid()`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import torch

from lerobot.export.configs import ExportConfig
from lerobot.export.recipe import ActionChunkRecipe, Part
from lerobot.processor import (
    EnvTransition,
    PolicyProcessorPipeline,
    ProcessorStep,
    TransitionKey,
    transition_to_batch,
)

from ..common.fixed_prompt import FixedPromptStep
from ..common.qwen_vision import prepare_image_grid
from .modeling_vla_jepa import VLAJEPAPolicy


class VLAJEPAExportRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] with VLA-JEPA's Qwen inputs made in the preprocessor, the vision tower on a fixed grid."""

    def __init__(
        self,
        policy: VLAJEPAPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        cfg: ExportConfig,
    ) -> None:
        """Add the Qwen inputs step to the preprocessor, so the task check and the test case run it too."""
        steps = [*preprocessor.steps, _QwenInputsStep(policy)]
        super().__init__(policy, replace(preprocessor, steps=steps), postprocessor, cfg)

    def program_preprocessor(self) -> PolicyProcessorPipeline:
        """The preprocessor with its Qwen inputs step wrapped in [`FixedPromptStep`]."""
        steps = [
            FixedPromptStep(step, step.pixel_values) if isinstance(step, _QwenInputsStep) else step
            for step in self.preprocessor.steps
        ]
        return replace(self.preprocessor, steps=steps)

    def parts(self) -> Iterator[Part]:
        """The default recipe's part, checked inside `prepare_image_grid()` so the checks prepare the grid."""
        with prepare_image_grid():
            (part,) = super().parts()
        yield part


class _QwenInputsStep(ProcessorStep):
    """The Qwen3-VL processor tensors that `predict_action_chunk` would make, added to the complementary data."""

    def __init__(self, policy: VLAJEPAPolicy) -> None:
        self.policy = policy

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        comp = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        comp.update(self.policy.qwen_inputs(transition_to_batch(transition)))
        return {**transition, TransitionKey.COMPLEMENTARY_DATA: comp}

    def pixel_values(self, transition: EnvTransition) -> dict[str, torch.Tensor]:
        """The pixel values of the step's tensors, from the processor's image processor alone."""
        inputs = self.policy._prepare_model_inputs(transition_to_batch(transition), training=False)
        images = [image for views in inputs["images"] for image in views]
        qwen = self.policy.model.qwen
        processed = qwen.processor.image_processor(
            images=images, do_rescale=False, device=qwen.model.device, return_tensors="pt"
        )
        return {"pixel_values": processed["pixel_values"]}

    def transform_features(self, features):
        return features
