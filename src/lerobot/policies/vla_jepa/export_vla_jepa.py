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
renders a chat template and tokenizes it in Python, so a program cannot run it. The program's preprocessor
makes those inputs in a last step, with the policy's own `qwen_inputs`, run in [`FixedPromptStep`], with the
pixel values, computed by the processor's own image processor, as the only per-frame input. The checks compare
the program with the saved preprocessor and the policy making its own inputs, as rollout runs them. The vision
tower runs on the image grid of the export's frames ([`VisionGridRecipe`]).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.export.recipe import VisionGridRecipe
from lerobot.policies import PreTrainedPolicy
from lerobot.processor import (
    EnvTransition,
    PolicyProcessorPipeline,
    ProcessorStep,
    TransitionKey,
    transition_to_batch,
)

from ..common.fixed_prompt import FixedPromptStep
from ..common.qwen3_vision import Qwen3VisionGrid
from .modeling_vla_jepa import VLAJEPAPolicy


class VLAJEPAExportRecipe(VisionGridRecipe):
    """[`VisionGridRecipe`] with VLA-JEPA's Qwen inputs made in the preprocessor, the vision tower on a fixed grid."""

    grid_type = Qwen3VisionGrid

    def program_preprocessor(self) -> PolicyProcessorPipeline:
        """The saved preprocessor with VLA-JEPA's Qwen inputs made in a last step, in [`FixedPromptStep`]."""
        step = _QwenInputsStep(cast(VLAJEPAPolicy, self.policy))
        steps = [*self.preprocessor.steps, FixedPromptStep(step, step.pixel_values)]
        return replace(self.preprocessor, steps=steps)

    def predict_chunk(self, policy: PreTrainedPolicy, batch: dict[str, Any], noise: Tensor | None) -> Tensor:
        """The default recipe's chunk; the grid comes from the Qwen inputs the policy makes for the batch."""
        if self._vision_grid is None and "image_grid_thw" not in batch:
            grid_thw = cast(VLAJEPAPolicy, policy).qwen_inputs(batch)["image_grid_thw"]
            self._vision_grid = self.grid_type.build(self.vision_tower(policy), grid_thw)
        return super().predict_chunk(policy, batch, noise)

    def _reads_task(self, frame: dict[str, np.ndarray]) -> bool:
        # VLA-JEPA writes the task into its Qwen prompt itself, which the saved preprocessor cannot show.
        return True

    def vision_tower(self, policy: PreTrainedPolicy) -> nn.Module:
        """The policy's Qwen vision tower."""
        return policy.model.qwen.model.model.visual


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
