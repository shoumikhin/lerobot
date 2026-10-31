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

"""Export EO-1 as one program: its Qwen2.5-VL prompt made once, its pixel values made on every frame.

EO-1's processor step renders the chat template and tokenizes it, and its default image processor reads the
pixels with NumPy, so a program cannot run it. The recipe wraps that step in [`FixedPromptStep`]. Its per-frame
input is the pixel values, computed with tensor operations exactly as the NumPy image processor computes them
for an image it does not resize. The state reaches the model through the `<|state_pad|>` embedding, so the
prompt does not depend on it. The vision tower reads the patch grid prepared by an eager call inside
`prepare_image_grid()`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from lerobot.export.recipe import BFLOAT16_STEPS_TOLERANCE, ActionChunkRecipe, Part
from lerobot.processor import EnvTransition, PolicyProcessorPipeline, TransitionKey
from lerobot.utils.import_utils import _transformers_available

from ..common.fixed_prompt import FixedPromptStep
from ..common.qwen_vision import prepare_image_grid
from .processor_eo1 import EO1PrepareModelMessagesStep, EO1QwenProcessorStep

if TYPE_CHECKING or _transformers_available:
    from transformers import Qwen2VLImageProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


class EO1ExportRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] with EO-1's processor step made traceable, the vision tower on a fixed grid."""

    # EO-1 runs ten bfloat16 denoising steps.
    bfloat16_tolerance = BFLOAT16_STEPS_TOLERANCE

    def program_preprocessor(self) -> PolicyProcessorPipeline:
        """The saved preprocessor with its processor step wrapped in [`FixedPromptStep`]."""
        (messages,) = [
            step for step in self.preprocessor.steps if isinstance(step, EO1PrepareModelMessagesStep)
        ]
        steps = [
            FixedPromptStep(step, _PixelValues(step, messages._image_keys))
            if isinstance(step, EO1QwenProcessorStep)
            else step
            for step in self.preprocessor.steps
        ]
        return replace(self.preprocessor, steps=steps)

    def parts(self) -> Iterator[Part]:
        """The default recipe's part, checked inside `prepare_image_grid()` so the checks prepare the grid."""
        with prepare_image_grid():
            (part,) = super().parts()
        yield part


class _PixelValues:
    """The pixel values EO-1's processor step computes from the camera images of a frame."""

    def __init__(self, step: EO1QwenProcessorStep, image_keys: list[str]) -> None:
        if step.use_fast_processor:
            raise ValueError(
                "The EO-1 recipe computes the pixels of EO-1's default NumPy image processor, and this "
                "checkpoint uses the fast one (`use_fast_processor=True`)."
            )
        image_processor = step._processor.image_processor
        self.image_keys = image_keys
        self.factor = image_processor.patch_size * image_processor.merge_size
        self.min_pixels = step.image_min_pixels or image_processor.size["shortest_edge"]
        self.max_pixels = step.image_max_pixels or image_processor.size["longest_edge"]
        # The processor's own rescale and normalize of every uint8 value, so a lookup gives its pixels bit for
        # bit on any device (float32 arithmetic does not: CUDA divides by a number as a multiply by its inverse).
        values = np.tile(np.arange(256, dtype=np.uint8), (3, 1))[:, None]
        rescaled = image_processor.rescale(values, image_processor.rescale_factor)
        normalized = image_processor.normalize(
            rescaled, image_processor.image_mean, image_processor.image_std
        )
        self.table = torch.from_numpy(normalized[:, 0])
        # The NumPy processor's own patch layout, run by its torch twin with the resize and arithmetic off.
        self.patches = Qwen2VLImageProcessor.from_dict(image_processor.to_dict())

    def __call__(self, transition: EnvTransition) -> dict[str, torch.Tensor]:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        batch_size = len(observation[self.image_keys[0]])
        images = [observation[key][row] for row in range(batch_size) for key in self.image_keys]
        for image in images:
            height, width = image.shape[-2:]
            size = smart_resize(height, width, self.factor, self.min_pixels, self.max_pixels)
            if size != (height, width):
                raise ValueError(
                    f"EO-1 exports camera images its processor does not resize: {height}x{width} becomes "
                    f"{size[0]}x{size[1]}, which the program cannot do as Pillow does. Record or crop the camera "
                    f"at a size such as {size[0]}x{size[1]}."
                )
        table = self.table.to(images[0].device)
        # The uint8 values EO1PrepareModelMessagesStep makes, as indices.
        values = [image.clamp(0, 1).mul(255.0).round().long() for image in images]
        images = [table.gather(1, value.flatten(1)).view(value.shape) for value in values]
        pixels = self.patches(
            images=images, do_resize=False, do_rescale=False, do_normalize=False, return_tensors="pt"
        )
        return {"pixel_values": pixels["pixel_values"]}
