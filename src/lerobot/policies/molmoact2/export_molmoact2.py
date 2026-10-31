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

"""Export MolmoAct2 as one program: its prompt made once, the state's tokens and the pixels made on every frame.

MolmoAct2's pack step writes the state into the prompt, one `<state_k>` token per value, tokenizes the prompt in
Python and reads the camera pixels with NumPy, so a program cannot run it. The recipe wraps that step in
[`FixedPromptStep`]. Its per-frame inputs are computed with tensor operations exactly as the step computes them:

- the token ids: the prompt of the first eager frame, with the state's tokens written in from a table (see
  [`StatePromptTokens`]). They are special tokens, so the rest of the prompt does not depend on the state.
- the pixel values of a camera image the processor does not resize: its uint8 values looked up in a 256-entry
  table made by the processor's own functions, then cut into patches.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from lerobot.export.recipe import ActionChunkRecipe
from lerobot.processor import EnvTransition, PolicyProcessorPipeline, TransitionKey
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import _transformers_available

from ..common.fixed_prompt import FixedPromptStep
from ..common.qwen_vision import is_exporting
from ..common.state_prompt import StatePromptTokens
from .processor_molmoact2 import STATE_TOKEN_PREFIX, MolmoAct2PackInputsProcessorStep

if TYPE_CHECKING or _transformers_available:
    from .molmoact2_hf_model.image_processing_molmoact2 import normalize_image, resize_image


class MolmoAct2ExportRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] with MolmoAct2's pack step made traceable."""

    def program_preprocessor(self) -> PolicyProcessorPipeline:
        """The saved preprocessor with its pack step wrapped in [`FixedPromptStep`]."""
        steps = [
            FixedPromptStep(step, _FrameInputs(step))
            if isinstance(step, MolmoAct2PackInputsProcessorStep)
            else step
            for step in self.preprocessor.steps
        ]
        return replace(self.preprocessor, steps=steps)


class _FrameInputs:
    """The token ids and the pixel values MolmoAct2's pack step computes from a frame."""

    def __init__(self, step: MolmoAct2PackInputsProcessorStep) -> None:
        image_processor = step.processor.image_processor
        if image_processor.crop_mode != "resize":
            raise ValueError(
                "The MolmoAct2 recipe computes the pixels of the image processor's 'resize' crop mode, and this "
                f"checkpoint uses {image_processor.crop_mode!r}."
            )
        self.step = step
        self.size = (image_processor.size["height"], image_processor.size["width"])
        self.patch_size = image_processor.patch_size
        tokenizer = step.processor.tokenizer
        self.state_tokens = [
            [tokenizer.convert_tokens_to_ids(f"{STATE_TOKEN_PREFIX}{index}>")]
            for index in range(step.num_state_tokens)
        ]
        self.pad_id = tokenizer.pad_token_id
        self.prompt: StatePromptTokens | None = None
        # The processor's own resize, to the same size, and normalize of every uint8 value, so a lookup gives its
        # pixels bit for bit on any device (float32 arithmetic does not: CUDA divides as a multiply by the inverse).
        values = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
        resized = resize_image(values, [1, 256], image_processor.resample)
        normalized = normalize_image(resized, image_processor.image_mean, image_processor.image_std)
        self.table = torch.from_numpy(np.ascontiguousarray(normalized[0].T))

    def __call__(self, transition: EnvTransition) -> dict[str, Tensor]:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        bins = _state_bins(observation[OBS_STATE], self.step.num_state_tokens)
        if self.prompt is None:
            if is_exporting():
                raise ValueError("The prompt is made from an eager call before the program is traced.")
            ids = (self.step(transition)[TransitionKey.COMPLEMENTARY_DATA] or {})["input_ids"]
            self.prompt = StatePromptTokens.around(
                ids[0].tolist(), bins[0].tolist(), self.state_tokens, self.pad_id, device=bins.device
            )
        input_ids, _ = self.prompt(bins[0])
        return {"input_ids": input_ids[None], "pixel_values": self._pixels(observation)}

    def _pixels(self, observation: dict[str, Tensor]) -> Tensor:
        keys = self.step._resolve_image_keys(observation)
        images = [observation[key][row] for row in range(len(observation[keys[0]])) for key in keys]
        table = self.table.to(images[0].device)
        patches = []
        for image in images:
            height, width = image.shape[-2:]
            if (height, width) != self.size:
                raise ValueError(
                    f"MolmoAct2 exports camera images its processor does not resize: {height}x{width} becomes "
                    f"{self.size[0]}x{self.size[1]}, which the program cannot do bit for bit as the processor "
                    f"does on the CPU. Record or crop the camera at {self.size[0]}x{self.size[1]}."
                )
            # The uint8 values the pack step makes by truncating; rounding gives the same ones for a uint8 frame,
            # and also after a compiler divides by 255 as a multiply.
            values = image.mul(255.0).round().clamp(0, 255).long()
            pixels = table.gather(1, values.flatten(1)).view(values.shape)
            patches.append(_patches(pixels.permute(1, 2, 0), self.patch_size))
        return torch.cat(patches)


def _state_bins(state: Tensor, num_bins: int) -> Tensor:
    """The `<state_k>` index of each state value, as the pack step's `_build_discrete_state_string` computes it."""
    state = torch.nan_to_num(state.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    return ((state + 1.0) / 2.0 * float(num_bins - 1)).round().long().clamp(0, num_bins - 1)


def _patches(image: Tensor, patch_size: int) -> Tensor:
    """The image processor's `batch_pixels_to_patches` of one HWC image, as tensor operations."""
    height, width, channels = image.shape
    rows, columns = height // patch_size, width // patch_size
    patches = image.reshape(rows, patch_size, columns, patch_size, channels).permute(0, 2, 1, 3, 4)
    return patches.reshape(1, rows * columns, patch_size * patch_size * channels)
