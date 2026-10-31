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

"""Export Wall-OSS 0.5 as one program: its prompt made once, the state's bin numbers written into it on each frame.

Wall-OSS's tokenizer step writes the state into a Qwen2.5-VL prompt as 512-bin numbers and tokenizes it, and it
reads the camera images through Pillow, which a program cannot do. The recipe replaces that step with
[`_StatePromptStep`]: the tokens before and after the state come once from the saved step, and each bin number's
tokens from a table (see [`StatePromptTokens`]). That holds while the tokenizer splits a number inside the prompt as
it splits it alone, which is checked on every eager frame and on states beyond the first and last bins.

The model reads its action tokens at the end of the prompt, so the program left-pads the prompt to the longest
state, as the step pads a batch, and the padding is left out of attention. In float32 the actions then match
rollout's to rounding; ten bfloat16 denoising steps amplify that rounding as much as another attention kernel does,
so the program is held to the export's tolerance, the same as pi0.5's. The pixels come from the saved processor's
image processor, which runs in torch, for cameras whose long side is already the 448 pixels the step resizes to.
The vision tower reads the patch grid prepared by an eager call inside `prepare_image_grid()`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor

from lerobot.export.recipe import BFLOAT16_STEPS_TOLERANCE, ActionChunkRecipe, Part
from lerobot.processor import EnvTransition, PolicyProcessorPipeline, ProcessorStep, TransitionKey
from lerobot.utils.constants import OBS_STATE

from ..common.qwen_vision import is_exporting, prepare_image_grid
from ..common.state_prompt import StatePromptTokens, extreme_states
from .processor_wall_oss_05 import WALL_OSS_ACTION_TOKEN, WallOSS05TokenizerStep

# The long side the tokenizer step resizes every camera image to with Pillow.
IMAGE_SIZE = 448


class WallOSS05ExportRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] with Wall-OSS's prompt built in the program and held to the export's tolerance."""

    exact = False
    # Wall-OSS runs ten bfloat16 denoising steps.
    bfloat16_tolerance = BFLOAT16_STEPS_TOLERANCE

    def parts(self) -> Iterator[Part]:
        """The default recipe's part, checked inside `prepare_image_grid()` so the checks prepare the grid."""
        with prepare_image_grid():
            (part,) = super().parts()
        yield part

    def program_preprocessor(self) -> PolicyProcessorPipeline:
        """The saved preprocessor with its tokenizer step replaced by [`_StatePromptStep`]."""
        steps = [
            _StatePromptStep(step) if isinstance(step, WallOSS05TokenizerStep) else step
            for step in self.preprocessor.steps
        ]
        return replace(self.preprocessor, steps=steps)


class _StatePromptStep(ProcessorStep):
    """Wall-OSS's tokenizer step for one task and camera size, as tensor operations.

    The prompt's tokens before and after the state, and the step's inputs that do not change with the frame, are
    the saved step's on the first eager frame. Every eager frame is checked against the saved step, and so are
    states beyond the first and last bins on the first one.
    """

    def __init__(self, step: WallOSS05TokenizerStep) -> None:
        self.step = step
        processor = step._get_processor()
        self.image_processor = processor.image_processor
        self.pad_id = processor.tokenizer.pad_token_id
        self.action_token_id = processor.tokenizer.convert_tokens_to_ids(WALL_OSS_ACTION_TOKEN)
        # The bin numbers -1 to 511 the step writes, each with the space before it.
        self.pieces = [
            processor.tokenizer.encode(f" {value}", add_special_tokens=False) for value in range(-1, 512)
        ]
        self.prompt: StatePromptTokens | None = None
        self.constants: dict[str, Tensor] = {}

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not is_exporting():
            if self.prompt is None:
                self._fit(transition)
                for extreme in extreme_states(transition):
                    self._check(extreme)
            self._check(transition)
        return self._apply(transition)

    def _bins(self, state: Tensor) -> Tensor:
        """Each state value's bin number plus one, as the step digitizes it against 512 equal bins."""
        edges = -1 + torch.arange(512, dtype=torch.float32, device=state.device) / 256
        return (state.flatten()[:, None] >= edges).sum(-1)

    def _camera_images(self, observation: dict[str, Any]) -> list[Tensor]:
        """The camera images with the step's values, whole numbers from 0 to 255, in its camera order."""
        return [observation[key][0].clamp(0, 1).mul(255).round() for key in self.step.camera_key_mapping]

    def _apply(self, transition: EnvTransition) -> EnvTransition:
        if self.prompt is None:
            raise ValueError("The prompt is made from an eager call before the program is traced.")
        observation = transition[TransitionKey.OBSERVATION]
        input_ids, mask = self.prompt(self._bins(observation[OBS_STATE]))
        images = self._camera_images(observation)
        pixels = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
        inputs = {
            "input_ids": input_ids[None],
            "attention_mask": mask[None].long(),
            "pixel_values": pixels.to(images[0].device),
            "moe_token_types": input_ids[None] == self.action_token_id,
            **self.constants,
        }
        complementary = {**(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}), **inputs}
        return {**transition, TransitionKey.COMPLEMENTARY_DATA: complementary}

    def _saved(self, transition: EnvTransition) -> dict[str, Any]:
        """What the saved tokenizer step gives the model for this frame."""
        return self.step(transition)[TransitionKey.COMPLEMENTARY_DATA]

    def _fit(self, transition: EnvTransition) -> None:
        """Take the prompt around the state, and the inputs that do not change, from the saved step."""
        observation = transition[TransitionKey.OBSERVATION]
        for key in self.step.camera_key_mapping:
            height, width = observation[key].shape[-2:]
            if max(height, width) != IMAGE_SIZE:
                raise ValueError(
                    f"Wall-OSS exports cameras whose long side is {IMAGE_SIZE} pixels: its tokenizer step resizes "
                    f"the {height}x{width} images of {key} with Pillow, which a program cannot do as Pillow does."
                )
        saved = self._saved(transition)
        ids = saved["input_ids"][0][saved["attention_mask"][0].bool()]
        bins = self._bins(observation[OBS_STATE])
        device = bins.device
        self.prompt = StatePromptTokens.around(
            ids.tolist(), bins.tolist(), self.pieces, self.pad_id, device=device, padding_side="left"
        )
        # The image grid, the action mask and the action flag: the same for every frame of this task and camera.
        given = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        self.constants = {
            key: value.to(device)
            for key, value in saved.items()
            if key not in given
            and key not in ("input_ids", "attention_mask", "pixel_values", "moe_token_types")
        }

    def _check(self, transition: EnvTransition) -> None:
        """Refuse a prompt, pixels or fixed inputs other than the saved step's for this frame."""
        program = self._apply(transition)[TransitionKey.COMPLEMENTARY_DATA]
        saved = self._saved(transition)
        length = saved["input_ids"].shape[1]
        same_prompt = int(program["attention_mask"].sum()) == length and all(
            torch.equal(program[key][:, -length:], saved[key].to(program[key].device))
            for key in ("input_ids", "attention_mask", "moe_token_types")
        )
        if not same_prompt:
            state = transition[TransitionKey.OBSERVATION][OBS_STATE].tolist()
            raise ValueError(
                "The program writes another prompt than Wall-OSS's tokenizer step for the normalized state "
                f"{state}: this tokenizer splits the state's numbers differently inside the prompt, so Wall-OSS "
                "cannot be exported this way."
            )
        for key in ("pixel_values", *self.constants):
            if not torch.equal(program[key], saved[key].to(program[key].device)):
                raise ValueError(
                    f"The program's {key} differ from the ones Wall-OSS's tokenizer step computes."
                )

    def transform_features(self, features):
        return features
