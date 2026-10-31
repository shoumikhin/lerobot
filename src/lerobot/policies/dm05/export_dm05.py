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

"""Export DM05 as one program: its Gemma3 prompt made once, the state's bin numbers written into it on each frame.

DM05's tokenizer step renders the chat template with the camera images and the state's bin numbers as text, and
tokenizes it, which a program cannot do. The recipe replaces it, and the step that bins the state, with
[`_StatePromptStep`]: the tokens before and after the state come once from the saved step, and each bin number's
tokens come from a table (see [`StatePromptTokens`]). That holds while the tokenizer splits a number inside the
prompt as it splits it alone, which is checked on every eager frame and on states in the first and last bins. A
bin number has one to three digits, so the program right-pads the prompt to the longest state, as DM05's
processor pads a batch, and leaves the padding out of attention. The actions then differ from rollout's by
bfloat16 rounding only, and the program is held to the export's tolerance. The pixels come from the saved
processor's own image processor, which runs in torch.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from lerobot.export.recipe import ActionChunkRecipe
from lerobot.processor import EnvTransition, PolicyProcessorPipeline, ProcessorStep, TransitionKey
from lerobot.utils.constants import OBS_STATE

from ..common.qwen_vision import is_exporting
from ..common.state_prompt import StatePromptTokens, extreme_states
from .core.tokenization import action_to_bin_tokens
from .core.utils import DM05_STATE_BINS
from .processor_dm05 import MODEL_INPUT_PREFIX, DM05StateBinsProcessorStep, DM05TokenizerProcessorStep


class DM05ExportRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] with DM05's prompt built in the program and held to the export's tolerance."""

    exact = False

    def program_preprocessor(self) -> PolicyProcessorPipeline:
        """The saved preprocessor with its state bins and tokenizer steps replaced by [`_StatePromptStep`]."""
        (bins,) = [step for step in self.preprocessor.steps if isinstance(step, DM05StateBinsProcessorStep)]
        (tokenizer,) = [
            step for step in self.preprocessor.steps if isinstance(step, DM05TokenizerProcessorStep)
        ]
        if not tokenizer.add_state:
            raise ValueError(
                "The DM05 recipe writes the state into the prompt, and this checkpoint leaves it out "
                "(add_state=false)."
            )
        prompt = _StatePromptStep(bins, tokenizer)
        steps = [
            prompt if step is tokenizer else step for step in self.preprocessor.steps if step is not bins
        ]
        return replace(self.preprocessor, steps=steps)


class _StatePromptStep(ProcessorStep):
    """DM05's state bins and tokenizer steps for one task and camera size, as tensor operations.

    The prompt's tokens before and after the state are the saved tokenizer step's on the first eager frame. Every
    eager frame is checked against the saved steps, and so are states in the first and last bins on the first
    one: the program must place the same tokens and token types, and compute the same pixels.
    """

    def __init__(self, bins: DM05StateBinsProcessorStep, tokenizer: DM05TokenizerProcessorStep) -> None:
        self.bins = bins
        self.tokenizer = tokenizer
        processor = tokenizer.tokenization.processor
        self.image_processor = processor.image_processor
        self.pad_id = processor.tokenizer.pad_token_id
        # Each bin number with the space before it, as the state's text writes it.
        self.pieces = [
            processor.tokenizer.encode(f" {value}", add_special_tokens=False)
            for value in range(DM05_STATE_BINS)
        ]
        self.prompt: StatePromptTokens | None = None
        self.token_type_ids: Tensor | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not is_exporting():
            if self.prompt is None:
                self._fit(transition)
                for extreme in extreme_states(transition):
                    self._check(extreme)
            self._check(transition)
        return self._apply(transition)

    def _apply(self, transition: EnvTransition) -> EnvTransition:
        if self.prompt is None or self.token_type_ids is None:
            raise ValueError("The prompt is made from an eager call before the program is traced.")
        observation = dict(transition[TransitionKey.OBSERVATION])
        bins = action_to_bin_tokens(observation[OBS_STATE], DM05_STATE_BINS)
        input_ids, mask = self.prompt(bins[0])
        images = self.tokenizer.camera_images(observation, 1)
        pixels = self.image_processor([[image[0] for image in images]], do_rescale=False, return_tensors="pt")
        inputs = {
            "input_ids": input_ids[None],
            "attention_mask": mask[None].long(),
            "token_type_ids": self.token_type_ids,
            "pixel_values": pixels["pixel_values"].to(images[0].device),
        }
        observation.update({f"{MODEL_INPUT_PREFIX}{key}": value for key, value in inputs.items()})
        return {**transition, TransitionKey.OBSERVATION: observation}

    def _saved(self, transition: EnvTransition) -> dict[str, Tensor]:
        """What the saved state bins and tokenizer steps give the model for this frame."""
        observation = self.tokenizer(self.bins(transition))[TransitionKey.OBSERVATION]
        return {key.removeprefix(MODEL_INPUT_PREFIX): value for key, value in observation.items()}

    def _fit(self, transition: EnvTransition) -> None:
        """Take the prompt around the state, and its token types, from the saved steps on this frame."""
        saved = self._saved(transition)
        ids = saved["input_ids"][0]
        bins = action_to_bin_tokens(transition[TransitionKey.OBSERVATION][OBS_STATE], DM05_STATE_BINS)[0]
        self.prompt = StatePromptTokens.around(
            ids.tolist(), bins.tolist(), self.pieces, self.pad_id, device=ids.device
        )
        # Only the image tokens have a type, and they all come before the state.
        start = len(self.prompt.start)
        types = saved["token_type_ids"][:, :start]
        self.token_type_ids = torch.nn.functional.pad(types, (0, len(self.prompt.positions)))

    def _check(self, transition: EnvTransition) -> None:
        """Refuse a prompt or pixels other than the saved steps' for this frame."""
        program = self._apply(transition)[TransitionKey.OBSERVATION]
        program = {key.removeprefix(MODEL_INPUT_PREFIX): value for key, value in program.items()}
        saved = self._saved(transition)
        length = saved["input_ids"].shape[1]
        valid = program["attention_mask"][0].bool()
        same_prompt = int(valid.sum()) == length and all(
            torch.equal(program[key][:, :length], saved[key])
            for key in ("input_ids", "attention_mask", "token_type_ids")
        )
        if not same_prompt:
            state = transition[TransitionKey.OBSERVATION][OBS_STATE].tolist()
            raise ValueError(
                f"The program writes another prompt than DM05's tokenizer step for the normalized state {state}: "
                "this tokenizer splits the state's numbers differently inside the prompt, so DM05 cannot be "
                "exported this way."
            )
        if not torch.equal(program["pixel_values"], saved["pixel_values"]):
            raise ValueError(
                "The program's pixel values differ from the ones DM05's tokenizer step computes."
            )

    def transform_features(self, features):
        return features
