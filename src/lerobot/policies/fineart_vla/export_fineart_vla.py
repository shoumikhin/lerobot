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

"""Export FineART-VLA as one program: its prompt made once, the state's bin numbers written into it on each frame.

FineART-VLA's text tokenizer step writes the state into the prompt as pi0.5 does, one bin number per state value,
and tokenizes the prompt with the PaliGemma tokenizer, which a program cannot do. The recipe replaces that step
with [`_StatePromptStep`]: the tokens before and after the state come once from the saved step, and each bin
number's tokens come from a table (see [`StatePromptTokens`]). The step pads its prompt to `max_length`, as the
program does, so the program's prompt is the processor's token for token, and the default recipe still checks
that the program computes rollout's actions bit for bit. Every eager frame is checked against the saved step, and
so are states beyond the first and last bins.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from lerobot.export.recipe import BFLOAT16_STEPS_TOLERANCE, ActionChunkRecipe
from lerobot.processor import EnvTransition, PolicyProcessorPipeline, TransitionKey
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from ..common.state_prompt import StatePromptStep, StatePromptTokens, uniform_bins
from .processor_fineart_vla import FineARTVLATextTokenizerStep


class FineARTVLAExportRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] with FineART-VLA's prompt built in the program."""

    # FineART-VLA runs pi0.5's ten bfloat16 denoising steps.
    bfloat16_tolerance = BFLOAT16_STEPS_TOLERANCE

    def program_preprocessor(self) -> PolicyProcessorPipeline:
        """The saved preprocessor with its text tokenizer step replaced by [`_StatePromptStep`]."""
        if not any(isinstance(step, FineARTVLATextTokenizerStep) for step in self.preprocessor.steps):
            raise ValueError(
                "This FineART-VLA checkpoint has no text recipe, so its preprocessor is pi0.5's, which the "
                "FineART-VLA recipe does not export."
            )
        steps = [
            _StatePromptStep(step) if isinstance(step, FineARTVLATextTokenizerStep) else step
            for step in self.preprocessor.steps
        ]
        return replace(self.preprocessor, steps=steps)


class _StatePromptStep(StatePromptStep):
    """FineART-VLA's text tokenizer step for one task, as tensor operations.

    The prompt's tokens before and after the state, and what the step adds to the complementary data, are the
    saved step's on the first eager frame. Every eager frame is checked against the saved step, and so are states
    beyond the first and last bins on the first one.
    """

    def __init__(self, step: FineARTVLATextTokenizerStep) -> None:
        self.step = step
        tokenizer = step._ensure_tokenizer()
        self.pad_id = tokenizer.pad_token_id
        # The bin numbers -1 to 255 of `discretize_state_str`, each with the space before it.
        self.pieces = [tokenizer.encode(f" {value}", add_special_tokens=False) for value in range(-1, 256)]
        self.prompt: StatePromptTokens | None = None
        self.complementary: dict[str, Any] = {}

    def _apply(self, transition: EnvTransition) -> EnvTransition:
        if self.prompt is None:
            raise ValueError("The prompt is made from an eager call before the program is traced.")
        observation = dict(transition[TransitionKey.OBSERVATION] or {})
        ids, mask = self.prompt(uniform_bins(observation[OBS_STATE], 256))
        observation[OBS_LANGUAGE_TOKENS] = ids[None]
        observation[OBS_LANGUAGE_ATTENTION_MASK] = mask[None]
        complementary = {**(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}), **self.complementary}
        return {
            **transition,
            TransitionKey.OBSERVATION: observation,
            TransitionKey.COMPLEMENTARY_DATA: complementary,
        }

    def _fit(self, transition: EnvTransition) -> None:
        """Take the prompt around the state, and what the step adds to the complementary data, from this frame."""
        saved = self.step(transition)
        tokens = saved[TransitionKey.OBSERVATION] or {}
        ids = tokens[OBS_LANGUAGE_TOKENS][0]
        mask = tokens[OBS_LANGUAGE_ATTENTION_MASK][0]
        bins = uniform_bins((transition[TransitionKey.OBSERVATION] or {})[OBS_STATE], 256)
        self.prompt = StatePromptTokens.around(
            ids[mask].tolist(), bins.tolist(), self.pieces, self.pad_id, len(ids), bins.device
        )
        given = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        added = saved[TransitionKey.COMPLEMENTARY_DATA] or {}
        self.complementary = {key: value for key, value in added.items() if key not in given}

    def _check(self, transition: EnvTransition) -> None:
        """Refuse a prompt other than the saved step's for this frame."""
        program = self._apply(transition)[TransitionKey.OBSERVATION] or {}
        saved = self.step(transition)[TransitionKey.OBSERVATION] or {}
        if not all(
            torch.equal(program[key], saved[key].to(program[key].device))
            for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK)
        ):
            state = (transition[TransitionKey.OBSERVATION] or {})[OBS_STATE].tolist()
            raise ValueError(
                "The program writes another prompt than FineART-VLA's text tokenizer step for the normalized "
                f"state {state}: this tokenizer splits the state's numbers differently inside the prompt, so "
                "FineART-VLA cannot be exported this way."
            )
