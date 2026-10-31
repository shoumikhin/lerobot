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

from lerobot.export.recipe import BFLOAT16_STEPS_TOLERANCE, ActionChunkRecipe
from lerobot.processor import EnvTransition, PolicyProcessorPipeline

from ..common.state_prompt import PI05StatePromptStep
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


class _StatePromptStep(PI05StatePromptStep):
    """FineART-VLA's text tokenizer step for one task, as tensor operations."""

    policy = "FineART-VLA"
    saved_steps = "text tokenizer step"

    def __init__(self, step: FineARTVLATextTokenizerStep) -> None:
        super().__init__(step._ensure_tokenizer())
        self.step = step

    def _saved(self, transition: EnvTransition) -> EnvTransition:
        return self.step(transition)
