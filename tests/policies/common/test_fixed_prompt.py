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

import pytest
import torch

from lerobot.policies.common.fixed_prompt import FixedPromptStep
from lerobot.processor import ProcessorStep, TransitionKey

IMAGE = "observation.image"


class Tokenizer(ProcessorStep):
    """Like a VLM processor: Python token counting, with one more token when the image is bright if asked."""

    def __init__(self, bright_token=False):
        self.bright_token = bright_token

    def __call__(self, transition):
        image = transition[TransitionKey.OBSERVATION][IMAGE]
        count = 3 + int(self.bright_token and image.mean().item() > 0.5)
        comp = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        comp.update(input_ids=torch.arange(count)[None], labels=None, pixel_values=image.flatten(1) * 2)
        return {**transition, TransitionKey.COMPLEMENTARY_DATA: comp}

    def transform_features(self, features):
        return features


def pixels(transition):
    return {"pixel_values": transition[TransitionKey.OBSERVATION][IMAGE].flatten(1) * 2}


def frame(value):
    return {
        TransitionKey.OBSERVATION: {IMAGE: torch.full((1, 3, 2, 2), value)},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick"]},
    }


class Program(torch.nn.Module):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def forward(self, image):
        comp = self.step({TransitionKey.OBSERVATION: {IMAGE: image}})[TransitionKey.COMPLEMENTARY_DATA]
        return comp["input_ids"].float().sum() + comp["pixel_values"]


def test_the_program_keeps_the_prompt_and_computes_the_frame():
    step = FixedPromptStep(Tokenizer(), pixels)
    eager = step(frame(0.2))[TransitionKey.COMPLEMENTARY_DATA]
    assert eager["labels"] is None and eager["task"] == ["pick"]

    program = torch.export.export(Program(step), (torch.full((1, 3, 2, 2), 0.2),), strict=False)

    image = torch.rand(1, 3, 2, 2)
    torch.testing.assert_close(program.module()(image), 3.0 + image.flatten(1) * 2, rtol=0, atol=0)


def test_a_prompt_that_changes_with_the_frame_is_refused():
    step = FixedPromptStep(Tokenizer(bright_token=True), pixels)
    step(frame(0.2))

    with pytest.raises(ValueError, match="prompt changed between two frames"):
        step(frame(0.8))


def test_a_frame_function_that_disagrees_with_the_step_is_refused():
    step = FixedPromptStep(Tokenizer(), lambda t: {"pixel_values": pixels(t)["pixel_values"] + 1})

    with pytest.raises(ValueError, match="per-frame 'pixel_values' differs"):
        step(frame(0.2))


def test_tracing_before_an_eager_call_is_refused():
    step = FixedPromptStep(Tokenizer(), pixels)

    with pytest.raises(ValueError, match="eager call before the program is traced"):
        torch.export.export(Program(step), (torch.rand(1, 3, 2, 2),), strict=False)
