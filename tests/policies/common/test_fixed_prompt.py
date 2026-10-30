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

from lerobot.policies.common import fixed_prompt
from lerobot.policies.common.fixed_prompt import FixedPromptStep
from lerobot.processor import ProcessorStep, TransitionKey

IMAGE = "observation.image"


class Tokenizer(ProcessorStep):
    """Like a VLM processor: Python token counting, with one more token, or one more entry, when the image is
    bright if asked. It counts its calls."""

    def __init__(self, bright_token=False, bright_entry=False):
        self.bright_token = bright_token
        self.bright_entry = bright_entry
        self.calls = 0

    def __call__(self, transition):
        self.calls += 1
        image = transition[TransitionKey.OBSERVATION][IMAGE]
        bright = image.mean().item() > 0.5
        count = 3 + int(self.bright_token and bright)
        comp = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        comp.update(input_ids=torch.arange(count)[None], labels=None, pixel_values=image.flatten(1) * 2)
        if self.bright_entry and bright:
            comp["bright"] = True
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


@pytest.mark.parametrize("tokenizer", [Tokenizer(bright_token=True), Tokenizer(bright_entry=True)])
def test_a_prompt_that_changes_with_the_frame_is_refused(tokenizer):
    step = FixedPromptStep(tokenizer, pixels)
    step(frame(0.2))

    with pytest.raises(ValueError, match="prompt changed between two frames"):
        step(frame(0.8))


def test_each_frame_gets_its_own_tensors_and_only_eager_frames_run_the_step(monkeypatch):
    tokenizer = Tokenizer()
    step = FixedPromptStep(tokenizer, pixels)
    eager = step(frame(0.2))[TransitionKey.COMPLEMENTARY_DATA]
    monkeypatch.setattr(fixed_prompt, "is_non_strict_exporting", lambda: True)
    exported = step(frame(0.4))[TransitionKey.COMPLEMENTARY_DATA]

    assert tokenizer.calls == 1
    torch.testing.assert_close(eager["pixel_values"], torch.full((1, 12), 0.4))
    torch.testing.assert_close(exported["pixel_values"], torch.full((1, 12), 0.8))
    assert eager["input_ids"].tolist() == exported["input_ids"].tolist() == [[0, 1, 2]]


class ChatRenderer(Tokenizer):
    """Also drops the chat messages it renders, as EO-1's step does, unless asked to keep them on a bright image."""

    def __init__(self, keep_when_bright=False):
        super().__init__()
        self.keep_when_bright = keep_when_bright

    def __call__(self, transition):
        transition = super().__call__(transition)
        comp = dict(transition[TransitionKey.COMPLEMENTARY_DATA])
        if not (self.keep_when_bright and transition[TransitionKey.OBSERVATION][IMAGE].mean().item() > 0.5):
            del comp["messages"]
        return {**transition, TransitionKey.COMPLEMENTARY_DATA: comp}


def chat(value):
    transition = frame(value)
    transition[TransitionKey.COMPLEMENTARY_DATA]["messages"] = [{"role": "user", "content": "pick"}]
    return transition


def test_an_entry_the_step_removes_is_removed_under_export_too(monkeypatch):
    step = FixedPromptStep(ChatRenderer(), pixels)
    eager = step(chat(0.2))[TransitionKey.COMPLEMENTARY_DATA]
    monkeypatch.setattr(fixed_prompt, "is_non_strict_exporting", lambda: True)
    exported = step(chat(0.4))[TransitionKey.COMPLEMENTARY_DATA]

    assert "messages" not in eager
    assert "messages" not in exported


def test_a_step_that_removes_other_entries_on_another_frame_is_refused():
    step = FixedPromptStep(ChatRenderer(keep_when_bright=True), pixels)
    step(chat(0.2))

    with pytest.raises(ValueError, match="prompt changed between two frames"):
        step(chat(0.8))


def test_a_frame_function_that_disagrees_with_the_step_is_refused():
    step = FixedPromptStep(Tokenizer(), lambda t: {"pixel_values": pixels(t)["pixel_values"] + 1})

    with pytest.raises(ValueError, match="per-frame 'pixel_values' differs"):
        step(frame(0.2))


class Rewriter(Tokenizer):
    """Also rewrites the task it is given, as MolmoAct2's step does."""

    def __call__(self, transition):
        transition = super().__call__(transition)
        comp = transition[TransitionKey.COMPLEMENTARY_DATA]
        return {**transition, TransitionKey.COMPLEMENTARY_DATA: {**comp, "task": ["Pick"]}}


def test_a_key_the_step_rewrites_is_part_of_the_prompt_in_eager_frames_and_under_export(monkeypatch):
    step = FixedPromptStep(Rewriter(), pixels)
    eager = step(frame(0.2))[TransitionKey.COMPLEMENTARY_DATA]
    monkeypatch.setattr(fixed_prompt, "is_non_strict_exporting", lambda: True)
    exported = step(frame(0.4))[TransitionKey.COMPLEMENTARY_DATA]

    assert eager["task"] == exported["task"] == ["Pick"]


class Normalizer(Tokenizer):
    """Also changes the observation, which the program would not, since it does not run the step."""

    def __call__(self, transition):
        transition = super().__call__(transition)
        observation = transition[TransitionKey.OBSERVATION]
        return {**transition, TransitionKey.OBSERVATION: {IMAGE: observation[IMAGE] * 0.5}}


def test_a_step_that_changes_the_observation_is_refused():
    with pytest.raises(ValueError, match=r"changes the observation's \['observation.image'\]"):
        FixedPromptStep(Normalizer(), pixels)(frame(0.2))


class Annotator(Tokenizer):
    """Also adds an observation entry, which the program would not."""

    def __call__(self, transition):
        transition = super().__call__(transition)
        observation = transition[TransitionKey.OBSERVATION]
        return {**transition, TransitionKey.OBSERVATION: {**observation, "observation.extra": torch.ones(1)}}


def test_a_step_that_adds_to_the_observation_is_refused():
    with pytest.raises(ValueError, match=r"changes the observation's \['observation.extra'\]"):
        FixedPromptStep(Annotator(), pixels)(frame(0.2))
