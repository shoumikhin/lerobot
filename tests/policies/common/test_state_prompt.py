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

import numpy as np
import pytest
import torch

from lerobot.policies.common.state_prompt import (
    StatePromptStep,
    StatePromptTokens,
    uniform_bins,
)
from lerobot.processor import TransitionKey
from lerobot.utils.constants import OBS_STATE

# Nonzero, as Qwen's padding id is, so an unused piece slot that leaked into a token would show.
PAD = 8
# Like Gemma's tokens of " 0" to " 120": a space, then one token per digit.
PIECES = [[1, *(10 + int(digit) for digit in str(number))] for number in range(121)]
START, END = [2, 3, 4], [5, 6]


def processor_prompt(bins: list[int]) -> list[int]:
    return [*START, *(token for number in bins for token in PIECES[number]), *END]


class StatePrompt(torch.nn.Module):
    def __init__(self, tokens: StatePromptTokens):
        super().__init__()
        self.tokens = tokens

    def forward(self, bins):
        return self.tokens(bins)


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_a_program_writes_the_processors_prompt_for_any_state_padded_on_its_side(padding_side):
    tokens = StatePromptTokens.around(
        processor_prompt([42, 42, 42]), [42, 42, 42], PIECES, PAD, length=20, padding_side=padding_side
    )
    program = torch.export.export(StatePrompt(tokens), (torch.tensor([100, 110, 120]),), strict=False)

    for bins in ([7, 120, 0], [120, 120, 120], [3, 4, 5], [9, 0, 57]):
        ids, mask = program.module()(torch.tensor(bins))

        expected = processor_prompt(bins)
        padding = 20 - len(expected)
        if padding_side == "right":
            assert ids.tolist() == expected + [PAD] * padding, bins
            assert mask.tolist() == [True] * len(expected) + [False] * padding, bins
        else:
            assert ids.tolist() == [PAD] * padding + expected, bins
            assert mask.tolist() == [False] * padding + [True] * len(expected), bins


def test_by_default_the_prompt_is_as_long_as_the_widest_state_makes_it():
    tokens = StatePromptTokens(START, END, PIECES, PAD, state_dim=3)

    ids, mask = tokens(torch.tensor([100, 110, 120]))

    assert ids.tolist() == processor_prompt([100, 110, 120])
    assert bool(mask.all())


def test_a_prompt_with_nothing_around_the_state_still_holds_token_ids():
    ids, _ = StatePromptTokens([], [], PIECES, PAD, state_dim=2)(torch.tensor([7, 120]))

    assert ids.dtype == torch.long
    assert ids.tolist() == [*PIECES[7], *PIECES[120], PAD, PAD]


@pytest.mark.parametrize(
    "ids",
    [
        # The processor wrote 7 and 12 as one token each, not as the pieces say.
        [*START, 99, 98, *END],
        # The task holds the state's numbers too, so the program would not know which ones to write over.
        [*START, *PIECES[7], *PIECES[12], *processor_prompt([7, 12])[len(START) :]],
    ],
)
def test_a_prompt_without_the_states_numbers_exactly_once_is_refused(ids):
    with pytest.raises(ValueError, match="does not hold the state's numbers once"):
        StatePromptTokens.around(ids, [7, 12], PIECES, PAD)


def test_a_length_the_widest_state_does_not_fit_is_refused():
    with pytest.raises(ValueError, match="up to 17 tokens"):
        StatePromptTokens(START, END, PIECES, PAD, state_dim=3, length=16)


@pytest.mark.parametrize("count", [256, 512])
def test_uniform_bins_are_numpys_digitize_against_equal_bins(count):
    extremes = [-1.0, 0.0, 1.0, 1 - 2 / count, float("nan"), float("inf"), -float("inf")]
    state = torch.cat([torch.linspace(-1.5, 1.5, 1001), torch.tensor(extremes)])
    edges = np.linspace(-1, 1, count + 1)[:-1]
    assert uniform_bins(state[None], count).tolist() == np.digitize(state.numpy(), edges).tolist()


@pytest.mark.parametrize("count", [255, 1000])
def test_uniform_bins_refuse_a_count_whose_edges_float32_rounds(count):
    with pytest.raises(ValueError, match=f"power of two bins, not {count}"):
        uniform_bins(torch.zeros(1, 3), count)


class RecordingStep(StatePromptStep):
    def __init__(self):
        self.prompt = None
        self.calls = []

    def _fit(self, transition):
        self.calls.append("fit")
        self.prompt = StatePromptTokens(START, END, PIECES, PAD, state_dim=2)

    def _check(self, transition):
        self.calls.append(f"check {transition[TransitionKey.OBSERVATION][OBS_STATE][0, 0].item():g}")

    def _apply(self, transition):
        self.calls.append("apply")
        return transition


def test_a_state_prompt_step_fits_once_checks_eager_frames_and_only_builds_under_export():
    step = RecordingStep()
    frame = {TransitionKey.OBSERVATION: {OBS_STATE: torch.tensor([[0.5, -0.5]])}}
    step(frame)
    step(frame)
    assert step.calls == ["fit", "check -1e+06", "check 1e+06", "check 0.5", "apply", "check 0.5", "apply"]

    class Program(torch.nn.Module):
        def forward(self, state):
            return step({TransitionKey.OBSERVATION: {OBS_STATE: state}})[TransitionKey.OBSERVATION][OBS_STATE]

    step.calls.clear()
    torch.export.export(Program(), (torch.tensor([[0.5, -0.5]]),), strict=False)
    assert step.calls == ["apply"]
