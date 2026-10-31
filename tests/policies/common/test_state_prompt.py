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
    extreme_states,
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


def test_a_prompt_without_the_states_numbers_is_refused():
    # The processor wrote 7 and 12 as one token each, not as the pieces say.
    ids = [*START, 99, 98, *END]

    with pytest.raises(ValueError, match="does not hold the state's numbers once"):
        StatePromptTokens.around(ids, [7, 12], PIECES, PAD)


def test_a_length_the_widest_state_does_not_fit_is_refused():
    with pytest.raises(ValueError, match="up to 17 tokens"):
        StatePromptTokens(START, END, PIECES, PAD, state_dim=3, length=16)


def test_extreme_states_change_only_the_state():
    state, image = torch.tensor([[0.1, -0.2]]), torch.rand(3, 4, 4)
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: state, "image": image},
        TransitionKey.COMPLEMENTARY_DATA: {"task": "pick up the cube"},
    }

    low, high = extreme_states(transition)

    assert low[TransitionKey.OBSERVATION][OBS_STATE].tolist() == [[-1e6, -1e6]]
    assert high[TransitionKey.OBSERVATION][OBS_STATE].tolist() == [[1e6, 1e6]]
    assert low[TransitionKey.OBSERVATION]["image"] is image
    assert high[TransitionKey.COMPLEMENTARY_DATA] == {"task": "pick up the cube"}
    assert transition[TransitionKey.OBSERVATION][OBS_STATE] is state
    assert torch.equal(state, torch.tensor([[0.1, -0.2]]))


@pytest.mark.parametrize("count", [256, 512])
def test_uniform_bins_are_numpys_digitize_against_equal_bins(count):
    state = torch.cat([torch.linspace(-1.5, 1.5, 1001), torch.tensor([-1.0, 0.0, 1.0, 1 - 2 / count])])
    edges = np.linspace(-1, 1, count + 1)[:-1]
    assert uniform_bins(state[None], count).tolist() == np.digitize(state.numpy(), edges).tolist()


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
