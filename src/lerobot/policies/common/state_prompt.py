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

"""The token ids of a prompt that writes the state's bin numbers, built with tensor operations.

Some vision-language policies write the state into their prompt as text, one bin number per state value, so the
prompt's tokens, and its length, change on every frame. A transformers tokenizer cannot run in a program, but for
one task the prompt is the same before and after the state, and each bin number has the same tokens wherever it
stands, for a tokenizer that splits numbers off the text around them as Gemma's does. `StatePromptTokens` keeps
those pieces and places them as pi0.5's `StatePrompt` places its rows: each valid token takes the next position,
and the prompt is right-padded to a fixed length, with a mask that leaves the padding out. A recipe still compares
what it builds with the policy's processor, on its eager frames.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor

from lerobot.processor import EnvTransition, ProcessorStep, TransitionKey
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .export_utils import is_non_strict_exporting

# States far outside the training data, so a prompt check covers the first and last bins.
EXTREME_STATE = 1e6


def compact(tokens: Tensor, valid: Tensor, positions: Tensor, pad_id: int) -> tuple[Tensor, Tensor]:
    """The valid `tokens` in order, then `pad_id`, at `positions`, and the mask of the valid positions."""
    # Each valid token owns one position; the others own none.
    owner = (valid.cumsum(0)[:, None] - 1 == positions) & valid[:, None]
    ids = (tokens[:, None] * owner).sum(0)
    mask = positions < valid.sum()
    return torch.where(mask, ids, pad_id), mask


class StatePromptTokens:
    """A prompt's token ids for any state: its start, the state's bin numbers from a table, its end, padding.

    Args:
        start (`list[int]`):
            The prompt's token ids before the state.
        end (`list[int]`):
            The prompt's token ids after the state.
        pieces (`list[list[int]]`):
            The token ids of each bin number, with the space before it, by bin index.
        pad_id (`int`):
            The padding token.
        state_dim (`int`):
            How many bin numbers the state writes.
        length (`int`, *optional*):
            The prompt's fixed length. `None` is the longest a state can make.
        device (`torch.device`, *optional*):
            Where the program runs.
    """

    def __init__(
        self,
        start: list[int],
        end: list[int],
        pieces: list[list[int]],
        pad_id: int,
        state_dim: int,
        length: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        width = max(map(len, pieces))
        longest = len(start) + state_dim * width + len(end)
        length = longest if length is None else length
        if length < longest:
            raise ValueError(
                f"The prompt takes up to {longest} tokens with the state's widest numbers, more than its "
                f"{length}."
            )
        self.start = torch.tensor(start, dtype=torch.long, device=device)
        self.end = torch.tensor(end, dtype=torch.long, device=device)
        padded = [piece + [pad_id] * (width - len(piece)) for piece in pieces]
        self.pieces = torch.tensor(padded, device=device)
        self.lengths = torch.tensor(list(map(len, pieces)), device=device)
        self.piece_positions = torch.arange(width, device=device)
        self.positions = torch.arange(length - len(start), device=device)
        self.pad_id = pad_id

    @classmethod
    def around(
        cls,
        ids: list[int],
        bins: list[int],
        pieces: list[list[int]],
        pad_id: int,
        length: int | None = None,
        device: torch.device | None = None,
    ) -> StatePromptTokens:
        """The prompt `ids` a processor made for a state with these bin indices, split around the state."""
        state = [token for index in bins for token in pieces[index]]
        starts = [i for i in range(len(ids) - len(state) + 1) if ids[i : i + len(state)] == state]
        if len(starts) != 1:
            raise ValueError(
                "The prompt does not hold the state's numbers once, each as its tokenizer writes it alone, so a "
                "program cannot write them in."
            )
        start, end = starts[0], starts[0] + len(state)
        return cls(ids[:start], ids[end:], pieces, pad_id, len(bins), length, device)

    def __call__(self, bins: Tensor) -> tuple[Tensor, Tensor]:
        """The token ids and the mask of the prompt's tokens, both `[length]`, for the state's bin indices."""
        tokens = torch.cat((self.pieces[bins].flatten(), self.end))
        valid = torch.cat(
            (
                (self.piece_positions < self.lengths[bins, None]).flatten(),
                torch.ones_like(self.end, dtype=torch.bool),
            )
        )
        tail, tail_mask = compact(tokens, valid, self.positions, self.pad_id)
        mask = torch.cat((torch.ones_like(self.start, dtype=torch.bool), tail_mask))
        return torch.cat((self.start, tail)), mask


def extreme_states(transition: EnvTransition) -> Iterator[EnvTransition]:
    """The transition with every state value far below the first bin, then far above the last one.

    A recipe checks its prompt on these too, so that the check covers the first and last bins.
    """
    for value in (-EXTREME_STATE, EXTREME_STATE):
        observation = dict(transition[TransitionKey.OBSERVATION] or {})
        observation[OBS_STATE] = torch.full_like(observation[OBS_STATE], value)
        yield {**transition, TransitionKey.OBSERVATION: observation}


def uniform_bins(state: Tensor, count: int) -> Tensor:
    """Each state value's bin number plus one, as `np.digitize` gives it against `count` equal bins from -1 to 1.

    `count` is a power of two, whose bin edges float32 holds exactly, as the float64 edges of NumPy are.
    """
    if count < 2 or count & (count - 1):
        raise ValueError(f"uniform_bins takes a power of two bins, not {count}.")
    edges = -1 + torch.arange(count, dtype=torch.float32, device=state.device) / (count // 2)
    values = state.flatten()
    # np.digitize puts NaN after the last edge, where no comparison would.
    return torch.where(values.isnan(), count, (values[:, None] >= edges).sum(-1))


class StatePromptStep(ProcessorStep):
    """A policy's tokenizer step for one task, as tensor operations that write the state into the prompt.

    The first eager call takes the prompt around the state from the saved step (`_fit`). Every eager call then
    checks that the program builds what the saved step does for its frame (`_check`), the first one also for
    states beyond the first and last bins. While `torch.export` traces, the step only builds its inputs (`_apply`).
    """

    prompt: StatePromptTokens | None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not is_non_strict_exporting():
            if self.prompt is None:
                self._fit(transition)
                for extreme in extreme_states(transition):
                    self._check(extreme)
            self._check(transition)
        return self._apply(transition)

    def _prompt_tokens(self, bins: Tensor) -> tuple[Tensor, Tensor]:
        """The prompt's token ids and mask for the state's bin indices, once an eager call took the prompt."""
        if self.prompt is None:
            raise ValueError("The prompt is made from an eager call before the program is traced.")
        return self.prompt(bins)

    @abstractmethod
    def _fit(self, transition: EnvTransition) -> None:
        """Take the prompt around the state, and whatever else does not change with the frame, from this frame."""

    @abstractmethod
    def _check(self, transition: EnvTransition) -> None:
        """Refuse a prompt, or other inputs, other than the saved step's for this frame."""

    @abstractmethod
    def _apply(self, transition: EnvTransition) -> EnvTransition:
        """The saved step's output for this frame, built with tensor operations."""

    def transform_features(self, features):
        return features


class PI05StatePromptStep(StatePromptStep):
    """A tokenizer step for one task that writes the state as pi0.5's prompt does, as tensor operations.

    The prompt writes each state value's bin among 256 equal bins of [-1, 1] as a number, and the step gives its
    token ids and mask as the observation's language tokens, padded to the saved steps' length. The prompt around
    the state, and what the saved steps add to the complementary data, come from the first eager frame. A subclass
    gives the saved steps (`_saved`) and names them in its error (`policy`, `saved_steps`).

    Args:
        tokenizer:
            The Transformers tokenizer of the saved steps.
    """

    policy: str
    saved_steps: str

    def __init__(self, tokenizer: Any) -> None:
        self.pad_id = tokenizer.pad_token_id
        # The bin numbers -1 to 255 of the prompt's state, each with the space before it.
        self.pieces = [tokenizer.encode(f" {value}", add_special_tokens=False) for value in range(-1, 256)]
        self.prompt: StatePromptTokens | None = None
        self.complementary: dict[str, Any] = {}

    @abstractmethod
    def _saved(self, transition: EnvTransition) -> EnvTransition:
        """The saved steps' output for this frame."""

    def _apply(self, transition: EnvTransition) -> EnvTransition:
        observation = dict(transition[TransitionKey.OBSERVATION] or {})
        ids, mask = self._prompt_tokens(uniform_bins(observation[OBS_STATE], 256))
        observation[OBS_LANGUAGE_TOKENS] = ids[None]
        observation[OBS_LANGUAGE_ATTENTION_MASK] = mask[None]
        complementary = {**(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}), **self.complementary}
        return {
            **transition,
            TransitionKey.OBSERVATION: observation,
            TransitionKey.COMPLEMENTARY_DATA: complementary,
        }

    def _fit(self, transition: EnvTransition) -> None:
        """Take the prompt around the state, and what the saved steps add to the complementary data."""
        saved = self._saved(transition)
        tokens = saved[TransitionKey.OBSERVATION] or {}
        ids, mask = tokens[OBS_LANGUAGE_TOKENS][0], tokens[OBS_LANGUAGE_ATTENTION_MASK][0]
        bins = uniform_bins((transition[TransitionKey.OBSERVATION] or {})[OBS_STATE], 256)
        self.prompt = StatePromptTokens.around(
            ids[mask].tolist(), bins.tolist(), self.pieces, self.pad_id, len(ids), bins.device
        )
        given = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        added = saved.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        self.complementary = {key: value for key, value in added.items() if key not in given}

    def _check(self, transition: EnvTransition) -> None:
        """Refuse a prompt other than the saved steps' for this frame."""
        program = self._apply(transition)[TransitionKey.OBSERVATION] or {}
        saved = self._saved(transition)[TransitionKey.OBSERVATION] or {}
        if not all(
            torch.equal(program[key], saved[key].to(program[key].device))
            for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK)
        ):
            state = (transition[TransitionKey.OBSERVATION] or {})[OBS_STATE].tolist()
            raise ValueError(
                f"The program writes another prompt than {self.policy}'s {self.saved_steps} for the normalized "
                f"state {state}: this tokenizer splits the state's numbers differently inside the prompt, so "
                f"{self.policy} cannot be exported this way."
            )
