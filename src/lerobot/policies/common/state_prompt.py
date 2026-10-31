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
and the prompt is padded to a fixed length, with a mask that leaves the padding out. The padding goes on the right,
or on the left for a model that reads the end of its prompt. A recipe still compares what it builds with the
policy's processor, on its eager frames.
"""

from __future__ import annotations

import torch
from torch import Tensor


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
        padding_side (`str`, *optional*, defaults to `"right"`):
            Where the padding goes, `"right"` or `"left"`, as a tokenizer's `padding_side`.
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
        padding_side: str = "right",
    ) -> None:
        if padding_side not in ("right", "left"):
            raise ValueError(f"padding_side must be 'right' or 'left', not {padding_side!r}.")
        width = max(map(len, pieces))
        longest = len(start) + state_dim * width + len(end)
        length = longest if length is None else length
        if length < longest:
            raise ValueError(
                f"The prompt takes up to {longest} tokens with the state's widest numbers, more than its "
                f"{length}."
            )
        self.start = torch.tensor(start, device=device)
        self.end = torch.tensor(end, device=device)
        padded = [piece + [pad_id] * (width - len(piece)) for piece in pieces]
        self.pieces = torch.tensor(padded, device=device)
        self.lengths = torch.tensor(list(map(len, pieces)), device=device)
        self.piece_positions = torch.arange(width, device=device)
        self.positions = torch.arange(length - len(start), device=device)
        self.pad_id = pad_id
        self.padding_side = padding_side
        self.length = length

    @classmethod
    def around(
        cls,
        ids: list[int],
        bins: list[int],
        pieces: list[list[int]],
        pad_id: int,
        length: int | None = None,
        device: torch.device | None = None,
        padding_side: str = "right",
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
        return cls(ids[:start], ids[end:], pieces, pad_id, len(bins), length, device, padding_side)

    def __call__(self, bins: Tensor) -> tuple[Tensor, Tensor]:
        """The token ids and the mask of the prompt's tokens, both `[length]`, for the state's bin indices."""
        tokens = torch.cat((self.pieces[bins].flatten(), self.end))
        valid = torch.cat(
            (
                (self.piece_positions < self.lengths[bins, None]).flatten(),
                torch.ones_like(self.end, dtype=torch.bool),
            )
        )
        # Each valid token owns one position after the start; unused piece slots own none.
        compact = (valid.cumsum(0)[:, None] - 1 == self.positions) & valid[:, None]
        tail_mask = self.positions < valid.sum()
        tail = torch.where(tail_mask, (tokens[:, None] * compact).sum(0), self.pad_id)
        ids = torch.cat((self.start, tail))
        mask = torch.cat((torch.ones_like(self.start, dtype=torch.bool), tail_mask))
        if self.padding_side == "left":
            # Rotate the padding from the end to the front.
            order = (torch.arange(self.length, device=ids.device) - (~mask).sum()) % self.length
            ids, mask = ids[order], mask[order]
        return ids, mask
