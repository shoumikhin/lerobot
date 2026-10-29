#!/usr/bin/env python

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

"""Shared pieces for running transformers Qwen-VL vision towers on a fixed image grid.

The stock towers turn the image grid (`image_grid_thw`) into Python ints on every call, which
`torch.export` cannot trace. When every batch uses the same camera grid, what a tower derives from the
grid can be computed once in eager mode and reused. `FixedImageGrid` holds that cache and decides when it
can be reused; the tower-specific subclasses live in `qwen2_5_vision.py` and `qwen3_vision.py`.

The vision blocks are still called as modules, so FSDP hooks and gradient checkpointing work as in the
stock forward. Their cumulative sequence lengths are rebuilt on every call with `torch.tensor` from cached
Python ints: under `torch.export` such a tensor is a constant, so the stock attention can still read it
back with `.tolist()`.

The grid checks use transformers' `torch_compilable_check`, so `TRANSFORMERS_DISABLE_TORCH_CHECK=1` turns
them off like every other check of that kind. In an exported program a failed check keeps the assert but
loses the message.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.utils import torch_compilable_check
else:
    torch_compilable_check = None


@dataclass
class FixedImageGrid:
    """What a vision tower derives from one image grid, built once in eager mode.

    Subclasses add the tower's metadata and implement `_from_grid`. Only weight-independent tensors are
    cached, built outside inference mode, so the cache works for inference and training alike.
    """

    grid_thw: list[list[int]]
    device: torch.device
    dtype: torch.dtype

    @classmethod
    def _from_grid(cls, visual: nn.Module, grid_thw: list[list[int]]) -> "FixedImageGrid":
        raise NotImplementedError

    @classmethod
    def build(cls, visual: nn.Module, grid_thw: Tensor) -> "FixedImageGrid":
        """Compute the metadata of `grid_thw` for `visual`; this reads the grid on the host."""
        grid = grid_thw.tolist()
        # Tensors made under inference mode cannot be saved for backward later, so a cache built during
        # prediction must still be made of normal tensors.
        with torch.inference_mode(False), torch.no_grad():
            return cls._from_grid(visual, grid)

    @classmethod
    def reuse_or_build(
        cls, cache: "FixedImageGrid | None", visual: nn.Module, grid_thw: Tensor
    ) -> "FixedImageGrid":
        """Return `cache` if it was built for `grid_thw` and the tower's current device and dtype.

        Otherwise build a new grid. Building reads `grid_thw` on the host, so it cannot happen inside
        `torch.export`: run one eager call first, and again after moving or casting the model.
        """
        if (
            cache is None
            or len(cache.grid_thw) != grid_thw.shape[0]
            or (cache.device, cache.dtype) != (visual.device, visual.dtype)
        ):
            if torch.compiler.is_exporting():
                raise RuntimeError(
                    "precompute_grid_thw has no image grid cached for this model, device and dtype. Run one "
                    "eager call with the same inputs before torch.export."
                )
            return cls.build(visual, grid_thw)
        cache.check(grid_thw)
        return cache

    def check(self, grid_thw: Tensor) -> None:
        """Raise if `grid_thw` differs from the cached grid (a runtime assert in an exported program)."""
        torch_compilable_check(
            grid_thw == grid_thw.new_tensor(self.grid_thw),
            "The image grid differs from the one cached for precompute_grid_thw.",
        )


def cumulative_lengths(lengths: list[int], device: torch.device) -> Tensor:
    """`cu_seqlens` for a vision block, built from Python ints so it is a constant under `torch.export`."""
    total, cumulative = 0, [0]
    for length in lengths:
        total += length
        cumulative.append(total)
    return torch.tensor(cumulative, dtype=torch.int32, device=device)


def qwen_image_rope_index(
    input_ids: Tensor,
    attention_mask: Tensor,
    image_grid_thw: list[list[int]],
    spatial_merge_size: int,
    image_token_id: int,
) -> Tensor:
    """The 3D rope positions of Qwen2.5-VL and Qwen3-VL `get_rope_index` for images, with tensor ops only.

    Text tokens count up by one. An image's tokens get `(start, start + row, start + col)` and move the
    next start by `max(rows, cols)`. Padding gets 0. `image_grid_thw` lists the (t, h, w) grid of every
    image block in `input_ids`, in order, as Python ints, including blocks hidden by `attention_mask`.

    This is what transformers 4.x and 5.5.4+ compute. For Qwen2.5-VL, 5.4.0 to 5.5.3 also scale the
    temporal position of still images (a bug fixed upstream in 5.5.4), so their stock positions differ.
    """
    device = input_ids.device
    valid = attention_mask.bool()
    is_image = input_ids == image_token_id

    # Per image token, in batch order: its (t, h, w) offset inside the image, and how far the image
    # moves the text position, put on the image's last token so earlier tokens keep the image's start.
    offsets, advances = [], []
    for t, h, w in image_grid_thw:
        h, w = h // spatial_merge_size, w // spatial_merge_size
        rows = torch.arange(h, device=device).repeat_interleave(w * t)
        cols = torch.arange(w, device=device).repeat(h * t)
        offsets.append(torch.stack([torch.zeros_like(rows), rows, cols]))
        advance = torch.zeros(t * h * w, dtype=torch.long, device=device)
        advance[-1] = max(h, w)
        advances.append(advance)
    offsets = torch.cat(offsets, dim=1)
    advances = torch.cat(advances)

    steps = torch.ones_like(input_ids).masked_scatter(is_image, advances) * valid
    start = steps.cumsum(dim=1) - steps
    position_ids = start.unsqueeze(0).expand(3, -1, -1)
    image_offsets = torch.zeros_like(position_ids).masked_scatter(is_image.expand_as(position_ids), offsets)
    return (position_ids + image_offsets).masked_fill(~valid, 0)
