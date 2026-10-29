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

"""Shared pieces for running transformers Qwen-VL vision towers under `torch.export` on a fixed image grid.

The stock towers turn the image grid (`image_grid_thw`) into Python ints on every call, which
`torch.export` cannot trace. When every batch uses the same camera grid, what a tower derives from the
grid can be computed once in eager mode and reused. `FixedImageGrid` holds that cache and decides when it
can be reused; the tower-specific subclasses live in `qwen2_5_vision.py` and `qwen3_vision.py`.

Only an exported program needs the cache. In eager mode the policies keep their stock calls, so FSDP and
gradient checkpointing see exactly the stock module calls, and the cache is only built, and rebuilt
whenever the grid, device or dtype changes, for a later `torch.export`. Under export, `replay_vision_block`
runs each vision block from its submodules with the attention chunk lengths as Python ints, because the
stock attention reads them back from a tensor.

Under export the cached grid is checked with transformers' `torch_compilable_check`, a runtime assert in
the program that `TRANSFORMERS_DISABLE_TORCH_CHECK=1` turns off like every other check of that kind; a
failed assert loses its message. `check_input` is for a policy's own input checks: always on in eager mode.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import torch
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.utils import torch_compilable_check
else:
    ALL_ATTENTION_FUNCTIONS = None
    torch_compilable_check = None


@dataclass
class FixedImageGrid:
    """What a vision tower derives from one image grid, built in eager mode.

    Subclasses add the tower's metadata and implement `_from_grid`. Only weight-independent tensors are
    cached, built outside inference mode, so the cache never holds stale weights and works for inference
    and training alike.
    """

    grid_thw: list[list[int]]
    device: torch.device
    dtype: torch.dtype

    @classmethod
    def _from_grid(cls, visual: nn.Module, grid_thw: list[list[int]]) -> Self:
        raise NotImplementedError

    @classmethod
    def build(cls, visual: nn.Module, grid_thw: Tensor) -> Self:
        """Compute the metadata of `grid_thw` for `visual`; this reads the grid on the host."""
        # A cache built during prediction runs in inference mode, but it must hold normal tensors: inference
        # tensors cannot be saved for backward when a later export traces a trainable tower.
        with torch.inference_mode(False), torch.no_grad():
            return cls._from_grid(visual, grid_thw.tolist())

    @classmethod
    def reuse_or_build(cls, cache: Self | None, visual: nn.Module, grid_thw: Tensor) -> Self:
        """Return the cache for `grid_thw` on the tower's device and dtype.

        In eager mode this reads `grid_thw` on the host and builds a new cache whenever the grid, the device or
        the dtype changed. Under `torch.export` the grid cannot be read, so the cache must already fit: the
        image count, device and dtype are checked while tracing, and the grid values by a runtime assert in the
        program. So the last eager call before export must use the export inputs, on the export device and
        dtype.
        """
        if not torch.compiler.is_exporting():
            if (
                cache is None
                or (cache.device, cache.dtype) != (visual.device, visual.dtype)
                or cache.grid_thw != grid_thw.tolist()
            ):
                return cls.build(visual, grid_thw)
            return cache
        if cache is None:
            problem = "no image grid is cached"
        elif len(cache.grid_thw) != grid_thw.shape[0]:
            problem = (
                f"the cached grid has {len(cache.grid_thw)} images but the export inputs have "
                f"{grid_thw.shape[0]} (another batch size or camera count)"
            )
        elif (cache.device, cache.dtype) != (visual.device, visual.dtype):
            problem = (
                f"the cached grid was built for {cache.device} {cache.dtype}, but the tower is now on "
                f"{visual.device} {visual.dtype}"
            )
        else:
            cache.check(grid_thw)
            return cache
        raise RuntimeError(
            f"precompute_grid_thw cannot export: {problem}. The last eager call before torch.export must use "
            "the export inputs."
        )

    def check(self, grid_thw: Tensor) -> None:
        """Assert in an exported program that `grid_thw` equals the cached grid."""
        torch_compilable_check(
            grid_thw == grid_thw.new_tensor(self.grid_thw),
            "The image grid differs from the one cached for precompute_grid_thw.",
        )


def check_input(
    condition: Tensor | bool, message: str | Callable[[], str], error_type: type[Exception] = ValueError
) -> None:
    """Raise `error_type` with `message` unless `condition` holds (for every element of a tensor).

    In eager mode this is a plain check that always raises, as these input checks were before. While
    `torch.export` or `torch.compile` traces, it is `torch_compilable_check`, a runtime assert in the program
    (which `TRANSFORMERS_DISABLE_TORCH_CHECK=1` turns off).
    """
    if torch.compiler.is_compiling() or torch.compiler.is_exporting():
        torch_compilable_check(condition, message, error_type=error_type)
    elif not bool(torch.as_tensor(condition).all()):
        raise error_type(message() if callable(message) else message)


def replay_vision_block(
    block: nn.Module,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    lengths: list[int],
    apply_rotary_pos_emb_vision: Callable,
    eager_attention_forward: Callable,
) -> Tensor:
    """Under `torch.export`, what a Qwen-VL vision block computes when its attention runs per chunk of `lengths`.

    This is the block forward with the non-flash path of the stock vision attention, split by Python ints
    instead of `cu_seqlens`. The last two arguments come from the tower's transformers modeling module.
    """
    attn, seq_length = block.attn, hidden_states.shape[0]
    query, key, value = (
        attn.qkv(block.norm1(hidden_states))
        .reshape(seq_length, 3, attn.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    query, key = apply_rotary_pos_emb_vision(query, key, *position_embeddings)
    query, key, value = (state.transpose(0, 1).unsqueeze(0) for state in (query, key, value))
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        attn.config._attn_implementation, eager_attention_forward
    )
    outputs = [
        attention_interface(
            attn, q, k, v, attention_mask=None, scaling=attn.scaling, dropout=0.0, is_causal=False
        )[0]
        for q, k, v in zip(
            *(torch.split(state, lengths, dim=2) for state in (query, key, value)), strict=True
        )
    ]
    hidden_states = hidden_states + attn.proj(torch.cat(outputs, dim=1).reshape(seq_length, -1).contiguous())
    return hidden_states + block.mlp(block.norm2(hidden_states))


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
