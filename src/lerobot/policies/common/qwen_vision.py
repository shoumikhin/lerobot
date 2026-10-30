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
grid can be computed once in eager mode and reused by the exported program. `FixedImageGrid` holds that
cache; the tower-specific subclasses live in `qwen2_5_vision.py` and `qwen3_vision.py`.

The cache is prepared explicitly, by one eager call on the export inputs inside `prepare_image_grid()`:

    with prepare_image_grid():
        policy.predict_action_chunk(batch)
    program = torch.export.export(module, args, strict=False)

Outside that block the policies run their stock code and neither read nor cache the grid. Under export,
`replay_vision_block` runs each vision block from its submodules with the attention chunk lengths as
Python ints, because the stock attention reads them back from a tensor.

Under export the cached grid is checked with transformers' `torch_compilable_check`, a runtime assert in
the program that `TRANSFORMERS_DISABLE_TORCH_CHECK=1` turns off like every other check of that kind; a
failed assert loses its message. `check_input` is for a policy's own input checks: always on in eager mode.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

import torch
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.utils import torch_compilable_check
else:
    ALL_ATTENTION_FUNCTIONS = None
    torch_compilable_check = None

_preparing_image_grid: ContextVar[bool] = ContextVar("preparing_image_grid", default=False)


@contextmanager
def prepare_image_grid() -> Iterator[None]:
    """Cache, in the eager calls inside this block, the image grid metadata that `torch.export` needs.

    Run the policy once on the export inputs, on the export device and dtype, inside the block; then export.
    """
    token = _preparing_image_grid.set(True)
    try:
        yield
    finally:
        _preparing_image_grid.reset(token)


def is_exporting() -> bool:
    """Whether non-strict `torch.export` is tracing."""
    # torch.compile on PyTorch 2.11 also reports is_exporting(); non-strict export runs without Dynamo.
    return torch.compiler.is_exporting() and not torch.compiler.is_dynamo_compiling()


def is_preparing_image_grid() -> bool:
    """Whether an eager call runs inside `prepare_image_grid()`."""
    return _preparing_image_grid.get() and not is_exporting()


@dataclass
class FixedImageGrid:
    """What a vision tower derives from one image grid, built in eager mode.

    Subclasses add the tower's metadata and implement `_from_grid`. Only weight-independent tensors are
    cached, built outside inference mode, so the cache never holds stale weights and an export can trace a
    trainable tower.
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
        # Preparation usually runs a prediction in inference mode, but the cache must hold normal tensors:
        # inference tensors cannot be saved for backward when the export traces a trainable tower.
        with torch.inference_mode(False), torch.no_grad():
            return cls._from_grid(visual, grid_thw.tolist())

    @classmethod
    def track(cls, cache: Self | None, visual: nn.Module, grid_thw: Tensor) -> Self | None:
        """Return the cache a policy keeps for `grid_thw`.

        Inside `prepare_image_grid()` an eager call builds it, and rebuilds it whenever the grid, the device or
        the dtype changed. Under `torch.export` the grid cannot be read, so the prepared cache must fit: the
        image count, device and dtype are checked while tracing, and the grid values by a runtime assert in
        the program. Otherwise the cache is returned as it is and the grid is not read.
        """
        if is_exporting():
            if cache is None:
                problem = "no image grid was prepared"
            elif len(cache.grid_thw) != grid_thw.shape[0]:
                problem = (
                    f"the prepared grid has {len(cache.grid_thw)} images but the export inputs have "
                    f"{grid_thw.shape[0]} (another batch size or camera count)"
                )
            elif (cache.device, cache.dtype) != (visual.device, visual.dtype):
                problem = (
                    f"the grid was prepared for {cache.device} {cache.dtype}, but the tower is now on "
                    f"{visual.device} {visual.dtype}"
                )
            else:
                cache.check(grid_thw)
                return cache
            raise RuntimeError(
                f"Cannot export on a fixed image grid: {problem}. Run the policy once on the export inputs "
                "inside `prepare_image_grid()` before torch.export."
            )
        if is_preparing_image_grid() and (
            cache is None
            or (cache.device, cache.dtype) != (visual.device, visual.dtype)
            or cache.grid_thw != grid_thw.tolist()
        ):
            return cls.build(visual, grid_thw)
        return cache

    def check(self, grid_thw: Tensor) -> None:
        """Assert in an exported program that `grid_thw` equals the cached grid."""
        torch_compilable_check(
            grid_thw == grid_thw.new_tensor(self.grid_thw),
            "The image grid differs from the one prepared for torch.export.",
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


def chunked_vision_attention(
    attn: nn.Module,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    lengths: list[int],
    apply_rotary_pos_emb_vision: Callable,
    eager_attention_forward: Callable,
) -> Tensor:
    """The non-flash path of a stock Qwen-VL vision attention, split per chunk of `lengths` (Python ints).

    The stock attention splits by `cu_seqlens`, which it reads back from a tensor. The last two arguments come
    from the tower's transformers modeling module.
    """
    seq_length = hidden_states.shape[0]
    query, key, value = (
        attn.qkv(hidden_states).reshape(seq_length, 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
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
    return attn.proj(torch.cat(outputs, dim=1).reshape(seq_length, -1).contiguous())


def replay_vision_block(
    block: nn.Module,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    lengths: list[int],
    apply_rotary_pos_emb_vision: Callable,
    eager_attention_forward: Callable,
) -> Tensor:
    """Under `torch.export`, what a Qwen-VL vision block computes when its attention runs per chunk of `lengths`.

    The attention is `chunked_vision_attention`, unless the block's attention module has its own
    `replay_on_lengths(hidden_states, position_embeddings, lengths)` (WALL-X's packed varlen attention does).
    """
    attn, normed = block.attn, block.norm1(hidden_states)
    replay = getattr(attn, "replay_on_lengths", None)
    if replay is None:
        attended = chunked_vision_attention(
            attn, normed, position_embeddings, lengths, apply_rotary_pos_emb_vision, eager_attention_forward
        )
    else:
        attended = replay(normed, position_embeddings, lengths)
    hidden_states = hidden_states + attended
    return hidden_states + block.mlp(block.norm2(hidden_states))


def qwen_image_rope_index(
    input_ids: Tensor,
    attention_mask: Tensor,
    image_grid_thw: list[list[int]],
    spatial_merge_size: int,
    image_token_id: int,
    image_temporal_scale: int = 1,
    image_blocks: Literal["visible_runs", "vision_starts", "every_image"] = "visible_runs",
    vision_start_token_id: int | None = None,
) -> Tensor:
    """The 3D rope positions that Qwen-VL `get_rope_index` code gives still images, with tensor ops only.

    Text tokens count up by one. An image block's tokens get `(start * image_temporal_scale, start + row,
    start + col)` and move the next start by `max(rows, cols)`. Padding gets 0. `image_grid_thw` lists the
    (t, h, w) grid of every image in `input_ids`, in order, as Python ints, including images hidden by
    `attention_mask`. Only still images (t == 1) are supported, as the image processors produce them; video
    grids raise.

    `image_blocks` says how the eager code finds the image blocks and gives them grids:

    - `"visible_runs"`, as transformers 5.x `get_rope_index`: each run of visible image tokens is an image and
      takes the next grid, counting across the batch. A run that does not cover exactly its grid makes the
      eager code fail (or place the image's positions on other tokens) and is refused.
    - `"vision_starts"`, as the transformers 4.x `get_rope_index` that WALL-X and Wall-OSS keep: each visible
      `vision_start_token_id` followed by a visible image token counts an image, which takes the next grid and
      covers as many visible tokens as the grid has, from the next visible image token on. Layouts the eager
      code fails on (more images than grids, a block running past the visible tokens, a start token last) are
      refused.
    - `"every_image"`: every image block keeps its own grid, hidden or not, as for eager code that drops the
      grids of hidden images before calling `get_rope_index`.

    A refusal is a `ValueError` in eager mode and a runtime assert in an exported program.

    For Qwen2.5-VL, transformers 5.4.0 to 5.5.3 multiply the temporal position of an image's start by
    `tokens_per_second` (fixed upstream in 5.5.4); `image_temporal_scale` reproduces that, and its default of
    1 matches transformers 4.x, 5.5.4 and later, and Qwen3-VL.
    """
    if any(t != 1 for t, _, _ in image_grid_thw):
        raise ValueError(f"qwen_image_rope_index supports still images only (t == 1), got {image_grid_thw}.")
    sizes = [(h // spatial_merge_size, w // spatial_merge_size) for _, h, w in image_grid_thw]
    valid = attention_mask.bool()
    if image_blocks == "every_image":
        in_block = input_ids == image_token_id
        grid_index, offset = _every_image_blocks(in_block, sizes)
    elif image_blocks == "visible_runs":
        in_block = valid & (input_ids == image_token_id)
        grid_index, offset = _visible_run_blocks(in_block, valid, sizes)
    elif image_blocks == "vision_starts" and vision_start_token_id is not None:
        in_block, grid_index, offset = _vision_start_blocks(
            input_ids, valid, sizes, image_token_id, vision_start_token_id
        )
    else:
        raise ValueError(f"Unknown image_blocks {image_blocks!r}, or no vision_start_token_id given.")

    device = input_ids.device
    grid_index = grid_index.clamp(0, len(sizes) - 1)
    width = torch.tensor([w for _, w in sizes], device=device)[grid_index]
    last = torch.tensor([h * w - 1 for h, w in sizes], device=device)[grid_index]
    advance = torch.tensor([max(h, w) for h, w in sizes], device=device)[grid_index]
    # An image moves the text position on its last token, so its earlier tokens keep the image's start.
    steps = torch.where(in_block, torch.where(offset == last, advance, 0), 1) * valid
    start = steps.cumsum(dim=1) - steps
    temporal = torch.where(in_block, start * image_temporal_scale, start)
    rows = torch.where(in_block, offset // width, 0)
    cols = torch.where(in_block, offset % width, 0)
    return torch.stack([temporal, start + rows, start + cols]).masked_fill(~valid, 0)


def _every_image_blocks(is_image: Tensor, sizes: list[tuple[int, int]]) -> tuple[Tensor, Tensor]:
    """Per image token in batch order: the index of its image, and its index inside the image."""
    device = is_image.device
    grid_index = torch.cat([torch.full((h * w,), i, device=device) for i, (h, w) in enumerate(sizes)])
    offset = torch.cat([torch.arange(h * w, device=device) for h, w in sizes])
    zeros = torch.zeros(is_image.shape, dtype=torch.long, device=device)
    return zeros.masked_scatter(is_image, grid_index), zeros.masked_scatter(is_image, offset)


def _visible_ranks(flags: Tensor, valid: Tensor) -> Tensor:
    """`flags` of the visible tokens of each row, moved to their rank among the visible tokens.

    Has one extra column, so that the rank after the last visible token can be read; it is False.
    """
    batch, length = valid.shape
    # Hidden tokens go to distinct columns past the visible ones, so no two tokens write the same column.
    rank = torch.where(valid, valid.cumsum(dim=1) - 1, length + torch.arange(length, device=valid.device))
    ranked = torch.zeros(batch, 2 * length, dtype=torch.bool, device=valid.device)
    return ranked.scatter(1, rank, flags & valid)[:, : length + 1]


def _visible_run_blocks(
    is_image: Tensor, valid: Tensor, sizes: list[tuple[int, int]]
) -> tuple[Tensor, Tensor]:
    """The `"visible_runs"` blocks of `qwen_image_rope_index`: grid index and index inside it per token."""
    rank = valid.cumsum(dim=1) - 1
    ranked = _visible_ranks(is_image, valid)
    after_image = (rank > 0) & ranked.gather(1, (rank - 1).clamp(min=0))
    before_image = ranked.gather(1, (rank + 1).clamp(0, valid.shape[1]))
    runs = (is_image & ~after_image).flatten().cumsum(0).view_as(is_image) - 1
    grid_index = runs.clamp(0, len(sizes) - 1)
    counts = [h * w for h, w in sizes]
    first_token = torch.tensor([sum(counts[:i]) for i in range(len(counts))], device=valid.device)
    offset = is_image.flatten().cumsum(0).view_as(is_image) - 1 - first_token[grid_index]
    # Runs are numbered in order, so a run starts at its grid's first token when every earlier run ended on
    # its grid's last one; a run past the last grid then cannot end on that grid's last token either.
    run_ends_on_grid_end = offset == torch.tensor(counts, device=valid.device)[grid_index] - 1
    check_input(
        ~is_image | before_image | run_ends_on_grid_end,
        "Each run of visible image tokens must be one whole image, with at most as many runs as images, to run "
        "on a fixed image grid: eager rope positions fail or misplace an image otherwise.",
    )
    return grid_index, offset


def _vision_start_blocks(
    input_ids: Tensor,
    valid: Tensor,
    sizes: list[tuple[int, int]],
    image_token_id: int,
    vision_start_token_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """The `"vision_starts"` blocks of `qwen_image_rope_index`: membership, grid index and index inside it."""
    device, length = input_ids.device, input_ids.shape[1]
    visible = valid.sum(dim=1)
    image_at = _visible_ranks(input_ids == image_token_id, valid)
    start_at = _visible_ranks(input_ids == vision_start_token_id, valid)
    image_count = (start_at[:, :-1] & image_at[:, 1:]).sum(dim=1)
    first_grid = image_count.cumsum(0) - image_count
    ranks = torch.arange(length + 1, device=device)
    counts = torch.tensor([h * w for h, w in sizes], device=device)
    # The eager code fails on a start token as the last visible token and on more images than grids.
    fits = (image_count.sum() <= len(sizes)) & ~start_at.gather(1, (visible - 1).clamp(min=0)[:, None]).any()
    # The blocks follow each other, so they are found one per image, at most one per grid.
    end = torch.zeros_like(visible)
    begins, ends = [], []
    for k in range(len(sizes)):
        counted = k < image_count
        begin = torch.where(image_at & (ranks >= end[:, None]), ranks, length + 1).amin(dim=1)
        block_end = begin + counts[(first_grid + k).clamp(max=len(sizes) - 1)]
        fits &= (~counted | (block_end <= visible)).all()
        begins.append(torch.where(counted, begin, length + 1))
        ends.append(torch.where(counted, block_end, length + 1))
        end = torch.where(counted, block_end, end)
    check_input(
        fits,
        "Each image counted by a visible vision start token must have a grid and lie within the visible "
        "tokens to run on a fixed image grid: eager rope positions fail otherwise.",
    )
    rank = (valid.cumsum(dim=1) - 1)[:, None]
    begins, ends = torch.stack(begins, dim=1)[..., None], torch.stack(ends, dim=1)[..., None]
    inside = valid[:, None] & (rank >= begins) & (rank < ends)
    block = inside.long().argmax(dim=1)
    offset = rank[:, 0] - begins[..., 0].gather(1, block)
    return inside.any(dim=1), first_grid[:, None] + block, offset
