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

"""The NVFP4 experts of Praxis-1 as TensorRT plugins, for its export.

With `--quantize=nvfp4` the routed experts multiply NVFP4 rows by NVFP4 weights (`praxis_core.layers._nvfp4_experts`),
as the original implementation's NVFP4 inference does. TensorRT has no grouped GEMM layer for that (its
mixture-of-experts layer takes no NVFP4 activations and runs on no sm_12x GPU), so three Triton kernels become
plugins:

- `nvfp4_row_globals`: the global scale of each row, from the largest magnitude of its segment's rows;
- `take_nvfp4_rows` and `nvfp4_rows`: the NVFP4 codes and block scales of the rows (the first gathers them);
- `nvfp4_gate_up` and `nvfp4_down`: each row times its own expert's weights on block-scaled FP4 tensor cores
  (`tl.dot_scaled`: `tcgen05` on sm_100 and sm_110, `mma.sync` on sm_120 and sm_121), float32 sums, bfloat16
  results multiplied by the global scales. The first also applies SwiGLU and returns the largest magnitude of each
  row's columns in each program, so TensorRT never writes or reads back the gate and up products.

The first two have the bits of their PyTorch sides (`_nvfp4_row_globals`, `_nvfp4_rows`); the grouped GEMMs differ
from `_nvfp4_gate_up` and `_nvfp4_down`, as the export runs them (with `set_custom_kernels(False)`, so with
`F.silu`), only in the order of the float32 sums. Run eagerly, the operators launch the same kernels.

Importing this module registers the operators if CUDA is available and the installed Torch-TensorRT makes TensorRT
plugins of Triton kernels and builds the grouped GEMMs for the GPU. `NVFP4_KERNELS` is then a
`praxis_core.layers.Nvfp4Kernels` that calls them, and None otherwise. A part that traces them names this module in
its `operator_modules`.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from .export_top_k_sum import _triton_op
from .praxis_core import layers as praxis_layers

logger = logging.getLogger(__name__)

# (rows, columns) that one program of the row kernels rounds; the columns a multiple of the NVFP4 block.
ROWS_TILE = (16, 256)
# The rows of a segment that one step of the global scale kernel reads.
GLOBALS_BLOCK = 1024
# (BLOCK_M, BLOCK_N, BLOCK_K) of the grouped GEMMs and their pipeline depths, deepest first: the fastest measured on
# Jetson Thor, within 5% of the fastest on DGX Spark. TensorRT gives a plugin kernel 48 KiB of shared memory, so each
# kernel gets the first depth that fits. A gate|up program computes BLOCK_N / 2 gate and as many up columns.
TILES = (128, 64, 256)
NUM_STAGES = (2, 1)
# The most experts the kernel finds a row tile's expert among, a power of two.
MAX_EXPERTS = 128
# bfloat16(1e-12), the smallest largest magnitude `_nvfp4_global` allows.
_TINY = torch.tensor(1e-12, dtype=torch.bfloat16).item()


def _row_globals_meta(row_largest: Tensor, segment_ends: Tensor) -> Tensor:
    return row_largest.new_empty(row_largest.shape[0], dtype=torch.float32)


# The gate|up operator takes the bank as (experts, 2, programs, columns, K / 8) words, gate then up, `programs` times
# `columns` = H: Torch-TensorRT derives a plugin's output sizes from its inputs' with sums and products only.
def _gate_up_meta(
    codes: Tensor,
    scales: Tensor,
    bank: Tensor,
    bank_scales: Tensor,
    bank_global: Tensor,
    offsets: Tensor,
    row_global: Tensor,
) -> tuple[Tensor, Tensor]:
    return (
        codes.new_empty(codes.shape[0], bank.shape[2] * bank.shape[3], dtype=torch.bfloat16),
        codes.new_empty(codes.shape[0], bank.shape[2], dtype=torch.bfloat16),
    )


def _down_meta(
    codes: Tensor,
    scales: Tensor,
    bank: Tensor,
    bank_scales: Tensor,
    bank_global: Tensor,
    offsets: Tensor,
    row_global: Tensor,
) -> Tensor:
    return codes.new_empty(codes.shape[0], bank.shape[1], dtype=torch.bfloat16)


# The row operators take the rows as (rows, blocks, 16) and return the codes as (rows, blocks, 2) words: Torch-TensorRT
# derives a plugin's output sizes from its inputs' with sums and products only.
def _rows_meta(blocks: Tensor, row_global: Tensor) -> tuple[Tensor, Tensor]:
    rows, count, _ = blocks.shape
    return (
        blocks.new_empty(rows, count, 2, dtype=torch.int32),
        blocks.new_empty(rows, count, dtype=torch.float8_e4m3fn),
    )


def _take_rows_meta(blocks: Tensor, index: Tensor, row_global: Tensor) -> tuple[Tensor, Tensor]:
    return _rows_meta(blocks.new_empty(index.shape[0], *blocks.shape[1:]), row_global)


def _row_grid(rows, blocks):
    """The programs of the row kernels for `rows` rows of `blocks` (rows, blocks, 16)."""
    return (
        (rows + ROWS_TILE[0] - 1) // ROWS_TILE[0],
        (blocks.shape_expr[1] * 16 + ROWS_TILE[1] - 1) // ROWS_TILE[1],
    )


def _register() -> praxis_layers.Nvfp4Kernels | None:
    triton_op = _triton_op()
    if triton_op is None or not torch.cuda.is_available():
        return None
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def e2m1_pairs(even, odd, NATIVE: tl.constexpr):  # noqa: N803
        # The codes of two values in one byte, the even column in the low nibble.
        if NATIVE:
            # sm_100 and later convert in one instruction, to `e2m1`'s code for every float32 but NaN (checked
            # exhaustively on Jetson Thor and DGX Spark), which `e2m1` turns into a signed zero first.
            even_zero = (even.to(tl.uint32, bitcast=True) & 0x80000000).to(tl.float32, bitcast=True)
            odd_zero = (odd.to(tl.uint32, bitcast=True) & 0x80000000).to(tl.float32, bitcast=True)
            even = tl.where(even != even, even_zero, even)
            odd = tl.where(odd != odd, odd_zero, odd)
            return tl.inline_asm_elementwise(
                "{ .reg .b8 byte; cvt.rn.satfinite.e2m1x2.f32 byte, $2, $1; cvt.u32.u8 $0, byte; }",
                constraints="=r,f,f",
                args=(even, odd),
                dtype=tl.uint32,
                is_pure=True,
                pack=1,
            ).to(tl.uint8)
        else:
            return e2m1(even) | (e2m1(odd) << 4)

    @triton.jit
    def e2m1(v):
        # Nearest FP4 e2m1 value, ties to the even code, as `_e2m1_codes`.
        m = tl.abs(v)
        code = (
            (m > 0.25).to(tl.uint8)
            + (m >= 0.75).to(tl.uint8)
            + (m > 1.25).to(tl.uint8)
            + (m >= 1.75).to(tl.uint8)
            + (m > 2.5).to(tl.uint8)
            + (m >= 3.5).to(tl.uint8)
            + (m > 5.0).to(tl.uint8)
        )
        return code | ((v.to(tl.uint32, bitcast=True) >> 28).to(tl.uint8) & 8)

    @triton.jit
    def round_rows(  # noqa: N803
        x,
        source,
        rows,
        count,
        width,
        row_global,
        codes,
        scales,
        R: tl.constexpr,
        C: tl.constexpr,
        NATIVE: tl.constexpr,
    ):
        # The int32 words of the codes, written two codes per byte (`Nvfp4Bank`).
        codes = codes.to(tl.pointer_type(tl.uint8))
        live = rows < count
        columns = tl.program_id(1) * C + tl.arange(0, C)
        values = tl.load(
            x + source[:, None] * width + columns[None, :],
            mask=live[:, None] & (columns[None, :] < width),
            other=0.0,
        ).to(tl.float32)
        blocks = values.reshape(R, C // 16, 16)
        largest = tl.max(tl.abs(blocks), axis=2) * (1.0 / 6.0)
        g = tl.load(row_global + rows, mask=live, other=1.0)[:, None]
        # Correctly rounded divisions, as PyTorch's, so that every block scale and code is `_nvfp4_quantize`'s.
        scale = tl.minimum(tl.maximum(tl.math.div_rn(largest, g), 0.015625), 448.0).to(tl.float8e4nv)
        reciprocal = tl.math.div_rn(tl.math.div_rn(1.0, g), scale.to(tl.float32))
        norm = tl.minimum(tl.maximum(blocks * reciprocal[:, :, None], -6.0), 6.0)
        even, odd = tl.split(norm.reshape(R, C // 2, 2))
        pairs = tl.program_id(1) * (C // 2) + tl.arange(0, C // 2)
        rows = rows.to(tl.int64)
        tl.store(
            codes + rows[:, None] * (width // 2) + pairs[None, :],
            e2m1_pairs(even, odd, NATIVE),
            mask=live[:, None] & (pairs[None, :] < width // 2),
        )
        blocks_index = tl.program_id(1) * (C // 16) + tl.arange(0, C // 16)
        tl.store(
            scales + rows[:, None] * (width // 16) + blocks_index[None, :],
            scale,
            mask=live[:, None] & (blocks_index[None, :] < width // 16),
        )

    @triton.jit
    def take_rows_kernel(  # noqa: N803
        x,
        index,
        row_global,
        count,
        width,
        codes,
        scales,
        R: tl.constexpr,
        C: tl.constexpr,
        NATIVE: tl.constexpr,
    ):
        rows = tl.program_id(0) * R + tl.arange(0, R)
        source = tl.load(index + rows, mask=rows < count, other=0)
        round_rows(x, source, rows, count, width, row_global, codes, scales, R, C, NATIVE)

    @triton.jit
    def rows_kernel(  # noqa: N803
        x, row_global, count, width, codes, scales, R: tl.constexpr, C: tl.constexpr, NATIVE: tl.constexpr
    ):
        rows = tl.program_id(0) * R + tl.arange(0, R)
        round_rows(x, rows.to(tl.int64), rows, count, width, row_global, codes, scales, R, C, NATIVE)

    @triton.jit
    def row_globals_kernel(  # noqa: N803
        row_largest, segment_ends, width, row_global, TINY: tl.constexpr, BLOCK: tl.constexpr
    ):
        # `row_largest` holds `width` largest magnitudes per row.
        segment = tl.program_id(0)
        start = tl.load(segment_ends + segment - 1, mask=segment > 0, other=0)
        end = tl.load(segment_ends + segment)
        largest = tl.zeros((BLOCK,), dtype=tl.float32)
        for first in range(start * width, end * width, BLOCK):
            r = first + tl.arange(0, BLOCK)
            largest = tl.maximum(
                largest, tl.load(row_largest + r, mask=r < end * width, other=0.0).to(tl.float32)
            )
        # `_nvfp4_global`: the largest magnitude is a bfloat16 already.
        g = tl.maximum(tl.max(largest, axis=0), TINY) * (1.0 / 2688.0)
        for first in range(start, end, BLOCK):
            r = first + tl.arange(0, BLOCK)
            tl.store(row_global + r, tl.zeros((BLOCK,), dtype=tl.float32) + g, mask=r < end)

    @triton.jit
    def row_tile(pid, tiles_n, offsets, M, EXPERTS, MAX_EXPERTS: tl.constexpr, BLOCK_M: tl.constexpr):  # noqa: N803
        # The expert, rows and column tile of program `pid`. N tiles vary fastest, so the row tiles of one expert run
        # together and share its weights through L2. Rows past the last offset belong to no expert (`expert` -1).
        tile = pid // tiles_n
        experts = tl.arange(0, MAX_EXPERTS)
        valid = experts < EXPERTS
        starts = tl.load(offsets + experts - 1, mask=valid & (experts > 0), other=0)
        ends = tl.where(valid, tl.load(offsets + experts, mask=valid, other=0), starts)
        expert_tiles = (ends - starts + BLOCK_M - 1) // BLOCK_M
        tiles_through = tl.cumsum(expert_tiles, axis=0)
        total_tiles = tl.sum(expert_tiles, axis=0)
        expert = tl.sum((tiles_through <= tile).to(tl.int32), axis=0)
        selected = experts == expert
        first_tile = tl.sum(tl.where(selected, tiles_through - expert_tiles, 0), axis=0)
        first_row = tl.sum(tl.where(selected, starts, 0), axis=0) + (tile - first_tile) * BLOCK_M
        end = tl.minimum(tl.sum(tl.where(selected, ends, 0), axis=0), M)
        if tile >= total_tiles:
            first_row = tl.load(offsets + EXPERTS - 1) + (tile - total_tiles) * BLOCK_M
            end = M
            expert = -1
        r = first_row + tl.arange(0, BLOCK_M)
        return expert, r, (r >= 0) & (r < end), pid % tiles_n

    @triton.jit
    def expert_dot(  # noqa: N803
        codes,
        scales,
        bank,
        bank_scales,
        expert,
        r,
        row_mask,
        cols,
        col_mask,
        N,
        K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # The float32 sums of rows `r` times columns `cols` of the expert's (N, K) weights.
        pairs = tl.arange(0, BLOCK_K // 2)
        blocks = tl.arange(0, BLOCK_K // 16)
        # The int32 words of the codes, read two codes per byte (`Nvfp4Bank`).
        codes = codes.to(tl.pointer_type(tl.uint8))
        bank = bank.to(tl.pointer_type(tl.uint8))
        a = codes + r.to(tl.int64)[:, None] * (K // 2) + pairs[None, :]
        a_scale = scales + r.to(tl.int64)[:, None] * (K // 16) + blocks[None, :]
        b = bank + expert.to(tl.int64) * N * (K // 2) + cols.to(tl.int64)[None, :] * (K // 2) + pairs[:, None]
        b_scale = bank_scales + expert.to(tl.int64) * N * (K // 16) + cols.to(tl.int64)[:, None] * (K // 16)
        b_scale += blocks[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, K // BLOCK_K):
            acc = tl.dot_scaled(
                tl.load(a, mask=row_mask[:, None], other=0),
                tl.load(a_scale, mask=row_mask[:, None], other=0.0),
                "e2m1",
                tl.load(b, mask=col_mask[None, :], other=0),
                tl.load(b_scale, mask=col_mask[:, None], other=0.0),
                "e2m1",
                acc,
            )
            a += BLOCK_K // 2
            a_scale += BLOCK_K // 16
            b += BLOCK_K // 2
            b_scale += BLOCK_K // 16
        return acc

    @triton.jit
    def rescale(acc, row_global, bank_global, expert, r, row_mask):
        # `_nvfp4_rescale`: the bfloat16 result times the bfloat16 product of the row's and the expert's scales.
        multiplier = tl.load(row_global + r, mask=row_mask, other=0.0) * tl.load(bank_global + expert)
        return (acc.to(tl.bfloat16).to(tl.float32) * multiplier.to(tl.bfloat16).to(tl.float32)[:, None]).to(
            tl.bfloat16
        )

    @triton.jit
    def gate_up_kernel(  # noqa: N803
        codes,
        scales,
        bank,
        bank_scales,
        bank_global,
        offsets,
        row_global,
        M,
        H,
        K,
        EXPERTS,
        hidden,
        largest,
        MAX_EXPERTS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # K is a multiple of BLOCK_K and H of BLOCK_N / 2 (`gate_up` checks). Written so, the compiler knows it.
        COLUMNS: tl.constexpr = BLOCK_N // 2
        H = (H // COLUMNS) * COLUMNS
        K = (K // BLOCK_K) * BLOCK_K
        programs = H // COLUMNS
        expert, r, row_mask, tile_n = row_tile(
            tl.program_id(0), programs, offsets, M, EXPERTS, MAX_EXPERTS, BLOCK_M
        )
        columns = tile_n * COLUMNS + tl.arange(0, COLUMNS)
        out = hidden + r.to(tl.int64)[:, None] * H + columns[None, :]
        out_largest = largest + r.to(tl.int64) * programs + tile_n
        if expert < 0:
            tl.store(out, tl.zeros((BLOCK_M, COLUMNS), dtype=tl.bfloat16), mask=row_mask[:, None])
            tl.store(out_largest, tl.zeros((BLOCK_M,), dtype=tl.bfloat16), mask=row_mask)
            return
        # Accumulator column 2j is gate column j and 2j + 1 the same up column, so one tile holds both.
        c = tl.arange(0, BLOCK_N)
        cols = tile_n * COLUMNS + c // 2 + (c % 2) * H
        acc = expert_dot(
            codes,
            scales,
            bank,
            bank_scales,
            expert,
            r,
            row_mask,
            cols,
            cols < 2 * H,
            2 * H,
            K,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
        )
        gate, up = tl.split(
            rescale(acc, row_global, bank_global, expert, r, row_mask)
            .to(tl.float32)
            .reshape(BLOCK_M, COLUMNS, 2)
        )
        # `F.silu` of the bfloat16 gate, rounded to bfloat16, times up: the same float32 operations as PyTorch's.
        silu = tl.math.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(tl.bfloat16).to(tl.float32)
        product = (silu * up).to(tl.bfloat16)
        tl.store(out, product, mask=row_mask[:, None])
        tl.store(out_largest, tl.max(tl.abs(product), axis=1), mask=row_mask)

    @triton.jit
    def down_kernel(  # noqa: N803
        codes,
        scales,
        bank,
        bank_scales,
        bank_global,
        offsets,
        row_global,
        M,
        N,
        K,
        EXPERTS,
        out,
        MAX_EXPERTS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # K is a multiple of BLOCK_K and N of 16 (`down` checks). Written so, the compiler knows it.
        N = (N // 16) * 16
        K = (K // BLOCK_K) * BLOCK_K
        expert, r, row_mask, tile_n = row_tile(
            tl.program_id(0), tl.cdiv(N, BLOCK_N), offsets, M, EXPERTS, MAX_EXPERTS, BLOCK_M
        )
        cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        out = out + r.to(tl.int64)[:, None] * N + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]
        if expert < 0:
            tl.store(out, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.bfloat16), mask=mask)
            return
        acc = expert_dot(
            codes,
            scales,
            bank,
            bank_scales,
            expert,
            r,
            row_mask,
            cols,
            col_mask,
            N,
            K,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
        )
        tl.store(out, rescale(acc, row_global, bank_global, expert, r, row_mask), mask=mask)

    block_m, block_n, block_k = TILES
    columns = block_n // 2
    native = torch.cuda.get_device_capability()[0] >= 10
    num_stages: dict[str, int] = {}
    constexprs = {"MAX_EXPERTS": MAX_EXPERTS, "BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k}

    # Run eagerly, as in the export's checks, the operators launch the same kernels, so that a part's PyTorch run
    # computes what its program does (and fast: the PyTorch sides dequantize every expert's weights).
    def gate_up_eager(
        codes: Tensor,
        scales: Tensor,
        bank: Tensor,
        bank_scales: Tensor,
        bank_global: Tensor,
        offsets: Tensor,
        row_global: Tensor,
    ) -> tuple[Tensor, Tensor]:
        rows, (experts, _, programs, width, words) = codes.shape[0], bank.shape
        hidden = torch.empty(rows, programs * width, dtype=torch.bfloat16, device=codes.device)
        largest = torch.empty(rows, programs, dtype=torch.bfloat16, device=codes.device)
        grid = ((triton.cdiv(rows, block_m) + experts) * programs,)
        gate_up_kernel[grid](
            codes,
            scales,
            bank,
            bank_scales,
            bank_global,
            offsets,
            row_global,
            rows,
            programs * width,
            8 * words,
            experts,
            hidden,
            largest,
            **constexprs,
            num_warps=4,
            num_stages=num_stages["gate_up"],
        )
        return hidden, largest

    def down_eager(
        codes: Tensor,
        scales: Tensor,
        bank: Tensor,
        bank_scales: Tensor,
        bank_global: Tensor,
        offsets: Tensor,
        row_global: Tensor,
    ) -> Tensor:
        rows, (experts, n, words) = codes.shape[0], bank.shape
        out = torch.empty(rows, n, dtype=torch.bfloat16, device=codes.device)
        grid = ((triton.cdiv(rows, block_m) + experts) * triton.cdiv(n, block_n),)
        down_kernel[grid](
            codes,
            scales,
            bank,
            bank_scales,
            bank_global,
            offsets,
            row_global,
            rows,
            n,
            8 * words,
            experts,
            out,
            **constexprs,
            num_warps=4,
            num_stages=num_stages["down"],
        )
        return out

    def rows_eager(blocks: Tensor, index: Tensor | None, row_global: Tensor) -> tuple[Tensor, Tensor]:
        count = blocks.shape[0] if index is None else index.shape[0]
        codes = blocks.new_empty(count, blocks.shape[1], 2, dtype=torch.int32)
        scales = blocks.new_empty(count, blocks.shape[1], dtype=torch.float8_e4m3fn)
        width = blocks.shape[1] * praxis_layers.NVFP4_BLOCK
        if count:
            grid = (triton.cdiv(count, ROWS_TILE[0]), triton.cdiv(width, ROWS_TILE[1]))
            if index is None:
                rows_kernel[grid](
                    blocks,
                    row_global,
                    count,
                    width,
                    codes,
                    scales,
                    *ROWS_TILE,
                    native,
                    num_warps=4,
                    num_stages=1,
                )
            else:
                take_rows_kernel[grid](
                    blocks,
                    index,
                    row_global,
                    count,
                    width,
                    codes,
                    scales,
                    *ROWS_TILE,
                    native,
                    num_warps=4,
                    num_stages=1,
                )
        return codes, scales

    def row_globals_eager(row_largest: Tensor, segment_ends: Tensor) -> Tensor:
        out = row_largest.new_empty(row_largest.shape[0], dtype=torch.float32)
        row_globals_kernel[(segment_ends.shape[0],)](
            row_largest,
            segment_ends,
            row_largest.shape[1],
            out,
            _TINY,
            GLOBALS_BLOCK,
            num_warps=4,
            num_stages=1,
        )
        return out

    bank_signature = {
        "codes": "*i32",
        "scales": "*fp8e4nv",
        "bank": "*i32",
        "bank_scales": "*fp8e4nv",
        "bank_global": "*fp32",
        "offsets": "*i32",
        "row_global": "*fp32",
        "M": "i32",
    }
    gemms = {
        "gate_up": {
            "kernel": gate_up_kernel,
            "signature": {
                **bank_signature,
                "H": "i32",
                "K": "i32",
                "EXPERTS": "i32",
                "hidden": "*bf16",
                "largest": "*bf16",
            },
            # Each expert and the rows past the last offset can end in a partial row tile.
            "grid": lambda inputs, outputs: (
                ((inputs[0].shape_expr[0] + block_m - 1) // block_m + inputs[2].shape_expr[0])
                * inputs[2].shape_expr[2],
            ),
            "meta_fn": _gate_up_meta,
            "extra_args_fn": lambda inputs, outputs: [
                inputs[0].shape_expr[0],
                inputs[2].shape_expr[2] * inputs[2].shape_expr[3],
                inputs[2].shape_expr[4] * 8,
                inputs[2].shape_expr[0],
            ],
            "eager_fn": gate_up_eager,
        },
        "down": {
            "kernel": down_kernel,
            "signature": {**bank_signature, "N": "i32", "K": "i32", "EXPERTS": "i32", "out": "*bf16"},
            "grid": lambda inputs, outputs: (
                ((inputs[0].shape_expr[0] + block_m - 1) // block_m + inputs[2].shape_expr[0])
                * ((inputs[2].shape_expr[1] + block_n - 1) // block_n),
            ),
            "meta_fn": _down_meta,
            "extra_args_fn": lambda inputs, outputs: [
                inputs[0].shape_expr[0],
                inputs[2].shape_expr[1],
                inputs[2].shape_expr[2] * 8,
                inputs[2].shape_expr[0],
            ],
            "eager_fn": down_eager,
        },
    }
    for name, gemm in gemms.items():
        failure: Exception | None = None
        for stages in NUM_STAGES:
            num_stages[name] = stages
            try:
                triton_op(
                    f"lerobot_praxis::nvfp4_{name}",
                    gemm["kernel"],
                    gemm["signature"],
                    constexprs,
                    grid=gemm["grid"],
                    meta_fn=gemm["meta_fn"],
                    extra_args_fn=gemm["extra_args_fn"],
                    eager_fn=gemm["eager_fn"],
                    num_warps=4,
                    num_stages=stages,
                )
            except ValueError as error:
                # More shared memory than TensorRT launches a plugin kernel with. Torch-TensorRT checks it before it
                # registers anything, so a shallower pipeline can still register the operator.
                failure = error
            except RuntimeError as error:
                # Such as a launch feature the TensorRT plugin path cannot reproduce on this GPU.
                failure = error
                break
            else:
                failure = None
                break
        if failure is not None:
            logger.warning("The NVFP4 %s grouped GEMM plugin does not build for this GPU: %s", name, failure)
            return None

    common = {"R": ROWS_TILE[0], "C": ROWS_TILE[1], "NATIVE": native}
    triton_op(
        "lerobot_praxis::take_nvfp4_rows",
        take_rows_kernel,
        {
            "x": "*bf16",
            "index": "*i64",
            "row_global": "*fp32",
            "count": "i32",
            "width": "i32",
            "codes": "*i32",
            "scales": "*fp8e4nv",
        },
        common,
        grid=lambda inputs, outputs: _row_grid(inputs[1].shape_expr[0], inputs[0]),
        meta_fn=_take_rows_meta,
        extra_args_fn=lambda inputs, outputs: [inputs[1].shape_expr[0], inputs[0].shape_expr[1] * 16],
        eager_fn=rows_eager,
        num_warps=4,
        num_stages=1,
    )
    triton_op(
        "lerobot_praxis::nvfp4_rows",
        rows_kernel,
        {
            "x": "*bf16",
            "row_global": "*fp32",
            "count": "i32",
            "width": "i32",
            "codes": "*i32",
            "scales": "*fp8e4nv",
        },
        common,
        grid=lambda inputs, outputs: _row_grid(inputs[0].shape_expr[0], inputs[0]),
        meta_fn=_rows_meta,
        extra_args_fn=lambda inputs, outputs: [inputs[0].shape_expr[0], inputs[0].shape_expr[1] * 16],
        eager_fn=lambda blocks, row_global: rows_eager(blocks, None, row_global),
        num_warps=4,
        num_stages=1,
    )
    triton_op(
        "lerobot_praxis::nvfp4_row_globals",
        row_globals_kernel,
        {"row_largest": "*bf16", "segment_ends": "*i32", "width": "i32", "row_global": "*fp32"},
        {"TINY": _TINY, "BLOCK": GLOBALS_BLOCK},
        grid=lambda inputs, outputs: (inputs[1].shape_expr[0],),
        meta_fn=_row_globals_meta,
        extra_args_fn=lambda inputs, outputs: [inputs[0].shape_expr[1]],
        eager_fn=row_globals_eager,
        num_warps=4,
        num_stages=1,
    )

    def rows(x: Tensor, index: Tensor | None, row_global: Tensor) -> tuple[Tensor, Tensor]:
        blocks = x.view(x.shape[0], -1, praxis_layers.NVFP4_BLOCK)
        if index is None:
            codes, scales = torch.ops.lerobot_praxis.nvfp4_rows(blocks, row_global)
        else:
            codes, scales = torch.ops.lerobot_praxis.take_nvfp4_rows(blocks, index, row_global)
        return codes.view(codes.shape[0], -1), scales

    def check(bank: praxis_layers.Nvfp4Bank, n_multiple: int) -> None:
        experts, n, words = bank.codes.shape
        if n % n_multiple or (8 * words) % block_k or experts > MAX_EXPERTS:
            raise ValueError(
                f"The NVFP4 grouped GEMM plugins take at most {MAX_EXPERTS} experts with N a multiple of "
                f"{n_multiple} and K of {block_k}, not {experts} experts of {8 * words} x {n}."
            )

    def gate_up(
        codes: Tensor, scales: Tensor, bank: praxis_layers.Nvfp4Bank, offsets: Tensor, row_global: Tensor
    ) -> tuple[Tensor, Tensor]:
        check(bank, 2 * columns)
        experts, n, words = bank.codes.shape
        words = bank.codes.view(experts, 2, n // (2 * columns), columns, words)
        return torch.ops.lerobot_praxis.nvfp4_gate_up(
            codes, scales, words, bank.scales, bank.global_scale, offsets, row_global
        )

    def down(
        codes: Tensor, scales: Tensor, bank: praxis_layers.Nvfp4Bank, offsets: Tensor, row_global: Tensor
    ) -> Tensor:
        check(bank, 16)
        return torch.ops.lerobot_praxis.nvfp4_down(
            codes, scales, bank.codes, bank.scales, bank.global_scale, offsets, row_global
        )

    return praxis_layers.Nvfp4Kernels(torch.ops.lerobot_praxis.nvfp4_row_globals, rows, gate_up, down)


NVFP4_KERNELS = _register()
