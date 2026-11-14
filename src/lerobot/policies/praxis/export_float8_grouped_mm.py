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

"""The grouped GEMMs of Praxis-1's float8 experts as one TensorRT plugin, for its export.

With `--quantize=fp8` each routed expert multiplies float8 rows, with a float32 scale per row, by its float8
weights, with a float32 scale per output column (`praxis_core.layers._float8_grouped_mm`). PyTorch's
`torch._scaled_grouped_mm` runs only on sm_90 and sm_100, and TensorRT has no layer for it on other GPUs. The
plugin is one Triton kernel that multiplies each row by its own expert's weights on float8 tensor cores
(`mma.sync` on sm_89, sm_120 and sm_121, `wgmma` on sm_90) and applies both scales to the float32 sums. Its
results differ from the operator's PyTorch side (`_float8_grouped_mm_pytorch`) only in the order of those sums.

Importing this module registers the operator `lerobot_praxis::float8_grouped_mm` if CUDA is available and the
installed Torch-TensorRT makes TensorRT plugins of Triton kernels and builds this one for the GPU.
`FLOAT8_GROUPED_MM` is then a `praxis_core.layers.Float8GroupedMM` that calls it, and None otherwise. A part that
traces it names this module in its `operator_modules`.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from .export_top_k_sum import _triton_op
from .praxis_core import layers as praxis_layers

logger = logging.getLogger(__name__)

# 128 x 128 x 64 tiles with 3 pipeline stages need at most the 48 KiB of shared memory that TensorRT gives a
# plugin kernel. Float8 tensor cores take 32 values of K per step, so a 64 deep tile keeps them busy.
BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES = 128, 128, 64, 3
# The most experts the kernel finds a row tile's expert among, a power of two.
MAX_EXPERTS = 128
# The kernel reads 16 float8 values at a time along K and writes rows of N bfloat16 values.
ALIGNMENT = 16


def _float8_grouped_mm_meta(
    rows: Tensor, rows_scale: Tensor, bank: Tensor, bank_scale: Tensor, offsets: Tensor
) -> Tensor:
    return rows.new_empty(rows.shape[0], bank.shape[1], dtype=torch.bfloat16)


def _float8_grouped_mm_eager(
    rows: Tensor, rows_scale: Tensor, bank: Tensor, bank_scale: Tensor, offsets: Tensor
) -> Tensor:
    return praxis_layers._float8_grouped_mm_pytorch(
        rows, rows_scale, bank, bank_scale, offsets, torch.bfloat16
    )


def _register() -> praxis_layers.Float8GroupedMM | None:
    triton_op = _triton_op()
    if triton_op is None or not torch.cuda.is_available():
        return None
    import triton
    import triton.language as tl

    @triton.jit
    def float8_grouped_mm_kernel(  # noqa: N803
        rows,
        rows_scale,
        bank,
        bank_scale,
        offsets,
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
        # N and K are multiples of 16 (`float8_grouped_mm` checks). Written so, the compiler knows it, and
        # pipelines and widens the loads as it does for constant sizes.
        N = (N // 16) * 16
        K = (K // 16) * 16
        # N tiles vary fastest, so the row tiles of one expert run together and share its weights through L2.
        pid = tl.program_id(0)
        tiles_n = tl.cdiv(N, BLOCK_N)
        tile = pid // tiles_n
        cols = (pid % tiles_n) * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = cols < N
        experts = tl.arange(0, MAX_EXPERTS)
        valid = experts < EXPERTS
        starts = tl.load(offsets + experts - 1, mask=valid & (experts > 0), other=0)
        ends = tl.where(valid, tl.load(offsets + experts, mask=valid, other=0), starts)
        expert_tiles = (ends - starts + BLOCK_M - 1) // BLOCK_M
        tiles_through = tl.cumsum(expert_tiles, axis=0)
        total_tiles = tl.sum(expert_tiles, axis=0)
        if tile >= total_tiles:
            # Rows past the last offset belong to no expert: zero them.
            r = tl.load(offsets + EXPERTS - 1) + (tile - total_tiles) * BLOCK_M + tl.arange(0, BLOCK_M)
            tl.store(
                out + r.to(tl.int64)[:, None] * N + cols[None, :],
                tl.zeros((BLOCK_M, BLOCK_N), dtype=out.dtype.element_ty),
                mask=((r >= 0) & (r < M))[:, None] & col_mask[None, :],
            )
            return
        expert = tl.sum((tiles_through <= tile).to(tl.int32), axis=0)
        selected = experts == expert
        first_tile = tl.sum(tl.where(selected, tiles_through - expert_tiles, 0), axis=0)
        r = (
            tl.sum(tl.where(selected, starts, 0), axis=0)
            + (tile - first_tile) * BLOCK_M
            + tl.arange(0, BLOCK_M)
        )
        row_mask = (r >= 0) & (r < tl.minimum(tl.sum(tl.where(selected, ends, 0), axis=0), M))
        ks = tl.arange(0, BLOCK_K)
        a = rows + r.to(tl.int64)[:, None] * K + ks[None, :]
        b = bank + expert.to(tl.int64) * K * N + cols.to(tl.int64)[None, :] * K + ks[:, None]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = ks < K - k * BLOCK_K
            a_tile = tl.load(a, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            b_tile = tl.load(b, mask=k_mask[:, None] & col_mask[None, :], other=0.0)
            acc = tl.dot(a_tile, b_tile, acc)
            a += BLOCK_K
            b += BLOCK_K
        a_scale = tl.load(rows_scale + r, mask=row_mask, other=0.0)
        b_scale = tl.load(bank_scale + expert.to(tl.int64) * N + cols, mask=col_mask, other=0.0)
        acc = acc * a_scale[:, None] * b_scale[None, :]
        tl.store(
            out + r.to(tl.int64)[:, None] * N + cols[None, :],
            acc.to(out.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )

    try:
        triton_op(
            "lerobot_praxis::float8_grouped_mm",
            float8_grouped_mm_kernel,
            {
                "rows": "*fp8e4nv",
                "rows_scale": "*fp32",
                "bank": "*fp8e4nv",
                "bank_scale": "*fp32",
                "offsets": "*i32",
                "M": "i32",
                "N": "i32",
                "K": "i32",
                "EXPERTS": "i32",
                "out": "*bf16",
            },
            {"MAX_EXPERTS": MAX_EXPERTS, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
            # Each expert and the rows past the last offset can end in a partial row tile.
            grid=lambda inputs, outputs: (
                ((inputs[0].shape_expr[0] + BLOCK_M - 1) // BLOCK_M + inputs[2].shape_expr[0])
                * ((inputs[2].shape_expr[1] + BLOCK_N - 1) // BLOCK_N),
            ),
            meta_fn=_float8_grouped_mm_meta,
            extra_args_fn=lambda inputs, outputs: [
                inputs[0].shape_expr[0],
                inputs[2].shape_expr[1],
                inputs[2].shape_expr[2],
                inputs[2].shape_expr[0],
            ],
            eager_fn=_float8_grouped_mm_eager,
            num_warps=4,
            num_stages=NUM_STAGES,
        )
    except RuntimeError as error:
        # Such as a GPU whose float8 tensor cores a plugin kernel cannot drive (sm_100 and later need tensor memory).
        logger.warning("The float8 grouped GEMM plugin does not build for this GPU: %s", error)
        return None

    def float8_grouped_mm(
        rows: Tensor, rows_scale: Tensor, bank: Tensor, bank_scale: Tensor, offsets: Tensor
    ) -> Tensor:
        experts, n, k = bank.shape
        if n % ALIGNMENT or k % ALIGNMENT or experts > MAX_EXPERTS:
            raise ValueError(
                f"The float8 grouped GEMM plugin takes at most {MAX_EXPERTS} experts whose weights' sizes are "
                f"multiples of {ALIGNMENT}, not {experts} experts of {k} x {n}."
            )
        return torch.ops.lerobot_praxis.float8_grouped_mm(rows, rows_scale, bank, bank_scale, offsets)

    return float8_grouped_mm


FLOAT8_GROUPED_MM = _register()
