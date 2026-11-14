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

"""The first half of Praxis-1's float8 experts as one TensorRT plugin, for its export.

With `--quantize=fp8` each mixture-of-experts layer rounds the input row of every (token, expert) pair to float8 in
expert order, multiplies it by its expert's float8 gate and up columns, and applies SwiGLU to the bfloat16 products
(`praxis_core.layers._float8_experts`). As three operations, TensorRT writes the float8 rows and both products to
memory and reads them back. A row's float8 scale depends only on its token, so the tokens are rounded once
(`export_float8_rows`), and the plugin is one Triton kernel that reads each token's float8 row and scale through the
index, multiplies it by the gate and the up columns of its expert on `tcgen05` float8 tensor cores, applies both
scales, and writes only `silu(gate) * up`. It rounds as the operator's PyTorch side does: both scaled products to
bfloat16, then the SiLU, then the product. Its results differ from that side only in the order of the products'
float32 sums, in where the scales apply (that side scales the float8 values before it multiplies them on GPUs where
PyTorch has no float8 grouped GEMM), and in the last bit of the SiLU's exponential.

The kernel needs `tcgen05` (sm_100 and sm_110), which keeps both products in tensor memory. Other tensor cores keep
them in registers, so their tiles hold half as many rows, and the separate operations run instead.

Importing this module registers the operator `lerobot_praxis::float8_gate_up` if the GPU has `tcgen05` and the
installed Torch-TensorRT makes TensorRT plugins of Triton kernels, lets them use tensor memory and builds this one.
`FLOAT8_GATE_UP` is then a `praxis_core.layers.Float8GateUp` that calls it, and None otherwise. A part that traces it
names this module in its `operator_modules`.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from .export_top_k_sum import _triton_op
from .praxis_core import layers as praxis_layers

logger = logging.getLogger(__name__)

# Each program computes the same columns of gate and of up, in two accumulators in tensor memory. Float8 tensor cores
# take 32 values of K per step, so a 64 deep tile keeps them busy, and 2 pipeline stages of these tiles fit the
# 48 KiB of shared memory TensorRT gives a plugin kernel (40 KiB; 3 would need 60 KiB).
BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES = 64, 128, 64, 4, 2
# The most experts the kernel finds a row tile's expert among, a power of two.
MAX_EXPERTS = 128
# The kernel reads 16 float8 values at a time along K and writes rows of bfloat16 values 16 at a time.
ALIGNMENT = 16


# The operator takes the bank's scales as (EXPERTS, 2, H): Torch-TensorRT derives the plugin's output shape from its
# inputs' dimensions only by sums and products, and TensorRT takes a float8 constant, the bank, only as it is stored.
def _float8_gate_up_meta(
    tokens: Tensor, tokens_scale: Tensor, index: Tensor, bank: Tensor, bank_scale: Tensor, offsets: Tensor
) -> Tensor:
    return tokens.new_empty(index.shape[0], bank_scale.shape[2], dtype=torch.bfloat16)


def _float8_gate_up_eager(
    tokens: Tensor, tokens_scale: Tensor, index: Tensor, bank: Tensor, bank_scale: Tensor, offsets: Tensor
) -> Tensor:
    gate_up = praxis_layers._float8_grouped_mm_pytorch(
        tokens[index], tokens_scale[index], bank, bank_scale.flatten(1), offsets, torch.bfloat16
    )
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def _has_tcgen05() -> bool:
    major, minor = torch.cuda.get_device_capability()
    return 100 <= major * 10 + minor < 120


def _register() -> praxis_layers.Float8GateUp | None:
    triton_op = _triton_op()
    if triton_op is None or not torch.cuda.is_available() or not _has_tcgen05():
        return None
    import triton
    import triton.language as tl

    @triton.jit
    def float8_gate_up_kernel(
        tokens,
        tokens_scale,
        index,
        bank,
        bank_scale,
        offsets,
        m,
        h,
        k,
        num_experts,
        hidden,
        MAX_EXPERTS: tl.constexpr,  # noqa: N803
        BLOCK_M: tl.constexpr,  # noqa: N803
        BLOCK_N: tl.constexpr,  # noqa: N803
        BLOCK_K: tl.constexpr,  # noqa: N803
    ):
        # h and k are multiples of 16 (`float8_gate_up` checks). Written so, the compiler knows it, and pipelines and
        # widens the loads as it does for constant sizes.
        h = (h // 16) * 16
        k = (k // 16) * 16
        pid = tl.program_id(0)
        tiles_n = tl.cdiv(h, BLOCK_N)
        experts = tl.arange(0, MAX_EXPERTS)
        valid = experts < num_experts
        starts = tl.load(offsets + experts - 1, mask=valid & (experts > 0), other=0)
        ends = tl.where(valid, tl.load(offsets + experts, mask=valid, other=0), starts)
        expert_tiles = (ends - starts + BLOCK_M - 1) // BLOCK_M
        tiles_through = tl.cumsum(expert_tiles, axis=0)
        total_tiles = tl.sum(expert_tiles, axis=0)
        if pid >= total_tiles * tiles_n:
            # Rows past the last offset belong to no expert: zero them.
            past = pid - total_tiles * tiles_n
            r = tl.load(offsets + num_experts - 1) + (past // tiles_n) * BLOCK_M + tl.arange(0, BLOCK_M)
            cols = (past % tiles_n) * BLOCK_N + tl.arange(0, BLOCK_N)
            tl.store(
                hidden + r.to(tl.int64)[:, None] * h + cols[None, :],
                tl.zeros((BLOCK_M, BLOCK_N), dtype=hidden.dtype.element_ty),
                mask=((r >= 0) & (r < m))[:, None] & (cols < h)[None, :],
            )
            return
        # The row tiles of an expert vary fastest, so the programs that run together read the same weight columns,
        # each from memory once.
        expert = tl.sum((tiles_through * tiles_n <= pid).to(tl.int32), axis=0)
        selected = experts == expert
        count = tl.sum(tl.where(selected, expert_tiles, 0), axis=0)
        local = pid - (tl.sum(tl.where(selected, tiles_through, 0), axis=0) - count) * tiles_n
        r = tl.sum(tl.where(selected, starts, 0), axis=0) + (local % count) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = (local // count) * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = (r >= 0) & (r < tl.minimum(tl.sum(tl.where(selected, ends, 0), axis=0), m))
        col_mask = cols < h
        token = tl.load(index + r, mask=row_mask, other=0)
        ks = tl.arange(0, BLOCK_K)
        a = tokens + token[:, None] * k + ks[None, :]
        # The bank holds each expert's gate columns, then its up columns, each column's k values together:
        # (experts, 2 * h, k).
        b = bank + expert.to(tl.int64) * 2 * h * k + cols.to(tl.int64)[None, :] * k + ks[:, None]
        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for step in range(0, tl.cdiv(k, BLOCK_K)):
            k_mask = ks < k - step * BLOCK_K
            a_tile = tl.load(a, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            b_mask = k_mask[:, None] & col_mask[None, :]
            acc_gate = tl.dot(a_tile, tl.load(b, mask=b_mask, other=0.0), acc_gate)
            acc_up = tl.dot(a_tile, tl.load(b + h * k, mask=b_mask, other=0.0), acc_up)
            a += BLOCK_K
            b += BLOCK_K
        a_scale = tl.load(tokens_scale + token, mask=row_mask, other=0.0)
        scales = bank_scale + expert.to(tl.int64) * 2 * h + cols
        gate_scale = tl.load(scales, mask=col_mask, other=0.0)
        up_scale = tl.load(scales + h, mask=col_mask, other=0.0)
        gate = (acc_gate * a_scale[:, None] * gate_scale[None, :]).to(tl.bfloat16).to(tl.float32)
        up = (acc_up * a_scale[:, None] * up_scale[None, :]).to(tl.bfloat16).to(tl.float32)
        silu = tl.math.div_rn(gate, 1.0 + tl.exp(-gate)).to(tl.bfloat16).to(tl.float32)
        tl.store(
            hidden + r.to(tl.int64)[:, None] * h + cols[None, :],
            (silu * up).to(hidden.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )

    signature = {
        "tokens": "*fp8e4nv",
        "tokens_scale": "*fp32",
        "index": "*i64",
        "bank": "*fp8e4nv",
        "bank_scale": "*fp32",
        "offsets": "*i32",
        "m": "i32",
        "h": "i32",
        "k": "i32",
        "num_experts": "i32",
        "hidden": "*bf16",
    }

    def float8_gate_up(
        tokens: Tensor, tokens_scale: Tensor, index: Tensor, bank: Tensor, bank_scale: Tensor, offsets: Tensor
    ) -> Tensor:
        experts, n, k = bank.shape
        if tokens.shape[1] != k or n % (2 * ALIGNMENT) or k % ALIGNMENT or experts > MAX_EXPERTS:
            raise ValueError(
                f"The float8 gate and up plugin takes rows of K values and at most {MAX_EXPERTS} experts of 2H x K "
                f"weights, K and H multiples of {ALIGNMENT}, not rows of {tokens.shape[1]} values and {experts} "
                f"experts of {n} x {k}."
            )
        return torch.ops.lerobot_praxis.float8_gate_up(
            tokens, tokens_scale, index, bank, bank_scale.unflatten(1, (2, n // 2)), offsets
        )

    try:
        triton_op(
            "lerobot_praxis::float8_gate_up",
            float8_gate_up_kernel,
            signature,
            {"MAX_EXPERTS": MAX_EXPERTS, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
            # Each expert and the rows past the last offset can end in a partial row tile.
            grid=lambda inputs, outputs: (
                ((inputs[2].shape_expr[0] + BLOCK_M - 1) // BLOCK_M + inputs[3].shape_expr[0])
                * ((inputs[4].shape_expr[2] + BLOCK_N - 1) // BLOCK_N),
            ),
            meta_fn=_float8_gate_up_meta,
            extra_args_fn=lambda inputs, outputs: [
                inputs[2].shape_expr[0],
                inputs[4].shape_expr[2],
                inputs[3].shape_expr[2],
                inputs[3].shape_expr[0],
            ],
            eager_fn=_float8_gate_up_eager,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    except (ValueError, RuntimeError) as error:
        # Such as more shared memory than TensorRT launches a plugin kernel with, or a launch feature the
        # plugin path cannot reproduce on this GPU.
        logger.warning("The float8 gate and up plugin does not build for this GPU: %s", error)
        return None
    return float8_gate_up


FLOAT8_GATE_UP = _register()
