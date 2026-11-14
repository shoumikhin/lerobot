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

"""The first half of Praxis-1's routed experts as one TensorRT plugin, for its export.

Each mixture-of-experts layer gathers the input row of every (token, expert) pair in expert order, multiplies it by
its expert's packed gate and up weights, and applies SwiGLU, `silu(gate) * up`
(`praxis_core.layers._routed_chunk_group`). As three operations, TensorRT writes the gathered rows and both products
to memory and reads them back: for Praxis-1 about 3 times as many bytes as the result. The plugin is one Triton
kernel that reads each row of `x` through the index, multiplies it by the gate and the up columns of its expert on
`tcgen05` tensor cores and writes only `silu(gate) * up`. It rounds as the operator's PyTorch side does: both
products to bfloat16, then the SiLU, then the product. Its results differ from that side only in the order of the
products' float32 sums and in the last bit of the SiLU's exponential.

The kernel needs `tcgen05` (sm_100 and sm_110), which keeps both products in tensor memory. Other tensor cores keep
them in registers, which leaves room for only half as many rows per tile, and there the three operations are faster.

Importing this module registers the operator `lerobot_praxis::expert_gate_up` if the GPU has `tcgen05` and the
installed Torch-TensorRT makes TensorRT plugins of Triton kernels, lets them use tensor memory and builds this one.
`EXPERT_GATE_UP` is then a `praxis_core.layers.ExpertGateUp` that calls it, and None otherwise. A part that traces
it names this module in its `operator_modules`.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from .export_top_k_sum import _triton_op
from .praxis_core import layers as praxis_layers

logger = logging.getLogger(__name__)

# Each program computes the same columns of gate and of up, in two accumulators in tensor memory. A 16 deep tile fits
# 3 pipeline stages in the 48 KiB of shared memory TensorRT gives a plugin kernel. With 4 warps, reading both
# accumulators for the SwiGLU spills a few registers once per tile, and still runs faster on Thor than 8 warps.
BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES = 128, 128, 16, 4, 3
# The most experts the kernel finds a row tile's expert among, a power of two.
MAX_EXPERTS = 128
# The kernel reads 16 bfloat16 values at a time along K and along the hidden columns.
ALIGNMENT = 16


# The operator takes the bank as (EXPERTS, K, 2, H): Torch-TensorRT derives the plugin's output shape from its inputs'
# dimensions only by sums and products.
def _expert_gate_up_meta(x: Tensor, index: Tensor, bank: Tensor, offsets: Tensor) -> Tensor:
    return x.new_empty(index.shape[0], bank.shape[3])


def _expert_gate_up_eager(x: Tensor, index: Tensor, bank: Tensor, offsets: Tensor) -> Tensor:
    gate, up = torch._grouped_mm(x[index], bank.flatten(2), offsets).chunk(2, dim=-1)
    return F.silu(gate) * up


def _has_tcgen05() -> bool:
    major, minor = torch.cuda.get_device_capability()
    return 100 <= major * 10 + minor < 120


def _register() -> praxis_layers.ExpertGateUp | None:
    triton_op = _triton_op()
    if triton_op is None or not torch.cuda.is_available() or not _has_tcgen05():
        return None
    import triton
    import triton.language as tl

    @triton.jit
    def expert_gate_up_kernel(
        x,
        index,
        bank,
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
        # h and k are multiples of 16 (`expert_gate_up` checks). Written so, the compiler knows it, and pipelines and
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
        # each from memory once. Where memory bandwidth limits the kernel, as on Thor, that is faster than varying
        # the columns fastest.
        expert = tl.sum((tiles_through * tiles_n <= pid).to(tl.int32), axis=0)
        selected = experts == expert
        count = tl.sum(tl.where(selected, expert_tiles, 0), axis=0)
        local = pid - (tl.sum(tl.where(selected, tiles_through, 0), axis=0) - count) * tiles_n
        r = tl.sum(tl.where(selected, starts, 0), axis=0) + (local % count) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = (local // count) * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = (r >= 0) & (r < tl.minimum(tl.sum(tl.where(selected, ends, 0), axis=0), m))
        col_mask = cols < h
        tokens = tl.load(index + r, mask=row_mask, other=0)
        ks = tl.arange(0, BLOCK_K)
        a = x + tokens[:, None] * k + ks[None, :]
        # The bank holds each expert's gate columns, then its up columns: (experts, k, 2, h).
        b = bank + expert.to(tl.int64) * k * 2 * h + ks.to(tl.int64)[:, None] * 2 * h + cols[None, :]
        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for step in range(0, tl.cdiv(k, BLOCK_K)):
            k_mask = ks < k - step * BLOCK_K
            a_tile = tl.load(a, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            b_mask = k_mask[:, None] & col_mask[None, :]
            acc_gate = tl.dot(a_tile, tl.load(b, mask=b_mask, other=0.0), acc_gate)
            acc_up = tl.dot(a_tile, tl.load(b + h, mask=b_mask, other=0.0), acc_up)
            a += BLOCK_K
            b += BLOCK_K * 2 * h
        gate = acc_gate.to(tl.bfloat16).to(tl.float32)
        up = acc_up.to(tl.bfloat16).to(tl.float32)
        silu = tl.math.div_rn(gate, 1.0 + tl.exp(-gate)).to(tl.bfloat16).to(tl.float32)
        tl.store(
            hidden + r.to(tl.int64)[:, None] * h + cols[None, :],
            (silu * up).to(hidden.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )

    signature = {
        "x": "*bf16",
        "index": "*i64",
        "bank": "*bf16",
        "offsets": "*i32",
        "m": "i32",
        "h": "i32",
        "k": "i32",
        "num_experts": "i32",
        "hidden": "*bf16",
    }

    def expert_gate_up(x: Tensor, index: Tensor, bank: Tensor, offsets: Tensor) -> Tensor:
        experts, k, n = bank.shape
        if x.shape[1] != k or n % (2 * ALIGNMENT) or k % ALIGNMENT or experts > MAX_EXPERTS:
            raise ValueError(
                f"The expert gate and up plugin takes rows of K values and at most {MAX_EXPERTS} experts of K x 2H "
                f"weights, K and H multiples of {ALIGNMENT}, not rows of {x.shape[1]} values and {experts} experts "
                f"of {k} x {n}."
            )
        return torch.ops.lerobot_praxis.expert_gate_up(x, index, bank.unflatten(2, (2, n // 2)), offsets)

    try:
        triton_op(
            "lerobot_praxis::expert_gate_up",
            expert_gate_up_kernel,
            signature,
            {"MAX_EXPERTS": MAX_EXPERTS, "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K},
            # Each expert and the rows past the last offset can end in a partial row tile.
            grid=lambda inputs, outputs: (
                ((inputs[1].shape_expr[0] + BLOCK_M - 1) // BLOCK_M + inputs[2].shape_expr[0])
                * ((inputs[2].shape_expr[3] + BLOCK_N - 1) // BLOCK_N),
            ),
            meta_fn=_expert_gate_up_meta,
            extra_args_fn=lambda inputs, outputs: [
                inputs[1].shape_expr[0],
                inputs[2].shape_expr[3],
                inputs[2].shape_expr[1],
                inputs[2].shape_expr[0],
            ],
            eager_fn=_expert_gate_up_eager,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    except (ValueError, RuntimeError) as error:
        # Such as more shared memory than TensorRT launches a plugin kernel with, or a launch feature the
        # plugin path cannot reproduce on this GPU.
        logger.warning("The expert gate and up plugin does not build for this GPU: %s", error)
        return None
    return expert_gate_up


EXPERT_GATE_UP = _register()
