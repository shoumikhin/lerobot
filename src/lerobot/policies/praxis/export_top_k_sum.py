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

"""The top-k sum of Praxis-1's mixture-of-experts layers as one TensorRT plugin, for its export.

Written in PyTorch (`_sum_rows_of_tokens`), the sum gathers each token's top-k expert rows, and TensorRT fuses it
with the next block's residual add and RMSNorm into one kernel per token, which is slower than the sum and the
norm as separate kernels. The plugin is the sum alone. It adds the scaled rows in float32 in the order of
`_sum_rows_of_tokens`, the operator's PyTorch side, but Triton may fuse a multiply and an add, so it can round
differently. The export's checks compare within tolerances.

Importing this module registers the operator `lerobot_praxis::top_k_sum` if CUDA is available and the installed
Torch-TensorRT makes TensorRT plugins of Triton kernels (`torch_tensorrt.kernels.triton_op`). `TOP_K_SUM` is then
the operator, and None otherwise. A part that traces it names this module in its `operator_modules`, so that the
compile step registers the operator before it loads the program.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .praxis_core import layers as praxis_layers

# The columns of the sum that one program of the kernel adds.
BLOCK = 1024


def _triton_op():
    try:
        import torch_tensorrt
    except ImportError:
        return None
    return getattr(getattr(torch_tensorrt, "kernels", None), "triton_op", None)


def _register() -> praxis_layers.TopKSum | None:
    triton_op = _triton_op()
    # The operator's PyTorch side runs only on CUDA.
    if triton_op is None or not torch.cuda.is_available():
        return None
    import triton
    import triton.language as tl

    @triton.jit
    def add_slot(acc, rows, rows_of_token, scale, token, slot, top_k, width, columns, inside):
        valid = slot < top_k
        row = tl.load(rows_of_token + token * top_k + slot, mask=valid, other=0)
        term = tl.load(rows + row * width + columns, mask=inside & valid, other=0.0).to(tl.float32)
        term = term * tl.load(scale + token * top_k + slot, mask=valid, other=0.0)
        return tl.where(valid, acc + term, acc)

    @triton.jit
    def top_k_sum_kernel(rows, rows_of_token, scale, width, top_k, out, BLOCK: tl.constexpr):  # noqa: N803
        token = tl.program_id(0).to(tl.int64)
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        inside = columns < width
        acc0 = tl.zeros([BLOCK], tl.float32)
        acc1 = tl.zeros([BLOCK], tl.float32)
        acc2 = tl.zeros([BLOCK], tl.float32)
        acc3 = tl.zeros([BLOCK], tl.float32)
        # Slot s goes to accumulator s % 4, as in `_sum_rows_of_tokens`.
        for first in range(0, top_k, 4):
            acc0 = add_slot(acc0, rows, rows_of_token, scale, token, first, top_k, width, columns, inside)
            acc1 = add_slot(acc1, rows, rows_of_token, scale, token, first + 1, top_k, width, columns, inside)
            acc2 = add_slot(acc2, rows, rows_of_token, scale, token, first + 2, top_k, width, columns, inside)
            acc3 = add_slot(acc3, rows, rows_of_token, scale, token, first + 3, top_k, width, columns, inside)
        tl.store(out + token * width + columns, ((acc0 + acc1) + acc2) + acc3, mask=inside)

    def top_k_sum_meta(rows: Tensor, rows_of_token: Tensor, scale: Tensor) -> Tensor:
        return rows.new_empty(rows_of_token.shape[0], rows.shape[1], dtype=torch.float32)

    triton_op(
        "lerobot_praxis::top_k_sum",
        top_k_sum_kernel,
        {
            "rows": "*bf16",
            "rows_of_token": "*i64",
            "scale": "*fp32",
            "width": "i32",
            "top_k": "i32",
            "out": "*fp32",
        },
        {"BLOCK": BLOCK},
        grid=lambda inputs, outputs: (
            inputs[1].shape_expr[0],
            (inputs[0].shape_expr[1] + BLOCK - 1) // BLOCK,
        ),
        meta_fn=top_k_sum_meta,
        extra_args_fn=lambda inputs, outputs: [inputs[0].shape_expr[1], inputs[1].shape_expr[1]],
        eager_fn=praxis_layers._sum_rows_of_tokens,
        num_warps=4,
        num_stages=1,
    )
    return torch.ops.lerobot_praxis.top_k_sum


TOP_K_SUM = _register()
