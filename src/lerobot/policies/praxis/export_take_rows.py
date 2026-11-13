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

"""The row gather of Praxis-1's mixture-of-experts dispatch as one TensorRT plugin, for its export.

Before its grouped matrix multiplies, each mixture-of-experts layer copies the input row of every (token,
expert) pair into expert order, `x[sorted_tokens]`. TensorRT runs that gather well below the GPU's memory
bandwidth, slower than a kernel that copies a block of columns of one row per program, as the original
implementation's dispatch kernel does. The plugin is that kernel. It copies the rows, so its result has the
bits of `x[index]`, the operator's PyTorch side.

Importing this module registers the operator `lerobot_praxis::take_rows` if CUDA is available and the
installed Torch-TensorRT makes TensorRT plugins of Triton kernels (`torch_tensorrt.kernels.triton_op`).
`TAKE_ROWS` is then the operator, and None otherwise. A part that traces it names this module in its
`operator_modules`, so that the compile step registers the operator before it loads the program.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .export_top_k_sum import _triton_op
from .praxis_core import layers as praxis_layers

# The columns of a row that one program of the kernel copies. Narrower blocks give more independent row
# reads, which hide the latency of the indirect loads.
BLOCK = 512


def _take_rows(x: Tensor, index: Tensor) -> Tensor:
    return x[index]


def _register() -> praxis_layers.TakeRows | None:
    triton_op = _triton_op()
    # The operator's PyTorch side runs only on CUDA.
    if triton_op is None or not torch.cuda.is_available():
        return None
    import triton
    import triton.language as tl

    @triton.jit
    def take_rows_kernel(x, index, width, rows, BLOCK: tl.constexpr):  # noqa: N803
        row = tl.program_id(0).to(tl.int64)
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        inside = columns < width
        values = tl.load(x + tl.load(index + row) * width + columns, mask=inside)
        tl.store(rows + row * width + columns, values, mask=inside)

    def take_rows_meta(x: Tensor, index: Tensor) -> Tensor:
        return x.new_empty(index.shape[0], x.shape[1])

    triton_op(
        "lerobot_praxis::take_rows",
        take_rows_kernel,
        {"x": "*bf16", "index": "*i64", "width": "i32", "rows": "*bf16"},
        {"BLOCK": BLOCK},
        grid=lambda inputs, outputs: (
            inputs[1].shape_expr[0],
            (inputs[0].shape_expr[1] + BLOCK - 1) // BLOCK,
        ),
        meta_fn=take_rows_meta,
        extra_args_fn=lambda inputs, outputs: [inputs[0].shape_expr[1]],
        eager_fn=_take_rows,
        num_warps=4,
        num_stages=1,
    )
    return torch.ops.lerobot_praxis.take_rows


TAKE_ROWS = _register()
