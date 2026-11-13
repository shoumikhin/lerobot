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

"""The float8 rounding of the input rows of Praxis-1's float8 experts as TensorRT plugins, for its export.

With `--quantize=fp8` the routed experts multiply float8 rows (`praxis_core.layers._float8_rows`): each row is
divided by its largest magnitude over 448 and rounded to float8 e4m3. The plugins do that in one pass over each
row, the first one while it gathers the rows in expert order (`x[index]`), which it then writes in one byte
instead of two. Their results have the bits of `_float8_rows`, the operators' PyTorch side.

Importing this module registers the operators `lerobot_praxis::take_float8_rows` and
`lerobot_praxis::float8_rows` if CUDA is available and the installed Torch-TensorRT makes TensorRT plugins of
Triton kernels. `FLOAT8_ROWS` is then a `praxis_core.layers.Float8Rows` that calls them, and None otherwise. A
part that traces them names this module in its `operator_modules`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .export_top_k_sum import _triton_op
from .praxis_core import layers as praxis_layers

# The columns of a row that one step of a program reads.
BLOCK = 1024


def _float8_rows_meta(x: Tensor) -> tuple[Tensor, Tensor]:
    return x.new_empty(x.shape, dtype=torch.float8_e4m3fn), x.new_empty(x.shape[0], dtype=torch.float32)


def _take_float8_rows_meta(x: Tensor, index: Tensor) -> tuple[Tensor, Tensor]:
    return _float8_rows_meta(x.new_empty(index.shape[0], x.shape[1]))


def _register() -> praxis_layers.Float8Rows | None:
    triton_op = _triton_op()
    if triton_op is None or not torch.cuda.is_available():
        return None
    import triton
    import triton.language as tl

    @triton.jit
    def round_row(source, row, width, rows, scales, BLOCK: tl.constexpr):  # noqa: N803
        # Correctly rounded divisions, as PyTorch's, so that every value rounds to the same float8. The floor
        # of the largest magnitude is float32's smallest normal number, torch.finfo(torch.float32).tiny.
        largest = 0.0
        for start in range(0, width, BLOCK):
            columns = start + tl.arange(0, BLOCK)
            values = tl.load(source + columns, mask=columns < width, other=0.0).to(tl.float32)
            largest = tl.maximum(largest, tl.max(tl.abs(values), axis=0))
        scale = tl.math.div_rn(tl.maximum(largest, 1.1754943508222875e-38), 448.0)
        for start in range(0, width, BLOCK):
            columns = start + tl.arange(0, BLOCK)
            inside = columns < width
            values = tl.load(source + columns, mask=inside, other=0.0).to(tl.float32)
            tl.store(
                rows + row * width + columns, tl.math.div_rn(values, scale).to(tl.float8e4nv), mask=inside
            )
        tl.store(scales + row, scale)

    @triton.jit
    def take_float8_rows_kernel(x, index, width, rows, scales, BLOCK: tl.constexpr):  # noqa: N803
        row = tl.program_id(0).to(tl.int64)
        round_row(x + tl.load(index + row) * width, row, width, rows, scales, BLOCK)

    @triton.jit
    def float8_rows_kernel(x, width, rows, scales, BLOCK: tl.constexpr):  # noqa: N803
        row = tl.program_id(0).to(tl.int64)
        round_row(x + row * width, row, width, rows, scales, BLOCK)

    common = {"BLOCK": BLOCK}
    triton_op(
        "lerobot_praxis::take_float8_rows",
        take_float8_rows_kernel,
        {"x": "*bf16", "index": "*i64", "width": "i32", "rows": "*fp8e4nv", "scales": "*fp32"},
        common,
        grid=lambda inputs, outputs: (inputs[1].shape_expr[0],),
        meta_fn=_take_float8_rows_meta,
        extra_args_fn=lambda inputs, outputs: [inputs[0].shape_expr[1]],
        eager_fn=lambda x, index: praxis_layers._float8_rows(x, index),
        num_warps=4,
        num_stages=1,
    )
    triton_op(
        "lerobot_praxis::float8_rows",
        float8_rows_kernel,
        {"x": "*bf16", "width": "i32", "rows": "*fp8e4nv", "scales": "*fp32"},
        common,
        grid=lambda inputs, outputs: (inputs[0].shape_expr[0],),
        meta_fn=_float8_rows_meta,
        extra_args_fn=lambda inputs, outputs: [inputs[0].shape_expr[1]],
        eager_fn=lambda x: praxis_layers._float8_rows(x),
        num_warps=4,
        num_stages=1,
    )

    def float8_rows(x: Tensor, index: Tensor | None) -> tuple[Tensor, Tensor]:
        if index is None:
            return torch.ops.lerobot_praxis.float8_rows(x)
        return torch.ops.lerobot_praxis.take_float8_rows(x, index)

    return float8_rows


FLOAT8_ROWS = _register()
