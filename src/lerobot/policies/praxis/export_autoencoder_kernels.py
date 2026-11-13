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

"""The norm and SiLU steps and the patch averages of Praxis-1's autoencoder as TensorRT plugins, for its export.

The encoder normalizes each convolution output over its channels (`RMSNorm`), then applies SiLU. Written in
PyTorch, TensorRT runs each such step as about five kernels: two casts to float32, a reduction over the channel
axis, the norm and SiLU, and a cast back, each of which also changes the layout. The patch averages of the
encoder's shortcuts (`AvgDown3D`) copy the clip into a patch-major layout first. Together these steps can take
longer than the encoder's convolutions. As plugins each step is one kernel, which reads the clip in TensorRT's
plain layout. TensorRT then copies to and from the layout of its convolutions around each plugin, since a plugin
made from a Triton kernel reads and writes only the plain layout, and the steps are still faster in all.

Importing this module registers the operators `lerobot_praxis::autoencoder_norm_silu` and
`lerobot_praxis::autoencoder_avg_down` if CUDA is available and the installed Torch-TensorRT makes TensorRT
plugins of Triton kernels (`torch_tensorrt.kernels.triton_op`). `AUTOENCODER_KERNELS` is then the
`vae.AutoencoderKernels` of them, and None otherwise. A part that traces them names this module in its
`operator_modules`, so that the compile step registers the operators before it loads the program.

The operators' PyTorch sides compute what the encoder's modules compute, so the export's checks run the same
values. The encoder's convolutions run in float16 (its autocast), so its activations, and the operators'
inputs and outputs, are float16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from .praxis_core import vae

# The channels and positions of the norm kernel's tile, the fastest of those measured on one GPU.
NORM_BLOCK_C, NORM_BLOCK_S = 32, 64
# The outputs that one program of the averaging kernel writes.
AVG_BLOCK = 512


def norm_silu_reference(x: Tensor, gamma: Tensor) -> Tensor:
    """``silu(RMSNorm(x))`` of a float16 ``(B, C, T, H, W)`` clip, for the norm's float32 ``(C,)`` ``gamma``.

    The norm plugin's PyTorch side. It computes in float32, as the encoder does under its autocast, and rounds to
    float16, the dtype the convolution after it reads.
    """
    y = F.normalize(x.float(), dim=1) * x.shape[1] ** 0.5 * gamma.view(-1, 1, 1, 1) + 0.0
    return F.silu(y).to(torch.float16)


def avg_down_reference(patches: Tensor) -> Tensor:
    """``AvgDown3D`` with 2x2 spatial patches, from ``(B, C, T/ft, ft, H/2, 2, W/2, 2)``, a view of the
    ``(B, C, T, H, W)`` clip, to ``(B, C*ft, T/ft, H/2, W/2)``. The averaging plugin's PyTorch side.

    The plugin takes the view, so its output shape is a product of its input's: Torch-TensorRT cannot pass a
    division of shapes to TensorRT.
    """
    b, c, t, ft, h, _, w, _ = patches.shape
    return patches.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(b, c * ft, 4, t, h, w).mean(dim=2)


def _triton_op():
    try:
        import torch_tensorrt
    except ImportError:
        return None
    return getattr(getattr(torch_tensorrt, "kernels", None), "triton_op", None)


def _register() -> vae.AutoencoderKernels | None:
    triton_op = _triton_op()
    # The operators' PyTorch sides run only on CUDA.
    if triton_op is None or not torch.cuda.is_available():
        return None
    import triton
    import triton.language as tl

    @triton.jit
    def norm_silu_kernel(x, gamma, positions, channels, out, BLOCK_C: tl.constexpr, BLOCK_S: tl.constexpr):  # noqa: N803
        # One program normalizes BLOCK_S positions of one batch row over all channels: in the plain layout each
        # channel's positions are contiguous.
        row = tl.program_id(1).to(tl.int64) * channels * positions
        position = tl.program_id(0) * BLOCK_S + tl.arange(0, BLOCK_S)
        inside = position < positions
        squares = tl.zeros([BLOCK_S], tl.float32)
        for first in range(0, channels, BLOCK_C):
            channel = first + tl.arange(0, BLOCK_C)
            mask = (channel[:, None] < channels) & inside[None, :]
            offset = row + channel[:, None].to(tl.int64) * positions + position[None, :]
            value = tl.load(x + offset, mask=mask, other=0.0).to(tl.float32)
            squares += tl.sum(value * value, axis=0)
        norm = tl.maximum(tl.sqrt_rn(squares), 1e-12)
        scale = tl.sqrt_rn(channels.to(tl.float32))
        for first in range(0, channels, BLOCK_C):
            channel = first + tl.arange(0, BLOCK_C)
            mask = (channel[:, None] < channels) & inside[None, :]
            offset = row + channel[:, None].to(tl.int64) * positions + position[None, :]
            value = tl.load(x + offset, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(gamma + channel, mask=channel < channels, other=0.0)
            y = tl.div_rn(value, norm[None, :]) * scale * g[:, None] + 0.0
            tl.store(out + offset, (y * tl.sigmoid(y)).to(tl.float16), mask=mask)

    def norm_silu_meta(x: Tensor, gamma: Tensor) -> Tensor:
        return torch.empty_like(x)

    def clip_positions(clip):
        return clip.shape_expr[2] * clip.shape_expr[3] * clip.shape_expr[4]

    triton_op(
        "lerobot_praxis::autoencoder_norm_silu",
        norm_silu_kernel,
        {"x": "*fp16", "gamma": "*fp32", "positions": "i32", "channels": "i32", "out": "*fp16"},
        {"BLOCK_C": NORM_BLOCK_C, "BLOCK_S": NORM_BLOCK_S},
        grid=lambda inputs, outputs: (
            (clip_positions(inputs[0]) + NORM_BLOCK_S - 1) // NORM_BLOCK_S,
            inputs[0].shape_expr[0],
        ),
        meta_fn=norm_silu_meta,
        extra_args_fn=lambda inputs, outputs: [clip_positions(inputs[0]), inputs[0].shape_expr[1]],
        eager_fn=norm_silu_reference,
        num_warps=8,
        num_stages=2,
    )

    @triton.jit
    def avg_down_kernel(x, size, frames, height, width, taps, out, BLOCK: tl.constexpr):  # noqa: N803
        # One output element per lane, at ((((b * C + c) * taps + tap) * frames + t) * height + h) * width + w.
        index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        inside = index < size
        w = index % width
        rest = index // width
        h = rest % height
        rest = rest // height
        t = rest % frames
        rest = rest // frames
        tap = rest % taps
        channel = rest // taps
        # The input (B, C, T/ft, ft, H/2, 2, W/2, 2): the patch's 2 columns are contiguous, its 2 rows a row apart.
        first = (((channel * frames + t) * taps + tap) * 2 * height + 2 * h) * 2 * width + 2 * w
        total = tl.load(x + first, mask=inside, other=0.0).to(tl.float32)
        total += tl.load(x + first + 1, mask=inside, other=0.0).to(tl.float32)
        total += tl.load(x + first + 2 * width, mask=inside, other=0.0).to(tl.float32)
        total += tl.load(x + first + 2 * width + 1, mask=inside, other=0.0).to(tl.float32)
        tl.store(out + index, (total * 0.25).to(tl.float16), mask=inside)

    def avg_down_meta(patches: Tensor) -> Tensor:
        b, c, t, ft, h, _, w, _ = patches.shape
        return patches.new_empty(b, c * ft, t, h, w)

    def outputs_of(patches):
        shape = patches.shape_expr
        return shape[0] * shape[1] * shape[2] * shape[3] * shape[4] * shape[6]

    triton_op(
        "lerobot_praxis::autoencoder_avg_down",
        avg_down_kernel,
        {
            "x": "*fp16",
            "size": "i32",
            "frames": "i32",
            "height": "i32",
            "width": "i32",
            "taps": "i32",
            "out": "*fp16",
        },
        {"BLOCK": AVG_BLOCK},
        grid=lambda inputs, outputs: ((outputs_of(inputs[0]) + AVG_BLOCK - 1) // AVG_BLOCK,),
        meta_fn=avg_down_meta,
        extra_args_fn=lambda inputs, outputs: [
            outputs_of(inputs[0]),
            inputs[0].shape_expr[2],
            inputs[0].shape_expr[4],
            inputs[0].shape_expr[6],
            inputs[0].shape_expr[3],
        ],
        eager_fn=avg_down_reference,
        num_warps=4,
        num_stages=1,
    )

    def norm_silu(x: Tensor, gamma: Tensor) -> Tensor:
        return torch.ops.lerobot_praxis.autoencoder_norm_silu(x.to(torch.float16), gamma.reshape(-1).float())

    def avg_down(x: Tensor, factor_t: int) -> Tensor:
        b, c, t, h, w = x.shape
        patches = x.to(torch.float16).reshape(b, c, t // factor_t, factor_t, h // 2, 2, w // 2, 2)
        return torch.ops.lerobot_praxis.autoencoder_avg_down(patches)

    return vae.AutoencoderKernels(norm_silu, avg_down)


AUTOENCODER_KERNELS = _register()
