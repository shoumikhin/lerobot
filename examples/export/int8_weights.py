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

"""Store the large matrix weights of an exported ONNX file in INT8, so the engine takes about half the memory.

Each 2-D weight that only feeds MatMul or Gemm layers becomes symmetric per-output-channel INT8 and a
`DequantizeLinear` back to its own dtype, so the layers still compute in that dtype. TensorRT turns an INT8
weight stored in the engine back into bfloat16 at build time, so the INT8 weights become engine inputs instead,
saved in `<name>_int8_weights.safetensors` and bound once when the engine loads.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors.numpy import save_file

# Smaller weights save little memory.
MIN_INT8_ELEMENTS = 1 << 20
NUMPY_DTYPES = {TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16}


def output_channel_axis(graph: onnx.GraphProto, name: str) -> int | None:
    """The axis of output channels of a weight only used by MatMul or Gemm layers, or None."""
    consumers = {}
    for node in graph.node:
        for index, tensor in enumerate(node.input):
            consumers.setdefault(tensor, []).append((node, index))
    axes = set()
    for node, index in consumers.get(name, []):
        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == "MatMul" and index == 1:
            axes.add(1)
        elif node.op_type == "Gemm" and index == 1:
            axes.add(0 if attributes.get("transB", 0) else 1)
        elif (
            node.op_type == "Transpose"
            and list(attributes.get("perm", [1, 0])) == [1, 0]
            and consumers.get(node.output[0])
            and all(c.op_type == "MatMul" and i == 1 for c, i in consumers[node.output[0]])
        ):
            axes.add(0)
        else:
            return None
    return axes.pop() if len(axes) == 1 else None


def read_weight(tensor: onnx.TensorProto, folder: Path) -> np.ndarray:
    """The weight as float32, memory-mapped from its external data file when it has one."""
    import ml_dtypes

    if tensor.data_location != TensorProto.EXTERNAL:
        return numpy_helper.to_array(tensor).astype(np.float32)
    dtype = ml_dtypes.bfloat16 if tensor.data_type == TensorProto.BFLOAT16 else NUMPY_DTYPES[tensor.data_type]
    info = {entry.key: entry.value for entry in tensor.external_data}
    raw = np.memmap(
        folder / info["location"],
        dtype=np.uint8,
        mode="r",
        offset=int(info.get("offset", 0)),
        shape=(int(info["length"]),),
    )
    return raw.view(dtype).reshape(tuple(tensor.dims)).astype(np.float32)


def quantize(weight: np.ndarray, axis: int, scale_dtype) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel INT8 values and the scales that map them back."""
    scale = np.abs(weight).max(axis=1 - axis, keepdims=True) / 127
    scale = np.where(scale == 0, 1, scale).astype(scale_dtype)
    values = np.clip(np.rint(weight / scale.astype(np.float32)), -127, 127).astype(np.int8)
    return values, scale.reshape(-1)


def int8_weights(onnx_path: Path) -> int:
    """Rewrite `onnx_path` in place; return how many weights now come in as INT8 engine inputs."""
    import ml_dtypes

    model = onnx.load(onnx_path, load_external_data=False)
    graph = model.graph
    weights, scales, dequantize = {}, [], []
    kept = []
    for tensor in graph.initializer:
        axis = None
        if (
            len(tensor.dims) == 2
            and np.prod(tensor.dims) >= MIN_INT8_ELEMENTS
            and tensor.data_type in (TensorProto.BFLOAT16, *NUMPY_DTYPES)
        ):
            axis = output_channel_axis(graph, tensor.name)
        if axis is None:
            kept.append(tensor)
            continue
        # The scale has the weight's own dtype, so DequantizeLinear gives back that dtype.
        scale_dtype = (
            ml_dtypes.bfloat16 if tensor.data_type == TensorProto.BFLOAT16 else NUMPY_DTYPES[tensor.data_type]
        )
        values, scale = quantize(read_weight(tensor, onnx_path.parent), axis, scale_dtype)
        name = f"{tensor.name}.int8"
        weights[name] = values
        graph.input.append(helper.make_tensor_value_info(name, TensorProto.INT8, list(values.shape)))
        scale_type = tensor.data_type
        scales.append(
            helper.make_tensor(f"{tensor.name}.scale", scale_type, scale.shape, scale.tobytes(), raw=True)
        )
        dequantize.append(
            helper.make_node("DequantizeLinear", [name, f"{tensor.name}.scale"], [tensor.name], axis=axis)
        )
    if not weights:
        return 0
    # The kept tensors still point into the original data file, so it stays beside the new ONNX file.
    del graph.initializer[:]
    graph.initializer.extend(kept + scales)
    nodes = dequantize + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    save_file(weights, str(onnx_path.with_name(f"{onnx_path.stem}_int8_weights.safetensors")))
    onnx.save(model, onnx_path)
    return len(weights)
