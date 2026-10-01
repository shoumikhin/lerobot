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

"""Run an exported policy's TensorRT engine, built from its ONNX file."""

from pathlib import Path

import torch

# TensorRT asks for the whole engine in one read; answering in pieces keeps the host copy to one piece.
READ_CHUNK_BYTES = 64 << 20


def load_engine(path: Path):
    """Deserialize a `.engine` file a piece at a time, without holding the whole file in memory."""
    import tensorrt as trt

    class FileReader(trt.IStreamReaderV2):
        def __init__(self, file):
            trt.IStreamReaderV2.__init__(self)
            self._file = file

        def read(self, num_bytes: int, stream: int) -> bytes:
            return self._file.read(min(num_bytes, READ_CHUNK_BYTES))

        def seek(self, offset: int, where: trt.SeekPosition) -> bool:
            whence = {trt.SeekPosition.SET: 0, trt.SeekPosition.CUR: 1, trt.SeekPosition.END: 2}[where]
            self._file.seek(offset, whence)
            return True

    with path.open("rb") as file:
        return trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(FileReader(file))


class TensorRTEngine:
    """Runs a `.engine` file on GPU tensors, on the current CUDA stream."""

    def __init__(self, path: Path, input_names: list[str], output_name: str):
        engine = load_engine(path)
        self._context = engine.create_execution_context()
        self._input_names = input_names
        self._input_shapes = [tuple(engine.get_tensor_shape(name)) for name in input_names]
        self._output_name = output_name
        self._output = torch.empty(tuple(engine.get_tensor_shape(output_name)), device="cuda")

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        for name, shape, tensor in zip(self._input_names, self._input_shapes, inputs, strict=True):
            # TensorRT reads raw memory, so a frame of another size would be read out of bounds, not refused.
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} has shape {tuple(tensor.shape)}; the engine was built for {shape}.")
            self._context.set_tensor_address(name, tensor.data_ptr())
        self._context.set_tensor_address(self._output_name, self._output.data_ptr())
        if not self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT could not run the engine.")
        return self._output
