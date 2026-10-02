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
from safetensors.torch import load_file

from lerobot.policies.common.flow_matching import euler_integrate

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


def torch_dtype(dtype) -> torch.dtype:
    """The torch dtype of a TensorRT tensor dtype."""
    import tensorrt as trt

    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }[dtype]


class TensorRTEngine:
    """Runs a `.engine` file on GPU tensors, on the current CUDA stream.

    Returns the output, or a tuple of outputs when the engine has several. With `own_scratch=False`, the
    engine runs in the scratch memory given to `use_scratch`, which engines run one after another can share.
    An engine built with `build_engine.py --int8_weights` gets its INT8 weights from the
    `<name>_int8_weights.safetensors` file beside it, loaded to the GPU once.
    """

    def __init__(self, path: Path, input_names: list[str], output_names: list[str], own_scratch: bool = True):
        import tensorrt as trt

        self._engine = load_engine(path)
        strategy = trt.ExecutionContextAllocationStrategy
        self._context = self._engine.create_execution_context(
            strategy.STATIC if own_scratch else strategy.USER_MANAGED
        )
        weights_file = path.with_name(f"{path.stem}_int8_weights.safetensors")
        self._weights = load_file(weights_file, device="cuda") if weights_file.exists() else {}
        for name, weight in self._weights.items():
            self._context.set_tensor_address(name, weight.data_ptr())
        self._input_names = input_names
        self._input_shapes = [tuple(self._engine.get_tensor_shape(name)) for name in input_names]
        self._output_names = output_names
        self._outputs = [
            torch.empty(
                tuple(self._engine.get_tensor_shape(name)),
                dtype=torch_dtype(self._engine.get_tensor_dtype(name)),
                device="cuda",
            )
            for name in output_names
        ]

    @property
    def scratch_bytes(self) -> int:
        return self._engine.device_memory_size_v2

    def use_scratch(self, scratch: torch.Tensor) -> None:
        self._context.set_device_memory(scratch.data_ptr(), self.scratch_bytes)

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        for name, shape, tensor in zip(self._input_names, self._input_shapes, inputs, strict=True):
            # TensorRT reads raw memory, so a frame of another size would be read out of bounds, not refused.
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} has shape {tuple(tensor.shape)}; the engine was built for {shape}.")
            self._context.set_tensor_address(name, tensor.data_ptr())
        for name, output in zip(self._output_names, self._outputs, strict=True):
            self._context.set_tensor_address(name, output.data_ptr())
        if not self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT could not run the engine.")
        return self._outputs[0] if len(self._outputs) == 1 else tuple(self._outputs)


class TensorRTDenoisingLoop:
    """Runs a flow-matching chunk exported as three engines, as the policy's `sample_actions` does.

    The prefix engine turns the observation into the KV cache, the step engine runs once per Euler step of
    LeRobot's own `euler_integrate`, and the actions engine turns the result into the actions to play. The
    engines run one after another, so they share one scratch buffer, and the KV cache stays on the GPU.
    """

    def __init__(self, folder: Path, programs: dict[str, dict], num_steps: int):
        self._prefix, self._step, self._actions = engines = [
            TensorRTEngine(folder / program["file"], program["inputs"], program["outputs"], own_scratch=False)
            for program in (programs["prefix"], programs["step"], programs["actions"])
        ]
        self._scratch = torch.empty(
            max(engine.scratch_bytes for engine in engines), dtype=torch.uint8, device="cuda"
        )
        for engine in engines:
            engine.use_scratch(self._scratch)
        self._num_steps = num_steps

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        *observation, noise = inputs
        cache = self._prefix(*observation)
        x_0 = euler_integrate(lambda x_t, timestep: self._step(*cache, x_t, timestep), noise, self._num_steps)
        return self._actions(x_0)
