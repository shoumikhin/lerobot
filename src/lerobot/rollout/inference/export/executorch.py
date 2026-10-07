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

"""Run an exported policy's ExecuTorch program, on its TensorRT or CUDA backend."""

from pathlib import Path

import numpy as np


class ExecuTorchProgram:
    """Run a `.pte` program with host arrays and read its outputs through DLPack."""

    def __init__(self, path: Path, tensorrt: bool = True):
        if tensorrt:
            import torch_tensorrt_executorch_runtime  # noqa: F401
        from executorch.runtime import Runtime

        data_path = next(path.parent.glob("*.ptd"), None)
        self._method = Runtime.get().load_program(path, data_path=data_path).load_method("forward")
        self._torch_inputs = False

    @property
    def input_shape(self) -> tuple[int, ...]:
        return tuple(self._method.metadata.input_tensor_meta(0).sizes())

    def initialize(self, *inputs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        """Select the input type while running the saved test case at load time."""
        try:
            return self(*inputs)
        except (TypeError, RuntimeError) as error:
            if "Unsupported python type <class 'numpy.ndarray'>" not in str(error):
                raise
        self._torch_inputs = True
        return self(*inputs)

    def __call__(self, *inputs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        arrays = [np.ascontiguousarray(value) for value in inputs]
        if self._torch_inputs:
            import torch

            arrays = [
                torch.from_numpy(value.view(np.uint16)).view(torch.bfloat16)
                if value.dtype.name == "bfloat16"
                else torch.from_numpy(value)
                for value in arrays
            ]
        outputs = self._method.execute(arrays)
        result = []
        for output in outputs:
            if output.__dlpack_device__()[0] == 1:
                try:
                    result.append(np.from_dlpack(output).copy())
                    continue
                except (BufferError, RuntimeError):
                    # NumPy cannot consume every DLPack dtype, including bfloat16.
                    pass
            import torch

            tensor = torch.from_dlpack(output).cpu()
            if tensor.dtype == torch.bfloat16:
                from ml_dtypes import bfloat16

                result.append(tensor.view(torch.uint16).numpy().view(bfloat16).copy())
            else:
                result.append(tensor.numpy().copy())
        return result[0] if len(result) == 1 else tuple(result)


class ExecuTorchDenoisingLoop:
    """Run every program before the step once, integrate the velocity, and convert the final actions.

    The programs run in the order `export.json` lists them, and each one reads the values its `inputs`
    name: the frame's arrays, or the `outputs` of a program before it. So a prefix can be one program, or
    a chain of smaller ones that each hold only part of the policy's weights.
    """

    def __init__(self, folder: Path, programs: dict[str, dict], num_steps: int):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        self._programs = {
            name: ExecuTorchProgram(folder / program["file"]) for name, program in programs.items()
        }
        self._inputs = {name: program["inputs"] for name, program in programs.items()}
        self._outputs = {name: program["outputs"] for name, program in programs.items()}
        self._num_steps = num_steps

    @property
    def input_shape(self) -> tuple[int, ...]:
        return next(iter(self._programs.values())).input_shape

    def initialize(self, *inputs: np.ndarray) -> np.ndarray:
        return self._run(inputs, initialize=True)

    def __call__(self, *inputs: np.ndarray) -> np.ndarray:
        return self._run(inputs)

    def _run(self, inputs: tuple[np.ndarray, ...], initialize: bool = False) -> np.ndarray:
        *observation, noise = inputs
        values = dict(zip(next(iter(self._inputs.values())), observation, strict=True))
        for name, program in self._programs.items():
            if name in ("step", "actions"):
                continue
            outputs = (program.initialize if initialize else program)(
                *(values[n] for n in self._inputs[name])
            )
            values.update(
                zip(self._outputs[name], outputs if isinstance(outputs, tuple) else (outputs,), strict=True)
            )
        # The step's last two inputs are the noisy actions and the time; the others come from the prefix.
        cache = [values[name] for name in self._inputs["step"][:-2]]
        dt = -1.0 / self._num_steps
        sample = noise
        for step in range(self._num_steps):
            timestep = np.array([1.0 + step * dt], dtype=np.float32)
            run_step = (
                self._programs["step"].initialize if initialize and step == 0 else self._programs["step"]
            )
            sample = sample + dt * run_step(*cache, sample, timestep)
        actions = self._programs["actions"].initialize if initialize else self._programs["actions"]
        return actions(sample)
