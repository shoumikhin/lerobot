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

    def __call__(self, *inputs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
        arrays = [np.ascontiguousarray(value) for value in inputs]
        if not self._torch_inputs:
            try:
                outputs = self._method.execute(arrays)
            except TypeError:
                # Older bindings accept only torch tensors.
                self._torch_inputs = True
        if self._torch_inputs:
            import torch

            tensors = [
                torch.from_numpy(value.view(np.uint16)).view(torch.bfloat16)
                if value.dtype.name == "bfloat16"
                else torch.from_numpy(value)
                for value in arrays
            ]
            outputs = self._method.execute(tensors)
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
    """Run the prefix once, integrate the velocity, and convert the final actions."""

    def __init__(self, folder: Path, programs: dict[str, dict], num_steps: int):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        self._prefix, self._step, self._actions = (
            ExecuTorchProgram(folder / programs[name]["file"]) for name in ("prefix", "step", "actions")
        )
        self._num_steps = num_steps

    def __call__(self, *inputs: np.ndarray) -> np.ndarray:
        *observation, noise = inputs
        cache = self._prefix(*observation)
        if not isinstance(cache, tuple):
            cache = (cache,)
        dt = -1.0 / self._num_steps
        sample = noise
        for step in range(self._num_steps):
            timestep = np.full((noise.shape[0],), 1.0 + step * dt, dtype=np.float32)
            sample = sample + dt * self._step(*cache, sample, timestep)
        return self._actions(sample)
