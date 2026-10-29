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


"""Run an ExecuTorch program (`.pte`) of a bundle."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def prepare() -> None:
    """Opt in to ExecuTorch's own result type, so running a program never loads PyTorch.

    ExecuTorch returns torch tensors by default. It reads this setting once, when its bindings load.
    """
    os.environ["EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT"] = "executorch"


class ExecuTorchProgram:
    """The `forward` method of a `.pte` program.

    A program places each of its inputs on the CPU or on a GPU, as its backend compiled it. NumPy inputs
    are copied where the program reads them, and outputs stay where the program writes them.

    Args:
        path (`Path`):
            The program file.
        data (`Path`, *optional*):
            The file that holds the program's weights, for a backend that keeps them apart.
    """

    def __init__(self, path: Path, data: Path | None = None):
        """Load the program at `path`."""
        try:
            from executorch.runtime import ExecuTorchResult, Runtime
        except ImportError as error:
            raise ImportError(
                "Running this bundle needs ExecuTorch 1.6: pip install 'executorch>=1.6.0,<1.7.0'"
            ) from error

        self._from_host = ExecuTorchResult.from_host
        try:
            self._method = Runtime.get().load_program(path, data_path=data).load_method("forward")
        except RuntimeError as error:
            raise RuntimeError(f"ExecuTorch could not load {path.name}: {error}") from error
        meta = self._method.metadata
        self._devices = [meta.input_tensor_meta(i).device() for i in range(meta.num_inputs())]

    def __call__(self, *inputs: Any) -> tuple[Any, ...]:
        """Run the program; its outputs stay where it writes them."""
        return tuple(
            self._method.execute([self._place(x, d) for x, d in zip(inputs, self._devices, strict=True)])
        )

    def to_numpy(self, value: Any) -> np.ndarray:
        """An output as a host array."""
        return np.asarray(value if str(value.device) == "cpu" else value.to_host())

    def _place(self, value: Any, device: str) -> Any:
        if not isinstance(value, np.ndarray):
            if str(value.device) == device:
                return value
            # The runtime hands a program whatever memory an input is in, so an output of another
            # program on another device is copied through the host first.
            value = self.to_numpy(value)
        value = np.ascontiguousarray(value)
        if device == "cpu":
            return value
        return self._from_host(value, int(device.removeprefix("cuda:")))
