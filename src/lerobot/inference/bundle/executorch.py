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

import functools
import importlib.util
import os
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np


def prepare() -> None:
    """Make ExecuTorch return its own results rather than torch tensors.

    ExecuTorch reads this once, when its bindings load, so it is set before any module imports them.
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
        prepare()
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
        outputs = [meta.output_tensor_meta(i).device() for i in range(meta.num_outputs())]
        self.cuda_devices = {
            int(device.removeprefix("cuda:")) for device in (*self._devices, *outputs) if device != "cpu"
        }

    def __call__(self, *inputs: Any) -> tuple[Any, ...]:
        """Run the program; its outputs stay where it writes them."""
        return tuple(
            self._method.execute([self._place(x, d) for x, d in zip(inputs, self._devices, strict=True)])
        )

    def to_numpy(self, value: Any) -> np.ndarray:
        """An output as a host array."""
        return np.asarray(value if value.device == "cpu" else value.to_host())

    def place(self, index: int, value: np.ndarray) -> Any:
        """`value` copied where input `index` reads it, once."""
        return self._place(value, self._devices[index])

    def _place(self, value: Any, device: str) -> Any:
        if not isinstance(value, np.ndarray):
            if value.device == device:
                return value
            # The runtime hands a program whatever memory an input is in, so an output of another
            # program on another device is copied through the host first.
            value = self.to_numpy(value)
        value = np.ascontiguousarray(value)
        if device == "cpu":
            return value
        return self._from_host(value, int(device.removeprefix("cuda:")))


def chunk_scope(programs: Sequence[ExecuTorchProgram]) -> Callable[[], AbstractContextManager[Any]]:
    """One CUDA stream for every copy and run, if the programs' inputs or outputs are on GPU 0 only.

    The CUDA and TensorRT delegates then queue their work on that stream rather than each on its own.
    ExecuTorch selects a stream on CUDA device 0 only, so programs on another GPU, and programs under an
    ExecuTorch that has no `executorch.runtime.cuda`, run as before.
    """
    devices = set().union(*(program.cuda_devices for program in programs))
    if devices != {0} or importlib.util.find_spec("executorch.runtime.cuda") is None:
        return nullcontext
    from executorch.runtime import cuda

    return functools.partial(cuda.stream, cuda.Stream(0))
