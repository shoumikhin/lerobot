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

"""Run an ExecuTorch program (`.pte`) of a bundle, without PyTorch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class ExecuTorchProgram:
    """The `forward` method of a `.pte` program.

    A program places each of its inputs on the CPU or on a GPU, as its backend compiled it. NumPy inputs
    are copied where the program reads them, and outputs stay where the program writes them.
    """

    def __init__(self, path: Path):
        """Load the program at `path`."""
        try:
            from executorch.runtime import ExecuTorchResult, Runtime
        except ImportError as error:
            raise ImportError(
                "Running this bundle needs ExecuTorch 1.6: pip install 'executorch>=1.6.0,<1.7.0'"
            ) from error

        self._from_host = ExecuTorchResult.from_host
        # A backend may keep the program's weights in a data file beside it.
        data = path.with_suffix(".ptd")
        program = Runtime.get().load_program(path, data_path=data if data.is_file() else None)
        self._method = program.load_method("forward")
        meta = self._method.metadata
        self._devices = [meta.input_tensor_meta(i).device() for i in range(meta.num_inputs())]

    def __call__(self, *inputs: Any) -> tuple[Any, ...]:
        """Run the program; its outputs stay where it writes them."""
        return tuple(
            self._method.execute([self._place(x, d) for x, d in zip(inputs, self._devices, strict=True)])
        )

    def to_numpy(self, value: Any) -> np.ndarray:
        """An output as a host array."""
        return np.asarray(value if value.device == "cpu" else value.to_host())

    def _place(self, value: Any, device: str) -> Any:
        if device == "cpu" or not isinstance(value, np.ndarray):
            return value
        return self._from_host(np.ascontiguousarray(value), int(device.removeprefix("cuda:")))
