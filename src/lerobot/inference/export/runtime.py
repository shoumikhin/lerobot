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

"""Load a bundle's program files, by the runtime its manifest names."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class Program(Protocol):
    """One compiled program, loaded by its runtime."""

    def __call__(self, *inputs: Any) -> tuple[Any, ...]:
        """Run the program once.

        Inputs are NumPy arrays or outputs of an earlier program of the same runtime. Outputs stay where the
        program left them, for example in GPU memory, so that the next program reads them without a copy.
        """
        ...

    def to_numpy(self, value: Any) -> np.ndarray:
        """`value`, an output of this program, as a NumPy array in host memory."""
        ...


def _load_executorch(path: Path) -> Program:
    from .executorch import ExecuTorchProgram

    return ExecuTorchProgram(path)


RUNTIMES: dict[str, Callable[[Path], Program]] = {"executorch": _load_executorch}
"""Program loaders by runtime name. A module a manifest names in `runtime_imports` may add one."""


def import_runtime_modules(modules: Sequence[str]) -> None:
    """Import the modules a bundle needs before its programs load."""
    # Set before any of them loads ExecuTorch, whose bindings read it once: ExecuTorch then returns its own
    # results rather than torch tensors, so running a bundle never imports torch.
    os.environ["EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT"] = "executorch"
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as error:
            raise ImportError(
                f"This bundle needs the Python module {module!r} to load its programs. Install it the way "
                "the export guide says for the backend that wrote the bundle: "
                "https://huggingface.co/docs/lerobot/export"
            ) from error


def load_program(runtime: str, path: Path) -> Program:
    """Load the program file `path` with the runtime named `runtime`."""
    if runtime not in RUNTIMES:
        raise ValueError(f"Unknown runtime {runtime!r}; this version of LeRobot runs {sorted(RUNTIMES)}.")
    return RUNTIMES[runtime](path)
