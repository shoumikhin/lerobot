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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import executorch


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


@dataclass(frozen=True)
class Runtime:
    """How to load the program files of one runtime.

    Args:
        load (`Callable[[Path, Path | None], Program]`):
            Loads a program file, with its data file if the manifest names one.
        prepare (`Callable[[], None]`, *optional*):
            Runs before the programs load, and before the manifest's `runtime_imports` import when the
            runtime is registered before them.
    """

    load: Callable[[Path, Path | None], Program]
    prepare: Callable[[], None] = lambda: None


RUNTIMES: dict[str, Runtime] = {"executorch": Runtime(executorch.ExecuTorchProgram, executorch.prepare)}
"""Runtimes by name. A module a manifest names in `runtime_imports` may add one."""


def prepare_runtime(runtime: str, modules: Sequence[str]) -> Runtime:
    """Get the runtime named `runtime` ready, import the modules a bundle needs, and return the runtime.

    The runtime prepares before the modules import, since a module may load native code that reads what
    `prepare` set. One that a module registers prepares once the modules are imported.
    """
    prepared = runtime in RUNTIMES
    if prepared:
        RUNTIMES[runtime].prepare()
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as error:
            raise ImportError(
                f"This bundle needs the Python module {module!r} to load its programs. Install it the way "
                "the export guide says for the backend that wrote the bundle: "
                "https://huggingface.co/docs/lerobot/export"
            ) from error
    if runtime not in RUNTIMES:
        raise ValueError(f"Unknown runtime {runtime!r}; this version of LeRobot runs {sorted(RUNTIMES)}.")
    if not prepared:
        RUNTIMES[runtime].prepare()
    return RUNTIMES[runtime]
