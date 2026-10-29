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
import importlib.util
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from lerobot.utils.import_utils import register_third_party_plugins

from . import executorch


class Program(Protocol):
    """One compiled program, loaded by its runtime."""

    def __call__(self, *inputs: Any) -> tuple[Any, ...]:
        """Run the program once.

        Inputs are what `place` returned, or outputs of an earlier program of the same runtime. Outputs stay
        where the program left them, for example in GPU memory, so that the next program reads them without a
        copy.
        """
        ...

    def to_numpy(self, value: Any) -> np.ndarray:
        """`value`, an output of this program, as a NumPy array in host memory."""
        ...

    def place(self, index: int, value: np.ndarray) -> Any:
        """`value` where input `index` reads it, to pass on every run without another copy."""
        ...


@dataclass(frozen=True)
class Runtime:
    """How to load the program files of one runtime.

    Args:
        load (`Callable[[Path, Path | None], Program]`):
            Loads a program file, with its data file if the manifest names one.
        prepare (`Callable[[], None]`, *optional*):
            Runs before the programs load, and before the manifest's `runtime_imports` import when the
            runtime is registered before them. It may run more than once.
    """

    load: Callable[[Path, Path | None], Program]
    prepare: Callable[[], None] = lambda: None


RUNTIMES: dict[str, Runtime] = {"executorch": Runtime(executorch.ExecuTorchProgram, executorch.prepare)}
"""The runtimes by name: the ones LeRobot has, and the ones packages added with `register_runtime`."""


def register_runtime(name: str, runtime: Runtime) -> None:
    """Add a runtime, from an installed package named `lerobot_runtime_*` or a module a manifest imports.

    Registering must not load the runtime's native code: that waits for `prepare` and `load`.
    """
    if name in RUNTIMES:
        raise ValueError(f"A runtime named {name!r} is already registered.")
    RUNTIMES[name] = runtime


def prepare_runtime(runtime: str, modules: Sequence[str], folder: Path | None = None) -> Runtime:
    """Get the runtime named `runtime` ready, import the modules a bundle needs, and return the runtime.

    `folder`, the bundle's, is off the module search path the whole time, so that a file the bundle ships
    cannot load in place of an installed module, whichever folder Python started in. A module the bundle
    names is refused when it would still load from inside `folder`, as `<folder name>.<module>` does when
    the folder above it is on the path.

    The runtime prepares before the modules import, since a module may load native code that reads what
    `prepare` set. A runtime LeRobot does not have is looked for in the installed `lerobot_runtime_*`
    packages; one that a module registers instead prepares once the modules are imported.
    """
    folder = None if folder is None else folder.resolve()
    removed = [] if folder is None else _leave_search_path(folder)
    try:
        if runtime not in RUNTIMES:
            register_third_party_plugins(kinds=("runtime",))
        registered_early = runtime in RUNTIMES
        if registered_early:
            RUNTIMES[runtime].prepare()
        for module in modules:
            if folder is not None and _loads_from(module, folder):
                raise ValueError(
                    f"This bundle names the Python module {module!r}, which would load from the bundle "
                    "folder. A bundle runs only installed modules."
                )
            try:
                importlib.import_module(module)
            except ImportError as error:
                raise ImportError(
                    f"This bundle could not import the Python module {module!r}, which loads its programs "
                    f"({error}). If it is not installed: install it the way the export guide says for the "
                    "backend that wrote the bundle, https://huggingface.co/docs/lerobot/export"
                ) from error
        if runtime not in RUNTIMES:
            raise ValueError(
                f"Unknown runtime {runtime!r}; this version of LeRobot runs {sorted(RUNTIMES)}. An installed "
                "package named lerobot_runtime_* can add one."
            )
        if not registered_early:
            RUNTIMES[runtime].prepare()
    finally:
        for index, entry in removed:
            sys.path.insert(index, entry)
    return RUNTIMES[runtime]


def _leave_search_path(folder: Path) -> list[tuple[int, str]]:
    """Take the entries of `sys.path` that name `folder` off it, and return them with their places.

    An empty entry is the current folder.
    """
    folder = folder.resolve()
    removed = [
        (index, entry) for index, entry in enumerate(sys.path) if Path(entry or ".").resolve() == folder
    ]
    for index, _ in reversed(removed):
        del sys.path[index]
    return removed


def _loads_from(module: str, folder: Path) -> bool:
    """Whether `module`, or a package it is in, would load from inside `folder`, a resolved path.

    Each package is looked up before the next part of the name, since finding a submodule imports its package.
    """
    parts = module.split(".")
    for end in range(1, len(parts) + 1):
        spec = importlib.util.find_spec(".".join(parts[:end]))
        if spec is None:
            return False
        places = list(spec.submodule_search_locations or [])
        if spec.has_location and spec.origin is not None:
            places.append(spec.origin)
        if any(Path(place).resolve().is_relative_to(folder) for place in places):
            return True
    return False
