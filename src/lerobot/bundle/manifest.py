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


"""The folder `lerobot-export` writes, and its manifest `export.json`."""

from __future__ import annotations

import json
import math
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

MANIFEST = "export.json"
FORMAT_VERSION = 1
TEST_CASE = "test.safetensors"
# The test case and the programs name the starting noise of a sampling policy this way.
NOISE = "noise"


@dataclass
class ProgramFiles:
    """The files of one compiled program, relative to the bundle folder.

    Args:
        file (`str`):
            The program file.
        data (`str`, *optional*):
            A second file the runtime loads with the program, such as its weights.
    """

    file: str
    data: str | None = None


@dataclass
class ProgramRun:
    """One run of a program. Runs connect to each other by the names of their values.

    Args:
        name (`str`):
            The name of this run, such as `model` or `step_3`.
        program (`str`):
            The program it runs: its name, a key of the manifest's `programs`. Several runs may run one
            program.
        inputs (`list[str]`):
            The names of the values it reads, in the order of the program's inputs: keys of the robot's
            frame such as `observation.state`, `noise`, constants, or outputs of an earlier run.
        outputs (`list[str]`):
            The names of the values it writes, in the order of the program's outputs. The last run writes
            `action`.
    """

    name: str
    program: str
    inputs: list[str]
    outputs: list[str]


@dataclass(kw_only=True)
class BundleManifest:
    """The contents of a bundle's `export.json`.

    Args:
        format_version (`int`, *optional*, defaults to `1`):
            Changes only when a reader of an older version could not run the bundle.
        backend (`str`):
            The `--backend.type` that compiled the programs.
        runtime (`str`):
            The runtime that loads the programs, such as `executorch`.
        runtime_imports (`list[str]`, *optional*):
            Python modules to import before the programs load, such as one that registers a compiler's
            kernels with the runtime.
        programs (`dict[str, ProgramFiles]`):
            The compiled programs, by name.
        runs (`list[ProgramRun]`):
            The runs of the programs, in order.
        constants (`str`, *optional*):
            A safetensors file of named arrays the programs read, such as the timesteps of a sampler.
        noise_shape (`list[int]`, *optional*):
            The shape of the starting noise the programs take as `noise`. `None` for a policy that takes
            none.
        task (`str`, *optional*):
            The task the programs were exported for. `None` for a policy that does not read the task.
        action_names (`list[str]`, *optional*):
            The robot's action components in the order the actions hold them, when the policy records them.
        action_scale (`list[float]`):
            For each action component, the robot units of one step of the policy's normalized actions.
        tolerance (`float`):
            The largest difference allowed between the actions the programs compute for the test case and
            the ones PyTorch computed, in normalized units: robot units divided by `action_scale`.
        test_case (`str`, *optional*, defaults to `"test.safetensors"`):
            The test case file: a frame, the noise, and the expected actions under `action`.
    """

    format_version: int = FORMAT_VERSION
    backend: str
    runtime: str
    runtime_imports: list[str] = field(default_factory=list)
    programs: dict[str, ProgramFiles]
    runs: list[ProgramRun]
    constants: str | None = None
    noise_shape: list[int] | None = None
    task: str | None = None
    action_names: list[str] | None = None
    action_scale: list[float]
    tolerance: float
    test_case: str = TEST_CASE

    def __post_init__(self) -> None:
        """Refuse values that would let any program pass its check, and files outside the bundle folder."""
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError(f"The tolerance must be a finite number of at least 0, not {self.tolerance}.")
        if not self.action_scale or not all(math.isfinite(s) and s > 0 for s in self.action_scale):
            raise ValueError(f"The action scale must hold finite numbers above 0, not {self.action_scale}.")
        if self.action_names is not None and len(self.action_names) != len(self.action_scale):
            raise ValueError("The manifest names a different number of actions than it scales.")
        if not self.runs:
            raise ValueError("The manifest lists no run.")
        names = [run.name for run in self.runs]
        if len(set(names)) != len(names):
            raise ValueError(f"The manifest names a run twice: {names}.")
        if unknown := sorted({run.program for run in self.runs} - set(self.programs)):
            raise ValueError(f"The manifest runs the programs {unknown}, which it does not list.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BundleManifest:
        """Read a manifest, refusing one this version of LeRobot cannot run."""
        version = data.get("format_version")
        if not isinstance(version, int) or version != FORMAT_VERSION:
            fix = (
                "Upgrade LeRobot."
                if isinstance(version, int) and version > FORMAT_VERSION
                else "Export the policy again."
            )
            raise ValueError(
                f"This bundle has format version {version}, but this version of LeRobot runs version "
                f"{FORMAT_VERSION}. {fix}"
            )
        _check_form(data)
        programs = data.get("programs") or {}
        runs = data.get("runs") or []
        # A key added later can change how the programs run, so a reader that does not know it stops.
        unknown = (
            sorted(set(data) - {f.name for f in fields(cls)})
            + sorted(
                {key for files in programs.values() for key in files} - {f.name for f in fields(ProgramFiles)}
            )
            + sorted({key for run in runs for key in run} - {f.name for f in fields(ProgramRun)})
        )
        if unknown:
            raise ValueError(
                f"This bundle uses {unknown}, which this version of LeRobot does not know. Upgrade LeRobot."
            )
        return cls(
            **{
                **data,
                "programs": {name: ProgramFiles(**files) for name, files in programs.items()},
                "runs": [ProgramRun(**run) for run in runs],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """The manifest as JSON-ready data, without the fields left at their default except `format_version`.

        A reader refuses a key it does not know, so a field a later version adds stops this one only in a
        bundle that sets it.
        """
        return {"format_version": self.format_version, **_set_fields(self)}


def _set_fields(value: Any) -> Any:
    """`value` as JSON-ready data, with each dataclass field that holds its default left out."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _set_fields(getattr(value, f.name))
            for f in fields(value)
            if not (
                (f.default is not MISSING and getattr(value, f.name) == f.default)
                or (f.default_factory is not MISSING and getattr(value, f.name) == f.default_factory())
            )
        }
    if isinstance(value, dict):
        return {key: _set_fields(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_set_fields(item) for item in value]
    return value


def _check_form(data: dict[str, Any]) -> None:
    """Refuse a manifest whose fields lack the form `from_dict` reads, or a field it must have.

    It runs before the check for unknown keys, which would read a name given in place of an object as the
    unknown keys of its letters.
    """
    form = {
        "programs": _map_of(data.get("programs") or {}, dict),
        "runs": _list_of(data.get("runs") or [], dict),
        "runtime_imports": _list_of(data.get("runtime_imports", []), str),
    }
    if wrong := [name for name, right in form.items() if not right]:
        raise ValueError(
            f"The bundle's manifest gives {wrong} in a form LeRobot does not read. Export the policy again."
        )
    if missing := [
        f.name
        for f in fields(BundleManifest)
        if f.default is MISSING and f.default_factory is MISSING and f.name not in data
    ]:
        raise ValueError(f"The bundle's manifest lacks {missing}. Export the policy again.")


def _list_of(value: Any, kind: type) -> bool:
    return isinstance(value, list) and all(isinstance(item, kind) for item in value)


def _map_of(value: Any, kind: type) -> bool:
    return isinstance(value, dict) and all(isinstance(item, kind) for item in value.values())


@dataclass
class Bundle:
    """A bundle folder and its manifest.

    The folder also holds the policy's `policy_config.json`, as a record of what was exported. Running the
    bundle does not read it.

    Args:
        path (`Path`):
            The bundle folder.
        manifest (`BundleManifest`):
            Its `export.json`.
    """

    path: Path
    manifest: BundleManifest

    @classmethod
    def read(cls, path: str | Path) -> Bundle:
        """Read the bundle in the folder `path`."""
        path = Path(path)
        return cls(path, BundleManifest.from_dict(json.loads((path / MANIFEST).read_text())))

    def write(self) -> None:
        """Write the manifest into the bundle folder."""
        (self.path / MANIFEST).write_text(json.dumps(self.manifest.to_dict(), indent=2) + "\n")


def is_bundle(path: str | Path) -> bool:
    """Whether `path` is a local folder written by `lerobot-export`, rather than a policy checkpoint."""
    return (Path(path) / MANIFEST).is_file()
