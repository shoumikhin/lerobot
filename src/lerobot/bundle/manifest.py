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
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

MANIFEST = "export.json"
FORMAT_VERSION = 1
TEST_CASE = "test.safetensors"
CONSTANTS = "constants.safetensors"
# Not `config.json`, so that the folder does not look like a policy checkpoint.
POLICY_CONFIG = "policy_config.json"
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


@dataclass
class Target:
    """The GPU a bundle's programs were built on, when they run only on that kind of GPU.

    Args:
        device (`str`):
            The kind of device, `cuda`.
        gpu (`str`):
            The GPU's name, such as `Orin`.
        compute_capability (`str`):
            Its compute capability, such as `"8.7"`.
    """

    device: str
    gpu: str
    compute_capability: str


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
        target (`Target`, *optional*):
            The GPU the programs were built on, when they run only on that kind of GPU. `None` for programs
            that run anywhere their runtime does.
        runtime_versions (`dict[str, str]`, *optional*):
            The version of each Python module the programs were built with and need to load, such as
            `executorch`, by module name.
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
    target: Target | None = None
    runtime_versions: dict[str, str] = field(default_factory=dict)
    test_case: str = TEST_CASE

    def __post_init__(self) -> None:
        """Refuse values that would let any program pass its check."""
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
        programs = data.get("programs") or {}
        runs = data.get("runs") or []
        target = data.get("target")
        # A key added later can change how the programs run, so a reader that does not know it stops.
        unknown = (
            sorted(set(data) - {f.name for f in fields(cls)})
            + sorted(
                {key for files in programs.values() for key in files} - {f.name for f in fields(ProgramFiles)}
            )
            + sorted({key for run in runs for key in run} - {f.name for f in fields(ProgramRun)})
            + sorted(set(target or {}) - {f.name for f in fields(Target)})
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
                "target": None if target is None else Target(**target),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """The manifest as JSON-ready data."""
        return asdict(self)


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
