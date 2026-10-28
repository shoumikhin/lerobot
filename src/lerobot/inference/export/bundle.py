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
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from huggingface_hub.constants import CONFIG_NAME

MANIFEST = "export.json"
FORMAT_VERSION = 1
TEST_CASE = "test.safetensors"
# The test case and the programs name the starting noise of a sampling policy this way.
NOISE = "noise"


@dataclass(frozen=True)
class ProgramSpec:
    """One compiled program of a bundle. Programs connect to each other by the names of their values.

    Args:
        name (`str`):
            The program's name, such as `model`.
        file (`str`):
            The program's file, relative to the bundle folder.
        inputs (`list[str]`):
            The names of its inputs, in order: keys of the robot's frame such as `observation.state`,
            `noise`, or outputs of an earlier program.
        outputs (`list[str]`):
            The names of its outputs, in order. The last program outputs `action`.
    """

    name: str
    file: str
    inputs: list[str]
    outputs: list[str]


@dataclass(frozen=True, kw_only=True)
class ExportManifest:
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
        programs (`list[ProgramSpec]`):
            The programs, in the order they run.
        noise_shape (`list[int]`, *optional*):
            The shape of the starting noise the programs take as `noise`. `None` for a policy that takes
            none.
        task (`str`, *optional*):
            The task the programs were exported for. `None` for a policy that does not read the task.
        tolerance (`float`):
            The largest difference allowed between the actions the programs compute for the test case and
            the actions PyTorch computed for it, in robot units.
        test (`str`, *optional*, defaults to `"test.safetensors"`):
            The test case file: a frame, the noise, and the expected actions under `action`.
    """

    format_version: int = FORMAT_VERSION
    backend: str
    runtime: str
    runtime_imports: list[str] = field(default_factory=list)
    programs: list[ProgramSpec]
    noise_shape: list[int] | None = None
    task: str | None = None
    tolerance: float
    test: str = TEST_CASE

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportManifest:
        """Read a manifest, refusing one written for a format this version of LeRobot cannot run."""
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"This bundle has format version {data.get('format_version')}, but this version of LeRobot "
                f"runs version {FORMAT_VERSION}. Export the policy again with this version of LeRobot."
            )
        # A key added later can change how the programs run, so a reader that does not know it stops.
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(
                f"This bundle uses {unknown}, which this version of LeRobot does not know. Upgrade LeRobot."
            )
        return cls(**{**data, "programs": [ProgramSpec(**program) for program in data["programs"]]})

    def to_dict(self) -> dict[str, Any]:
        """The manifest as JSON-ready data."""
        return asdict(self)


@dataclass(frozen=True)
class ExportBundle:
    """A bundle folder: its manifest, and what rollout reads from the policy's `config.json`.

    Args:
        path (`Path`):
            The bundle folder.
        manifest (`ExportManifest`):
            Its `export.json`.
        policy_type (`str`):
            The type of the exported policy, such as `act`.
        action_feature_names (`list[str]`, *optional*):
            The policy's joint order, when its checkpoint records one.
    """

    path: Path
    manifest: ExportManifest
    policy_type: str
    action_feature_names: list[str] | None = None

    @classmethod
    def read(cls, path: str | Path) -> ExportBundle:
        """Read the bundle in the folder `path`."""
        path = Path(path)
        manifest = ExportManifest.from_dict(json.loads((path / MANIFEST).read_text()))
        config = json.loads((path / CONFIG_NAME).read_text())
        return cls(path, manifest, config["type"], config.get("action_feature_names"))


def is_export_bundle(path: str | Path) -> bool:
    """Whether `path` is a folder written by `lerobot-export`, rather than a policy checkpoint."""
    return (Path(path) / MANIFEST).is_file()
