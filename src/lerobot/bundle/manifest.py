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
# The test case and the programs name the starting noise of a sampling policy this way.
NOISE = "noise"
# The programs of a policy that reads its task take it as these two values (see `TaskTokensSpec`).
TASK_EMBEDDINGS = "task_embeddings"
TASK_LENGTH = "task_length"
# The files of `task_tokens`, as the export writes them.
TOKENIZER = "tokenizer.json"
EMBEDDINGS_FILE = "token_embeddings.safetensors"
KNOWN_TASKS = "known_tasks.json"


@dataclass
class ProgramSpec:
    """One run of a compiled program. Programs connect to each other by the names of their values.

    Args:
        name (`str`):
            The name of this run, such as `model` or `step_3`.
        file (`str`):
            The program's file, relative to the bundle folder. Several runs may name one file, which is
            loaded once.
        inputs (`list[str]`):
            The names of its inputs, in order: keys of the robot's frame such as `observation.state`,
            `noise`, constants, or outputs of an earlier program.
        outputs (`list[str]`):
            The names of its outputs, in order. The last program outputs `action`.
        data (`str`, *optional*):
            A second file the runtime loads with the program, such as its weights, relative to the bundle
            folder.
    """

    name: str
    file: str
    inputs: list[str]
    outputs: list[str]
    data: str | None = None


@dataclass
class TaskTokensSpec:
    """How the task becomes the `task_embeddings` and `task_length` values the programs take.

    The task text is cleaned and placed in the prompt by `text`, the tokenizer splits it into token ids, and
    each id picks its row of the embedding table. The rows are padded with zeros to `max_tokens`, and
    `task_length` counts the real ones.

    Args:
        text (`list[list[str]]`):
            The ops that make the prompt text from the task, in order, each a name and its arguments:
            `["strip"]`, `["lower"]`, `["replace", old, new]`, `["keep", kind, ...]` (keeps the characters of
            each kind: `alnum`, `space`, or one literal character), `["default", text]` (for an empty
            task), and `["template", "... {task} ..."]`.
        tokenizer (`str`):
            The tokenizer file, in the format of the `tokenizers` library.
        table (`str`):
            A safetensors file holding one tensor, the token embedding table, in float32 or bfloat16.
        known (`str`):
            A JSON file mapping each task known at export to its token ids, so those tasks never load the
            tokenizer.
        max_tokens (`int`):
            The most tokens the programs take.
    """

    text: list[list[str]]
    tokenizer: str
    table: str
    known: str
    max_tokens: int


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
        programs (`list[ProgramSpec]`):
            The program runs, in order.
        constants (`str`, *optional*):
            A safetensors file of named arrays the programs read, such as the timesteps of a sampler.
        task_tokens (`TaskTokensSpec`, *optional*):
            How the task becomes the programs' `task_embeddings` and `task_length`. A bundle with it can run
            any task; one without it runs only `task`.
        noise_shape (`list[int]`, *optional*):
            The shape of the starting noise the programs take as `noise`. `None` for a policy that takes
            none.
        history (`dict[str, int]`, *optional*):
            For each key of the robot's frame that the programs read with its past values, how many: the
            programs take that key as its last N frames, oldest first, stacked on a new first axis. After a
            reset, the first frame stands in for the ones not seen yet. Keys not listed are the current frame
            only.
        task (`str`, *optional*):
            The task the programs were exported for. `None` for a policy that does not read the task.
        action_names (`list[str]`, *optional*):
            The robot's action components in the order the actions hold them, when the policy records them.
        action_scale (`list[float]`):
            For each action component, the robot units of one step of the policy's normalized actions.
        tolerance (`float`):
            The largest difference allowed between the actions the programs compute for the test case and
            the ones PyTorch computed, in normalized units: robot units divided by `action_scale`.
        test (`str`, *optional*, defaults to `"test.safetensors"`):
            The test case file: a frame, the noise, and the expected actions under `action`. With `history`,
            each frame value holds the frames of several control ticks, oldest first, and the actions are those
            computed at the last one.
    """

    format_version: int = FORMAT_VERSION
    backend: str
    runtime: str
    runtime_imports: list[str] = field(default_factory=list)
    programs: list[ProgramSpec]
    constants: str | None = None
    task_tokens: TaskTokensSpec | None = None
    noise_shape: list[int] | None = None
    history: dict[str, int] = field(default_factory=dict)
    task: str | None = None
    action_names: list[str] | None = None
    action_scale: list[float]
    tolerance: float
    test: str = TEST_CASE

    def __post_init__(self) -> None:
        """Refuse values that would let any program pass its check."""
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError(f"The tolerance must be a finite number of at least 0, not {self.tolerance}.")
        if not self.action_scale or not all(math.isfinite(s) and s > 0 for s in self.action_scale):
            raise ValueError(f"The action scale must hold finite numbers above 0, not {self.action_scale}.")
        if self.action_names is not None and len(self.action_names) != len(self.action_scale):
            raise ValueError("The manifest names a different number of actions than it scales.")
        if not self.programs:
            raise ValueError("The manifest lists no program.")
        if not all(type(n) is int and n > 0 for n in self.history.values()):
            raise ValueError(f"The history must keep a whole number of frames above 0, not {self.history}.")
        names = [program.name for program in self.programs]
        if len(set(names)) != len(names):
            raise ValueError(f"The manifest names a program run twice: {names}.")

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
        from .task_tokens import TEXT_OPS

        programs = data.get("programs") or []
        task_tokens = data.get("task_tokens")
        # A key added later can change how the programs run, so a reader that does not know it stops.
        unknown = (
            sorted(set(data) - {f.name for f in fields(cls)})
            + sorted({key for program in programs for key in program} - {f.name for f in fields(ProgramSpec)})
            + sorted(set(task_tokens or {}) - {f.name for f in fields(TaskTokensSpec)})
            + sorted({op[0] for op in (task_tokens or {}).get("text", [])} - set(TEXT_OPS))
        )
        if unknown:
            raise ValueError(
                f"This bundle uses {unknown}, which this version of LeRobot does not know. Upgrade LeRobot."
            )
        return cls(
            **{
                **data,
                "programs": [ProgramSpec(**program) for program in programs],
                "task_tokens": None if task_tokens is None else TaskTokensSpec(**task_tokens),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """The manifest as JSON-ready data."""
        return asdict(self)


@dataclass
class Bundle:
    """A bundle folder and its manifest.

    The folder also holds the policy's `config.json`, as a record of what was exported. Running the bundle
    does not read it.

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


def is_bundle(path: str | Path) -> bool:
    """Whether `path` is a local folder written by `lerobot-export`, rather than a policy checkpoint."""
    return (Path(path) / MANIFEST).is_file()
