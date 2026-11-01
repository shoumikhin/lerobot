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
CONSTANTS = "constants.safetensors"
# Not `config.json`, so that the folder does not look like a policy checkpoint.
POLICY_CONFIG = "policy_config.json"
# The test case and the programs name the starting noise of a sampling policy this way.
NOISE = "noise"
# The programs of a policy that reads its task take it as these two values (see `TaskTokens`).
TASK_EMBEDDINGS = "task_embeddings"
TASK_LENGTH = "task_length"
# The files of `task_tokens`, as the export writes them.
TOKENIZER = "tokenizer.json"
EMBEDDINGS = "token_embeddings.safetensors"
KNOWN_TASKS = "known_tasks.json"
# The ops of `TaskTokens.prompt_ops`, each with the number of arguments it takes (`None`: one or more);
# `lerobot.bundle.task_tokens` implements them.
PROMPT_OPS: dict[str, int | None] = {
    "strip": 0,
    "lower": 0,
    "replace": 2,
    "keep": None,
    "default": 1,
    "template": 1,
}
# The kinds of character `["keep", ...]` names besides one literal character.
KEEP_KINDS = frozenset({"alnum", "space"})
# The token ids the runs of a policy that writes its actions as tokens generate, and the actions the bundle decodes
# from them (see `ActionTokens`).
GENERATED_TOKENS = "generated_tokens"
DECODED_ACTIONS = "decoded_actions"
# The file of `action_tokens`, as the export writes it.
ACTION_TOKENIZER = "action_tokenizer.json"


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
class TaskTokens:
    """How the task becomes the `task_embeddings` and `task_length` values the programs take.

    The task text is cleaned and placed in the prompt by `prompt_ops`, the tokenizer splits it into token
    ids, with its special tokens and without padding or truncation, and each id picks its row of the
    embedding table. The programs take `task_embeddings` as float32 `[max_tokens, width]`, with no batch
    dimension: those rows as the table holds them, so a policy that scales them does so in its programs,
    then rows of zeros. `task_length`, int64 `[1]`, counts the real rows.

    Args:
        prompt_ops (`list[list[str]]`):
            The ops that make the prompt text from the task, in order, each a name and its arguments:
            `["strip"]`, `["lower"]`, `["replace", old, new]`, `["keep", kind, ...]` (keeps the characters of
            each kind: `alnum`, `space`, or one literal character), `["default", text]` (for an empty
            task), and `["template", "... {task} ..."]`.
        tokenizer (`str`):
            The tokenizer file, in the format of the `tokenizers` library.
        embeddings (`str`):
            A safetensors file holding one tensor, the token embedding table, in float32 or bfloat16.
        known_tasks (`str`):
            A JSON file mapping each task known at export to its token ids, so those tasks never load the
            tokenizer.
        max_tokens (`int`):
            The most tokens the programs take.
    """

    prompt_ops: list[list[str]]
    tokenizer: str
    embeddings: str
    known_tasks: str
    max_tokens: int


@dataclass
class ActionTokens:
    """How the token ids the runs write as `generated_tokens` become the actions later runs read as `decoded_actions`.

    pi0-FAST writes its action chunk as text: `Action: `, the chunk's FAST tokens, then an end marker. FAST quantizes
    the discrete cosine transform (DCT) of each action component over the chunk, writes each coefficient as one
    character, and compresses the characters with byte-pair encoding. The bundle keeps the ids before the first end
    marker, drops every `Action: `, turns each id into its FAST token, joins the tokens' characters back into the
    coefficients and inverts the DCT, without PyTorch (see `lerobot.bundle.action_tokens`). It decodes once every run
    that writes `generated_tokens` has run, before the first run that reads `decoded_actions`.

    Args:
        tokenizer (`str`):
            The FAST tokenizer file, in the format of the `tokenizers` library: a byte-level BPE.
        clean_up_spaces (`bool`):
            Whether the decoded text drops the space before punctuation, as the policy's Transformers tokenizer
            does when it decodes.
        id_offset (`int`):
            The FAST token of a generated id is `id_offset - id`.
        end (`int`):
            The id of the end marker.
        prefix (`list[int]`):
            The ids of `Action: `, dropped wherever they appear.
        start (`list[int]`):
            The ids the generated tokens must start with, or none for a policy that does not check.
        min_token (`int`):
            The coefficient of the character `chr(0)`.
        scale (`float`):
            What the coefficients were multiplied by before they were rounded.
        horizon (`int`):
            The number of actions the decoding makes: the rows of `decoded_actions`.
        action_dim (`int`):
            The components of one action: the columns of `decoded_actions`.
    """

    tokenizer: str
    clean_up_spaces: bool
    id_offset: int
    end: int
    prefix: list[int]
    start: list[int]
    min_token: int
    scale: float
    horizon: int
    action_dim: int


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
        runtime_options (`dict[str, dict[str, bool | int | str]]`, *optional*):
            Settings the runtime applies before the programs load, chosen by the backend that compiled them.
            For ExecuTorch, the options of its backends, by backend name. They hold for the rest of the
            process, so bundles loaded after this one run with them too. LeRobot passes them on as they
            are; what a backend does with an option it does not read is up to the backend.
        programs (`dict[str, ProgramFiles]`):
            The compiled programs, by name.
        runs (`list[ProgramRun]`):
            The runs of the programs, in order.
        first_runs (`list[list[ProgramRun]]`, *optional*):
            For a policy whose first chunks after a reset compute otherwise than the later ones, the runs of each
            of them, one list per chunk. The chunks after them run `runs`.
        constants (`str`, *optional*):
            A safetensors file of named arrays the programs read, such as the timesteps of a sampler.
        task_tokens (`TaskTokens`, *optional*):
            How the task becomes the programs' `task_embeddings` and `task_length`. A bundle with it can run
            any task; one without it runs only `task`.
        action_tokens (`ActionTokens`, *optional*):
            How the token ids the runs generate become the actions later runs take, for a policy that writes its
            actions as tokens.
        noise_shape (`list[int]`, *optional*):
            The shape of the starting noise the programs take as `noise`. `None` for a policy that takes
            none. A run that writes `noise` draws it; without one, the engine draws it on the host. The test
            case's recorded noise replaces the draw when the bundle checks itself.
        history (`dict[str, int]`, *optional*):
            For each key of the robot's frame that the programs read with its past values, how many: the
            programs take that key as its last N frames, oldest first, stacked on a new first axis. After a
            reset, the first frame stands in for the ones not seen yet. Keys not listed are the current frame
            only.
        carry (`dict[str, list[int] | None]`, *optional*):
            The values the runs pass from one chunk to the next, by name, with their shapes, such as the warm
            start of a planner. After a reset each is float32 zeros. The runs read one like any value, and what
            it holds after the last run is what the next chunk reads. A value without a shape has none after a
            reset: the runs of the first chunk write it, in its own type and where they run, such as a cache.
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
        exact_runtime_modules (`list[str]`, *optional*):
            The modules of `runtime_versions` the programs load only at that exact version, such as `tensorrt`.
        test_case (`str`, *optional*, defaults to `"test.safetensors"`):
            The test case file: a frame, the noise, and the expected actions under `action`. With `history`,
            each frame value holds the frames of several control ticks, oldest first, and the actions are those
            computed at the last one. With `carry`, the frame values and the noise hold those of several chunks,
            oldest first, and the actions are those of the last chunk. With `first_runs`, the test case plays
            several chunks from a reset as a robot would: the frame values hold the frames of every control
            tick, the noise that of each chunk, and the actions the one played at each tick.
    """

    format_version: int = FORMAT_VERSION
    backend: str
    runtime: str
    runtime_imports: list[str] = field(default_factory=list)
    runtime_options: dict[str, dict[str, bool | int | str]] = field(default_factory=dict)
    programs: dict[str, ProgramFiles]
    runs: list[ProgramRun]
    first_runs: list[list[ProgramRun]] = field(default_factory=list)
    constants: str | None = None
    task_tokens: TaskTokens | None = None
    action_tokens: ActionTokens | None = None
    noise_shape: list[int] | None = None
    history: dict[str, int] = field(default_factory=dict)
    carry: dict[str, list[int] | None] = field(default_factory=dict)
    task: str | None = None
    action_names: list[str] | None = None
    action_scale: list[float]
    tolerance: float
    target: Target | None = None
    runtime_versions: dict[str, str] = field(default_factory=dict)
    exact_runtime_modules: list[str] = field(default_factory=list)
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
        if not all(self.first_runs):
            raise ValueError("The manifest lists no run for one of the first chunks.")
        if not all(type(n) is int and n > 0 for n in self.history.values()):
            raise ValueError(f"The history must keep a whole number of frames above 0, not {self.history}.")
        if not all(
            shape is None or (isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape))
            for shape in self.carry.values()
        ):
            raise ValueError(f"The carry must give each value a shape of whole numbers, not {self.carry}.")
        for runs in (*self.first_runs, self.runs):
            names = [run.name for run in runs]
            if len(set(names)) != len(names):
                raise ValueError(f"The manifest names a run twice: {names}.")
            if unknown := sorted({run.program for run in runs} - set(self.programs)):
                raise ValueError(f"The manifest runs the programs {unknown}, which it does not list.")
        files = [self.test_case, self.constants]
        files += [name for program in self.programs.values() for name in (program.file, program.data)]
        if self.task_tokens is not None:
            files += [self.task_tokens.tokenizer, self.task_tokens.embeddings, self.task_tokens.known_tasks]
        if self.action_tokens is not None:
            files.append(self.action_tokens.tokenizer)
        if outside := [
            name for name in files if name is not None and (Path(name).anchor or ".." in Path(name).parts)
        ]:
            raise ValueError(f"The manifest names files outside the bundle folder: {outside}.")

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
        first_runs = data.get("first_runs") or []
        target = data.get("target")
        task_tokens = data.get("task_tokens")
        prompt_ops = (task_tokens or {}).get("prompt_ops", [])
        action_tokens = data.get("action_tokens")
        # A key added later can change how the programs run, so a reader that does not know it stops.
        unknown = (
            sorted(set(data) - {f.name for f in fields(cls)})
            + sorted(
                {key for files in programs.values() for key in files} - {f.name for f in fields(ProgramFiles)}
            )
            + sorted(
                {key for run in [*runs, *(run for chunk in first_runs for run in chunk)] for key in run}
                - {f.name for f in fields(ProgramRun)}
            )
            + sorted(set(target or {}) - {f.name for f in fields(Target)})
            + sorted(set(task_tokens or {}) - {f.name for f in fields(TaskTokens)})
            + sorted({op[0] for op in prompt_ops} - PROMPT_OPS.keys())
            + sorted(
                {kind for name, *kinds in prompt_ops if name == "keep" for kind in kinds if len(kind) != 1}
                - KEEP_KINDS
            )
            + sorted(set(action_tokens or {}) - {f.name for f in fields(ActionTokens)})
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
                "first_runs": [[ProgramRun(**run) for run in chunk] for chunk in first_runs],
                "target": None if target is None else Target(**target),
                "task_tokens": None if task_tokens is None else TaskTokens(**task_tokens),
                "action_tokens": None if action_tokens is None else ActionTokens(**action_tokens),
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
    task_tokens = data.get("task_tokens")
    prompt_ops = task_tokens.get("prompt_ops", []) if isinstance(task_tokens, dict) else []
    form = {
        "programs": _map_of(data.get("programs") or {}, dict),
        "runs": _list_of(data.get("runs") or [], dict),
        "runtime_imports": _list_of(data.get("runtime_imports", []), str),
        "runtime_options": _map_of(data.get("runtime_options", {}), dict)
        and all(_map_of(values, bool | int | str) for values in data.get("runtime_options", {}).values()),
        "runtime_versions": _map_of(data.get("runtime_versions", {}), str),
        "exact_runtime_modules": _list_of(data.get("exact_runtime_modules", []), str),
        "target": data.get("target") is None or _map_of(data["target"], str),
        "task_tokens": task_tokens is None
        or (
            isinstance(task_tokens, dict)
            and _list_of(prompt_ops, list)
            and all(op and _list_of(op, str) and _takes_its_arguments(op) for op in prompt_ops)
            and type(task_tokens.get("max_tokens")) is int
            and task_tokens["max_tokens"] > 0
        ),
        "action_tokens": isinstance(data.get("action_tokens"), dict | None),
        "first_runs": _list_of(data.get("first_runs") or [], list)
        and all(_list_of(runs, dict) for runs in data.get("first_runs") or []),
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


def _takes_its_arguments(op: list[str]) -> bool:
    """Whether a prompt op has as many arguments as it takes; an op LeRobot does not know is refused later."""
    name, *args = op
    if name not in PROMPT_OPS:
        return True
    count = PROMPT_OPS[name]
    return len(args) > 0 if count is None else len(args) == count


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
