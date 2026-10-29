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

"""Run a bundle written by `lerobot-export`: its programs, its test case, and the check of the robot.

Like the rest of `lerobot.bundle`, this imports no PyTorch.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
from safetensors.numpy import load_file

from lerobot.utils.constants import ACTION

from .manifest import NOISE, TASK_EMBEDDINGS, TASK_LENGTH, Bundle, BundleManifest
from .runtime import Program, load_program, prepare_runtime
from .target import check_runtime_versions, check_target
from .task_tokens import TaskEncoder

logger = logging.getLogger(__name__)


class BundleEngine:
    """Inline inference with a bundle: one run of its programs per action chunk.

    It plays the chunk one action per call, as [`SyncInferenceEngine`] plays the PyTorch policy's action
    queue. Before it runs, it replays the bundle's test case and refuses to start if the actions differ
    from the ones PyTorch computed by more than the bundle's tolerance. It is an
    [`~lerobot.utils.action_dispatch.ActionSource`], so a control loop dispatches its actions with the
    same helpers as `lerobot-rollout`.

    Args:
        bundle (`Bundle`):
            The bundle to run.
        task (`str`, *optional*, defaults to `""`):
            The task, by default the one the bundle was exported for. A bundle without `task_tokens` runs
            only that task.
        rename_map (`dict[str, str]`, *optional*):
            Renames the keys of the robot's frame to the names the bundle's programs take.
        stop_event (`threading.Event`, *optional*):
            Once set, no action may reach the robot, even one computed before it was set.
    """

    def __init__(
        self,
        bundle: Bundle,
        task: str = "",
        rename_map: dict[str, str] | None = None,
        stop_event: Event | None = None,
    ) -> None:
        """Load the bundle's programs and check them against its test case."""
        manifest = bundle.manifest
        self._fixed_task = manifest.task
        self._task_tokens = (
            None if manifest.task_tokens is None else TaskEncoder(bundle.path, manifest.task_tokens)
        )
        task = task or self._fixed_task or ""
        if error := self._task_error(task):
            raise ValueError(error)
        self.task = task
        self._stop = stop_event or Event()
        self._constants = {} if manifest.constants is None else load_file(bundle.path / manifest.constants)
        self._noise_shape = manifest.noise_shape
        self._action_names = manifest.action_names
        self._scale = np.asarray(manifest.action_scale, dtype=np.float32)
        if not (np.isfinite(self._scale) & (self._scale > 0)).all():
            raise ValueError(
                f"The bundle's action scale {manifest.action_scale} does not fit in 32-bit floats above 0. "
                "Export the policy again."
            )
        self._rename = rename_map or {}
        self._actions: deque[np.ndarray] = deque()
        case = load_file(bundle.path / manifest.test_case)
        expected = case.pop(ACTION, None)
        noise = case.pop(NOISE, None)
        if (
            expected is None
            or expected.ndim != 2
            or expected.shape[0] == 0
            or expected.shape[1] != len(self._scale)
            or not np.isfinite(expected).all()
        ):
            raise ValueError(
                "The bundle's test case holds no valid expected actions: finite numbers shaped "
                f"[execution_steps, {len(self._scale)}]. Export the policy again."
            )
        expected_noise = None if manifest.noise_shape is None else tuple(manifest.noise_shape)
        if (None if noise is None else noise.shape) != expected_noise:
            raise ValueError(
                f"The bundle's test case holds noise shaped {None if noise is None else noise.shape}, but its "
                f"programs take noise shaped {expected_noise}. Export the policy again."
            )
        self._frame_spec = {name: value.shape for name, value in case.items()}
        task_inputs = [TASK_EMBEDDINGS, TASK_LENGTH] if self._task_tokens else []
        _check_dataflow(
            manifest,
            [*self._frame_spec, *task_inputs],
            list(self._constants),
            manifest.noise_shape is not None,
        )
        check_target(manifest.target)
        check_runtime_versions(manifest.runtime_versions)
        prepare_runtime(manifest.runtime, manifest.runtime_imports)
        # Several runs of one program, such as the steps of a sampler, share it.
        loaded: dict[tuple[str, str | None], Program] = {}
        for spec in manifest.programs:
            if (spec.file, spec.data) not in loaded:
                loaded[spec.file, spec.data] = load_program(
                    manifest.runtime, bundle.path / spec.file, _data_file(bundle, spec.data)
                )
        self._programs = [(spec, loaded[spec.file, spec.data]) for spec in manifest.programs]
        # Each program gets the constants once, where it reads them, rather than a copy on every chunk.
        self._placed = [
            {
                index: program.place(index, self._constants[name])
                for index, name in enumerate(spec.inputs)
                if name in self._constants
            }
            for spec, program in self._programs
        ]
        self._check(case, noise, expected, manifest.tolerance)
        logger.info("Bundle loaded from %s (backend %s)", bundle.path, manifest.backend)

    def check_robot(self, observation_shapes: Mapping[str, tuple[int, ...]], action_names: list[str]) -> None:
        """Refuse a robot whose frame or actions differ from what the bundle takes.

        Args:
            observation_shapes (`Mapping[str, tuple[int, ...]]`):
                The shape of each key of the robot's frame, before `rename_map`.
            action_names (`list[str]`):
                The robot's action components, in the order the actions are sent.
        """
        renamed = [self._rename.get(name, name) for name in observation_shapes]
        if collisions := sorted({name for name in renamed if renamed.count(name) > 1}):
            raise ValueError(
                f"--rename_map gives several of the robot's observations the name {collisions}; each name "
                "must come from one observation."
            )
        shapes = dict(zip(renamed, map(tuple, observation_shapes.values()), strict=True))
        problems = [
            f"{name} {shape} (the robot gives {shapes.get(name)})"
            for name, shape in self._frame_spec.items()
            if shapes.get(name) != shape
        ]
        hint = " Use --rename_map to map the robot's names to the bundle's." if problems else ""
        if self._action_names is not None and list(action_names) != self._action_names:
            problems.append(f"the actions {self._action_names} (the robot takes {list(action_names)})")
        elif len(action_names) != len(self._scale):
            problems.append(f"{len(self._scale)} action components (the robot takes {len(action_names)})")
        if problems:
            raise ValueError(
                f"The robot does not match the exported policy, which takes {'; '.join(problems)}.{hint}"
            )

    def _run_chunk(
        self, frame: Mapping[str, np.ndarray], task: str, noise: np.ndarray | None = None
    ) -> np.ndarray:
        values: dict[str, Any] = {name: frame[name] for name in self._frame_spec}
        if self._task_tokens is not None:
            values[TASK_EMBEDDINGS], values[TASK_LENGTH] = self._task_tokens(task)
        if self._noise_shape is not None:
            values[NOISE] = (
                np.random.standard_normal(self._noise_shape).astype(np.float32) if noise is None else noise
            )
        for (spec, program), placed in zip(self._programs, self._placed, strict=True):
            inputs = []
            for index, name in enumerate(spec.inputs):
                if index in placed:
                    inputs.append(placed[index])
                    continue
                if isinstance(values[name], np.ndarray):
                    # Placed once per chunk; a later program that reads it on the same device gets this copy.
                    values[name] = program.place(index, values[name])
                inputs.append(values[name])
            values.update(zip(spec.outputs, program(*inputs), strict=True))
        actions = self._programs[-1][1].to_numpy(values[ACTION])
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != len(self._scale):
            raise RuntimeError(
                f"The exported policy returned actions shaped {actions.shape}, not [execution_steps, {len(self._scale)}]; "
                "they were not sent."
            )
        if not np.isfinite(actions).all():
            raise RuntimeError("The exported policy returned NaN or infinite actions; they were not sent.")
        return actions

    def _check(self, frame: dict, noise: np.ndarray | None, expected: np.ndarray, tolerance: float) -> None:
        actual = self._run_chunk(frame, self._fixed_task or "", noise)
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"The exported policy returns actions shaped {actual.shape}, but its test case expects "
                f"{expected.shape}. Export it again."
            )
        error = float((np.abs(actual - expected) / self._scale).max())
        logger.info("Bundle test case: largest difference %.2e (tolerance %g)", error, tolerance)
        if not np.isfinite(error) or error > tolerance:
            raise RuntimeError(
                f"The exported policy does not reproduce its test case ({error:.2e} > {tolerance}). "
                "Export it again on this device."
            )

    def _task_error(self, task: str) -> str | None:
        if self._task_tokens is not None:
            return self._task_tokens.error(task)
        if self._fixed_task is not None and task != self._fixed_task:
            return (
                f"This exported policy runs only the task it was exported for, {self._fixed_task!r}, "
                f"not {task!r}. Export it again for the new task."
            )
        return None

    def dispatch_allowed(self) -> bool:
        """Whether an action may reach the robot: until the stop event is set."""
        return not self._stop.is_set()

    def acknowledge_hold(self) -> None:
        """Inline inference has no worker waiting for the hold."""

    def record_dispatch(self) -> None:
        """Inline inference has nothing to do when an action reaches the robot."""

    def reset(self) -> None:
        """Drop the queued actions."""
        self._actions.clear()

    def get_action(self, obs_frame: dict | None) -> np.ndarray | None:
        """The next action of the chunk, running the programs when the chunk is used up."""
        if obs_frame is None:
            return None
        if not self._actions:
            self._actions.extend(
                self._run_chunk(
                    {self._rename.get(name, name): value for name, value in obs_frame.items()}, self.task
                )
            )
        return self._actions.popleft()


def _data_file(bundle: Bundle, name: str | None) -> Path | None:
    return None if name is None else bundle.path / name


def _check_dataflow(manifest: BundleManifest, frame: list[str], constants: list[str], noise: bool) -> None:
    """Refuse a bundle whose programs read a value that nothing gives them, before any program loads.

    A program may not write a constant either: each program reads the constants placed once, at load.
    """
    available = {*frame, *constants, *([NOISE] if noise else [])}
    for spec in manifest.programs:
        if shadowed := sorted(set(spec.outputs) & set(constants)):
            raise ValueError(
                f"The bundle's program run {spec.name!r} writes the constants {shadowed}. Export the policy again."
            )
        missing = [name for name in spec.inputs if name not in available]
        if missing:
            raise ValueError(
                f"The bundle's program run {spec.name!r} reads {missing}, which neither the frame, the noise, "
                "the constants nor an earlier program gives. Export the policy again."
            )
        available.update(spec.outputs)
    if ACTION not in manifest.programs[-1].outputs:
        raise ValueError(
            f"The bundle's last program run, {manifest.programs[-1].name!r}, gives no {ACTION!r}."
        )
