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

"""The inference engine that runs a bundle written by `lerobot-export`."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np
from safetensors.numpy import load_file

from lerobot.utils.constants import ACTION

from ..base import InferenceEngine
from .bundle import NOISE, ExportBundle
from .runtime import import_runtime_modules, load_program

logger = logging.getLogger(__name__)


class ExportInferenceEngine(InferenceEngine):
    """Inline inference with a bundle: one run of its programs per action chunk.

    It plays the chunk one action per call, as [`SyncInferenceEngine`] plays the PyTorch policy's action
    queue. Before it runs, it replays the bundle's test case and refuses to start if the actions differ
    from the ones PyTorch computed by more than the bundle's tolerance.

    Args:
        bundle (`ExportBundle`):
            The bundle to run.
        task (`str`, *optional*, defaults to `""`):
            The task. A bundle exported for one task runs only that task, which is also the default.
        rename_map (`dict[str, str]`, *optional*):
            Renames the keys of the robot's frame to the names the bundle's programs take.
    """

    def __init__(
        self, bundle: ExportBundle, task: str = "", rename_map: dict[str, str] | None = None
    ) -> None:
        """Load the bundle's programs and check them against its test case."""
        manifest = bundle.manifest
        self._fixed_task = manifest.task
        task = task or self._fixed_task or ""
        if error := self._task_error(task):
            raise ValueError(error)
        super().__init__(task=task)
        import_runtime_modules(manifest.runtime_imports)
        self._programs = [
            (spec, load_program(manifest.runtime, bundle.path / spec.file)) for spec in manifest.programs
        ]
        self._noise_shape = manifest.noise_shape
        self._rename = rename_map or {}
        self._actions: deque[np.ndarray] = deque()
        case = load_file(bundle.path / manifest.test)
        expected = case.pop(ACTION)
        noise = case.pop(NOISE, None)
        self._frame_spec = {name: value.shape for name, value in case.items()}
        self._action_dim = expected.shape[-1]
        self._check(case, noise, expected, manifest.tolerance)
        logger.info("Exported policy loaded from %s (backend %s)", bundle.path, manifest.backend)

    def check_robot(self, observation_shapes: Mapping[str, tuple[int, ...]], action_names: list[str]) -> None:
        """Refuse a robot whose frame, after renaming, or whose actions differ from what the bundle takes.

        Args:
            observation_shapes (`Mapping[str, tuple[int, ...]]`):
                The shape of each key of the robot's frame, after `rename_map`.
            action_names (`list[str]`):
                The robot's action components, in the order the actions are sent.
        """
        problems = [
            f"{name} {shape} (the robot gives {observation_shapes.get(name)})"
            for name, shape in self._frame_spec.items()
            if tuple(observation_shapes.get(name, ())) != shape
        ]
        if len(action_names) != self._action_dim:
            problems.append(f"{self._action_dim} action components (the robot takes {len(action_names)})")
        if problems:
            raise ValueError(
                "The robot does not match the exported policy, which takes " + "; ".join(problems) + ". "
                "Use --rename_map to map camera names."
            )

    def _run_chunk(self, frame: Mapping[str, np.ndarray], noise: np.ndarray | None = None) -> np.ndarray:
        values: dict[str, Any] = {name: frame[name] for name in self._frame_spec}
        if self._noise_shape is not None:
            values[NOISE] = (
                np.random.standard_normal(self._noise_shape).astype(np.float32) if noise is None else noise
            )
        for spec, program in self._programs:
            values.update(zip(spec.outputs, program(*(values[name] for name in spec.inputs)), strict=True))
        actions = self._programs[-1][1].to_numpy(values[ACTION])
        if not np.isfinite(actions).all():
            raise RuntimeError("The exported policy returned NaN or infinite actions; they were not sent.")
        return actions

    def _check(self, frame: dict, noise: np.ndarray | None, expected: np.ndarray, tolerance: float) -> None:
        actual = self._run_chunk(frame, noise)
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"The exported policy returns actions shaped {actual.shape}, but its test case expects "
                f"{expected.shape}. Export it again."
            )
        error = float(np.abs(actual - expected).max())
        logger.info("Exported policy test case: largest difference %.2e (tolerance %g)", error, tolerance)
        if error > tolerance:
            raise RuntimeError(
                f"The exported policy does not reproduce its test case ({error:.2e} > {tolerance}). "
                "Export it again on this device."
            )

    def _task_error(self, task: str) -> str | None:
        if self._fixed_task is not None and task != self._fixed_task:
            return (
                f"This exported policy runs only the task it was exported for, {self._fixed_task!r}, "
                f"not {task!r}. Export it again for the new task."
            )
        return None

    @property
    def control_thread_owns_policy(self) -> bool:
        """The control thread runs the programs itself."""
        return True

    def text_input_error(self, text: str, *, instruction: bool = False) -> str | None:
        """Refuse an instruction other than the task the bundle was exported for."""
        if instruction and (error := self._task_error(text)):
            return error
        return super().text_input_error(text, instruction=instruction)

    def set_task(self, task: str) -> bool:
        """Change the task, unless the bundle was exported for another one."""
        if error := self._task_error(task):
            logger.warning("Instruction rejected: %s", error)
            return False
        return super().set_task(task)

    def dispatch_allowed(self) -> bool:
        """Hold the robot while an external planner answers; inline inference has no worker to pause."""
        return not self.hold_for_planner()

    def start(self) -> None:
        """No background resources to start."""

    def stop(self) -> None:
        """No background resources to stop."""

    def reset(self) -> None:
        """Drop the queued actions and any pending task change."""
        self._actions.clear()
        self._discard_task_change()

    def get_action(self, obs_frame: dict | None) -> np.ndarray | None:
        """The next action of the chunk, running the programs when the chunk is used up."""
        if obs_frame is None:
            return None
        task, task_changed = self._take_task()
        if task_changed:
            self._actions.clear()
        if not self._actions:
            start = time.perf_counter()
            self._actions.extend(
                self._run_chunk({self._rename.get(name, name): value for name, value in obs_frame.items()})
            )
            end = time.perf_counter()
            self.inference_times.append((end, end - start))
        self._set_dispatched_task(task)
        return self._actions.popleft()
