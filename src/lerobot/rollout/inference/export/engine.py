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

"""The inference engine for an exported policy.

An export script writes a folder with the compiled program, the policy's `config.json`, and an
`export.json` naming the backend, the program's inputs, and a test case. The program takes the
tensors `prepare_observation_for_inference` makes and returns the actions to play, in robot units.

A policy that reads the task also gets the token ids its saved text steps make from it, and a
flow-matching policy gets the starting noise as its last input, drawn here for every chunk. A
program compiled for one task holds that task in its weights, and `export.json` says so with
`task_fixed`: the engine then refuses any other task.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file

from lerobot.policies.common.flow_matching import sample_noise
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, RenameObservationsProcessorStep

from ..base import InferenceEngine

logger = logging.getLogger(__name__)

EXPORT_INFO = "export.json"
# All three backends run TensorRT engines, which run on CUDA only.
DEVICE = torch.device("cuda")
NOISE = "noise"

Program = Callable[..., torch.Tensor]


def export_dir(path: str | Path | None) -> Path | None:
    """The folder an export script wrote, or None when `path` holds a regular policy checkpoint."""
    if path is not None and (Path(path) / EXPORT_INFO).is_file():
        return Path(path)
    return None


def load_program(folder: Path, info: dict) -> Program:
    """Load the folder's program with the backend `export.json` names."""
    path = folder / info["file"]
    if info["backend"] == "executorch_tensorrt":
        from .executorch import ExecuTorchProgram

        return ExecuTorchProgram(path)
    if info["backend"] == "onnx_tensorrt":
        from .tensorrt import TensorRTEngine

        return TensorRTEngine(path, info["inputs"], info["output"])
    if info["backend"] == "aoti_tensorrt":
        from .aoti import AOTInductorPackage

        return AOTInductorPackage(path)
    raise ValueError(f"Unknown export backend {info['backend']!r} in {folder / EXPORT_INFO}")


class ExportInferenceEngine(InferenceEngine):
    """Inline inference with an exported program: one program call per action chunk.

    Plays the chunk one action per call, as `SyncInferenceEngine` does with the PyTorch policy's
    action queue. Before running, it replays the folder's test case and refuses to start if the
    program's actions differ from what `lerobot-rollout` played with PyTorch at export time.
    """

    def __init__(
        self, folder: Path, task: str, robot_type: str, rename_map: dict[str, str] | None = None
    ) -> None:
        super().__init__(task=task)
        info = json.loads((folder / EXPORT_INFO).read_text())
        self._input_names: list[str] = info["inputs"]
        self._robot_type = robot_type
        self._rename = RenameObservationsProcessorStep(rename_map=rename_map or {})
        self._noise_shape: tuple[int, ...] | None = (
            tuple(info["noise_shape"]) if "noise_shape" in info else None
        )
        self._text_steps = (
            PolicyProcessorPipeline.from_pretrained(folder, config_filename=info["text_steps"])
            if "text_steps" in info
            else None
        )
        self._program = load_program(folder, info)
        self._actions: deque[np.ndarray] = deque()
        self._fixed_task: str | None = info["task"] if info.get("task_fixed") else None
        self._check_task(task)
        self._check(folder / info["test_case"], info.get("task", task), info["tolerance"])
        logger.info("Exported policy loaded from %s (%s)", folder, info["backend"])

    def _run_chunk(
        self, frame: dict[str, np.ndarray], task: str, noise: np.ndarray | None = None
    ) -> np.ndarray:
        observation = prepare_observation_for_inference(dict(frame), DEVICE, task, self._robot_type)
        if self._text_steps is not None:
            observation = self._text_steps(observation)
        if self._noise_shape is not None:
            observation[NOISE] = (
                torch.from_numpy(noise).to(DEVICE)
                if noise is not None
                else sample_noise(self._noise_shape, DEVICE)
            )
        with torch.inference_mode():
            actions = self._program(*(observation[name] for name in self._input_names))
        return actions[0].cpu().numpy()

    def _check_task(self, task: str) -> None:
        if self._fixed_task is not None and task != self._fixed_task:
            raise ValueError(
                f"This exported policy only runs the task it was exported for, {self._fixed_task!r}, "
                f"not {task!r}. Export it again with the new task."
            )

    def _check(self, path: Path, task: str, tolerance: float) -> None:
        case = load_file(path)
        expected = case.pop("expected_actions")
        actual = self._run_chunk(case, task, case.pop(NOISE, None))
        error = float(np.abs(actual - expected).max()) if actual.shape == expected.shape else np.inf
        logger.info("Exported policy test case: largest difference %.2e (tolerance %g)", error, tolerance)
        if not error <= tolerance:
            raise RuntimeError(
                f"The exported policy does not reproduce its test case ({error:.2e} > {tolerance}). "
                "Export it again on this device."
            )

    @property
    def control_thread_owns_policy(self) -> bool:
        return True

    def start(self) -> None:
        """No background resources to start."""

    def stop(self) -> None:
        """No background resources to stop."""

    def reset(self) -> None:
        self._actions.clear()
        self._discard_task_change()

    def get_action(self, obs_frame: dict | None) -> np.ndarray | None:
        if obs_frame is None:
            return None
        task, task_changed = self._take_task()
        if task_changed:
            self._check_task(task)
            self._actions.clear()
        if not self._actions:
            start = time.perf_counter()
            self._actions.extend(self._run_chunk(self._rename.observation(obs_frame), task))
            self.inference_seconds.append(time.perf_counter() - start)
        self._set_dispatched_task(task)
        return self._actions.popleft()
