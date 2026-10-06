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
`export.json` naming the backend, the program's inputs, and a test case. Programs can take raw
frame arrays or the batched, normalized images used by older exports. The raw_frame flag or the
program's declared input shape distinguishes these contracts. Both return actions in robot units.

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
from safetensors.numpy import load_file

from ..base import InferenceEngine

logger = logging.getLogger(__name__)

EXPORT_INFO = "export.json"
NOISE = "noise"

Program = Callable[..., np.ndarray]


def export_dir(path: str | Path | None) -> Path | None:
    """The folder an export script wrote, or None when `path` holds a regular policy checkpoint."""
    if path is not None and (Path(path) / EXPORT_INFO).is_file():
        return Path(path)
    return None


def load_program(folder: Path, info: dict) -> Program:
    """Load the folder's program with the backend `export.json` names."""
    if info["backend"] == "onnx_tensorrt" and "programs" in info:
        from .tensorrt import TensorRTDenoisingLoop

        return TensorRTDenoisingLoop(folder, info["programs"], info["num_steps"])
    if info["backend"] == "executorch_tensorrt" and "programs" in info:
        from .executorch import ExecuTorchDenoisingLoop

        return ExecuTorchDenoisingLoop(folder, info["programs"], info["num_steps"])
    path = folder / info["file"]
    if info["backend"] == "executorch_tensorrt":
        from .executorch import ExecuTorchProgram

        return ExecuTorchProgram(path)
    if info["backend"] == "executorch_cuda":
        from .executorch import ExecuTorchProgram

        return ExecuTorchProgram(path, tensorrt=False)
    if info["backend"] == "onnx_tensorrt":
        from .tensorrt import TensorRTEngine

        return TensorRTEngine(path, info["inputs"], [info["output"]])
    if info["backend"] == "torch_tensorrt":
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
        self._rename = rename_map or {}
        self._noise_shape: tuple[int, ...] | None = (
            tuple(info["noise_shape"]) if "noise_shape" in info else None
        )
        self._text_steps = None
        if "text_steps" in info:
            from lerobot.processor import PolicyProcessorPipeline

            self._text_steps = PolicyProcessorPipeline.from_pretrained(
                folder, config_filename=info["text_steps"]
            )
        self._program = load_program(folder, info)
        self._raw_frame: bool | None = info.get("raw_frame")
        self._image_resize = info.get("image_resize")
        self._actions: deque[np.ndarray] = deque()
        self._fixed_task: str | None = info["task"] if info.get("task_fixed") else None
        self._check_task(task)
        self._check(folder / info["test_case"], info.get("task", task), info["tolerance"])
        logger.info("Exported policy loaded from %s (%s)", folder, info["backend"])

    def _run_chunk(
        self,
        frame: dict[str, np.ndarray],
        task: str,
        noise: np.ndarray | None = None,
        initialize: bool = False,
    ) -> np.ndarray:
        if self._image_resize is not None:
            from .images import resize_images

            frame = resize_images(frame, self._image_resize)
        observation = {}
        names = [name for name in self._input_names if name != NOISE] if self._raw_frame else frame
        for name in names:
            value = frame[name]
            if not self._raw_frame:
                if "image" in name:
                    if value.dtype == np.uint8:
                        value = value.astype(np.float32) / 255
                    value = value.transpose(2, 0, 1)
                value = value[None]
            observation[name] = np.ascontiguousarray(value)
        if self._text_steps is not None:
            import torch

            tensors = {name: torch.from_numpy(value) for name, value in observation.items()}
            tensors.update(task=task or "", robot_type=self._robot_type or "")
            with torch.inference_mode():
                observation = self._text_steps(tensors)
            observation = {
                name: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                for name, value in observation.items()
            }
        if self._noise_shape is not None:
            observation[NOISE] = (
                noise
                if noise is not None
                else np.random.standard_normal(self._noise_shape).astype(np.float32)
            )
        program = getattr(self._program, "initialize", self._program) if initialize else self._program
        actions = program(*(observation[name] for name in self._input_names))
        return (actions if self._raw_frame else actions[0]).copy()

    def _check_task(self, task: str) -> None:
        if self._fixed_task is not None and task != self._fixed_task:
            raise ValueError(
                f"This exported policy only runs the task it was exported for, {self._fixed_task!r}, "
                f"not {task!r}. Export it again with the new task."
            )

    def _check(self, path: Path, task: str, tolerance: float) -> None:
        case = load_file(path)
        expected = case.pop("expected_actions")
        input_shape = getattr(self._program, "input_shape", None)
        if self._raw_frame is None:
            self._raw_frame = (
                self._text_steps is None
                and input_shape is not None
                and input_shape == case[self._input_names[0]].shape
            )
        actual = self._run_chunk(case, task, case.pop(NOISE, None), initialize=True)
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
            frame = {self._rename.get(name, name): value for name, value in obs_frame.items()}
            self._actions.extend(self._run_chunk(frame, task))
            self.inference_seconds.append(time.perf_counter() - start)
        self._set_dispatched_task(task)
        return self._actions.popleft()
