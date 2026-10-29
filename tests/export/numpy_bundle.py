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

"""Bundles run by a NumPy runtime, and a mock robot that records what it is told, all without PyTorch.

The runtime is not ExecuTorch, so the tests that use it show that running a bundle assumes no particular
runtime. A bundle names this module in its `runtime_imports`, which registers the runtime.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from lerobot.bundle.manifest import TEST_CASE, Bundle, BundleManifest, ProgramFiles, ProgramRun
from lerobot.bundle.runtime import RUNTIMES, Runtime
from lerobot.robots import RobotConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.mocks.mock_robot import MockRobot, MockRobotConfig

RUNTIME = "numpy"


class ScaleProgram:
    """A chunk of actions from a state: the state times each row of the scale the program file holds."""

    def __init__(self, path: Path, data: Path | None = None):
        self._scale = np.load(path)

    def __call__(self, state: np.ndarray) -> tuple[np.ndarray, ...]:
        return (state[None] * self._scale,)

    def to_numpy(self, value: np.ndarray) -> np.ndarray:
        return np.asarray(value)

    def place(self, index: int, value: np.ndarray) -> np.ndarray:
        return value


RUNTIMES[RUNTIME] = Runtime(ScaleProgram)


def make_numpy_bundle(
    path: Path,
    n_motors: int = 3,
    execution_steps: int = 2,
    action_names: list[str] | None = None,
    expected_offset: float = 0.0,
) -> Path:
    """A bundle whose chunk is the state times 1, 2, ..., `execution_steps`."""
    path.mkdir(parents=True)
    scale = np.arange(1, execution_steps + 1, dtype=np.float32)[:, None]
    np.save(path / "model.npy", scale)
    state = np.arange(1, n_motors + 1, dtype=np.float32)
    save_file({OBS_STATE: state, ACTION: state[None] * scale + expected_offset}, path / TEST_CASE)
    manifest = BundleManifest(
        backend="numpy",
        runtime=RUNTIME,
        runtime_imports=[__name__],
        programs={"model": ProgramFiles("model.npy")},
        runs=[ProgramRun("model", "model", [OBS_STATE], [ACTION])],
        action_names=action_names,
        action_scale=[1.0] * n_motors,
        tolerance=1e-6,
    )
    Bundle(path, manifest).write()
    return path


@RobotConfig.register_subclass("recording_robot")
@dataclass
class RecordingRobotConfig(MockRobotConfig):
    # A JSON-lines file of what happens to the robot: connect, each action sent, disconnect.
    log: str = ""
    random_values: bool = False


class RecordingRobot(MockRobot):
    """A mock robot that writes what it is told to `config.log`."""

    config_class = RecordingRobotConfig
    name = "recording_robot"

    def _log(self, event: str, value: object = None) -> None:
        with open(self.config.log, "a") as log:
            log.write(json.dumps([event, value]) + "\n")

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate)
        self._log("connect")

    def send_action(self, action):
        self._log("action", action)
        return super().send_action(action)

    def disconnect(self) -> None:
        super().disconnect()
        self._log("disconnect")


def read_log(path: Path) -> list[list]:
    """The events a `RecordingRobot` wrote, if any."""
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
