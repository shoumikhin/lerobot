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

r"""Run a bundle written by `lerobot-export` on a robot, without PyTorch: `lerobot-rollout --bundle.path`.

It reuses the robot and camera configs, the robot factory, the joint-order rule, the dispatch helpers, the
interpolator, the cycle timer and the return to the starting pose of `lerobot-rollout`, and loads none of
the PyTorch packages. So it runs the base strategy only: recording, teleoperation, the planner, RTC and
remote inference need those packages, and stay with `lerobot-rollout --policy.path`.

```bash
lerobot-rollout \\
    --bundle.path=outputs/export/act_executorch_tensorrt \\
    --robot.type=so101_follower --robot.port=/dev/ttyACM0 \\
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --duration=30
```
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from threading import Event
from typing import cast

import draccus
import numpy as np

# Registers the robot and camera configs `--robot.type` and `--robot.cameras` choose from. The Unitree G1
# is not here: its config imports the simulation environments, which import PyTorch.
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    lekiwi,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
)
from lerobot.utils import ActionInterpolator, init_logging, register_third_party_plugins
from lerobot.utils.action_dispatch import dispatch_tick
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.import_utils import PLUGIN_DISCOVERY_SUFFIX, filter_arg, load_plugin, parse_plugin_args
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import policy_features, return_to_position

from .engine import BundleEngine
from .manifest import Bundle, is_bundle

logger = logging.getLogger(__name__)


@dataclass
class BundleConfig:
    """The bundle to run."""

    # The folder `lerobot-export` wrote. A bundle on the Hub must be downloaded first, for example with
    # `hf download <repo> --local-dir <folder>`.
    path: Path


@dataclass
class BundleRolloutConfig:
    """`lerobot-rollout --bundle.path`. The fields mean what they mean for `lerobot-rollout`."""

    bundle: BundleConfig
    robot: RobotConfig
    fps: float = 30.0
    # Run time in seconds; 0 runs until Ctrl-C.
    duration: float = 0.0
    # Robot commands sent per action of the bundle, interpolated in between.
    interpolation_multiplier: int = 1
    # The task, for a bundle that reads one. A bundle without task tokens runs only the task it was exported for.
    task: str = ""
    # Renames the robot's observation keys to the names the bundle takes.
    rename_map: dict[str, str] = field(default_factory=dict)
    # Move the robot back to the joint positions it had when it connected, before disconnecting.
    return_to_initial_position: bool = True

    def __post_init__(self) -> None:
        """Refuse a folder that is not a bundle before anything loads."""
        if not is_bundle(self.bundle.path):
            raise ValueError(
                f"{self.bundle.path} is not a bundle written by lerobot-export: it has no export.json. A "
                "bundle on the Hub must be downloaded first, for example with "
                "`hf download <repo> --local-dir <folder>`."
            )


@dataclass(frozen=True)
class RobotFrame:
    """Which robot values make up the bundle's frame, and which robot actions its actions are.

    The rule is the one `lerobot-rollout` uses for a PyTorch policy (see
    [`~lerobot.utils.robot_utils.policy_features`]): the state holds the position and velocity channels, in
    the order of the bundle's action names when both hold the same joints, and each camera is one image.
    """

    state: list[str]
    cameras: dict[str, tuple[int, ...]]
    actions: list[str]

    @classmethod
    def of(cls, robot: Robot, action_names: list[str] | None) -> RobotFrame:
        """The frame of `robot` for a bundle whose actions are named `action_names`, if it records them."""
        observation, action = policy_features(robot, action_names)
        return cls(
            state=[name for name, feature in observation.items() if not isinstance(feature, tuple)],
            cameras={name: feature for name, feature in observation.items() if isinstance(feature, tuple)},
            actions=list(action),
        )

    def shapes(self) -> dict[str, tuple[int, ...]]:
        """The shape of each key of the frame."""
        shapes = {f"{OBS_IMAGES}.{name}": shape for name, shape in self.cameras.items()}
        if self.state:
            shapes[OBS_STATE] = (len(self.state),)
        return shapes

    def frame(self, observation: dict) -> dict[str, np.ndarray]:
        """The frame for one robot observation."""
        frame = {f"{OBS_IMAGES}.{name}": observation[name] for name in self.cameras}
        if self.state:
            frame[OBS_STATE] = np.array([observation[name] for name in self.state], dtype=np.float32)
        return frame

    def action(self, target: np.ndarray) -> dict[str, float]:
        """The robot action for one target of the bundle, refusing one that could not be a command."""
        if target.shape != (len(self.actions),) or not np.isfinite(target).all():
            raise RuntimeError(
                f"Refused to send {target.tolist()} to the robot: it takes {len(self.actions)} finite numbers."
            )
        return dict(zip(self.actions, target.tolist(), strict=True))


def rollout(cfg: BundleRolloutConfig) -> None:
    """Check the bundle and the robot, then run the bundle on the robot until Ctrl-C or `cfg.duration`."""
    stop = cast(Event, ProcessSignalHandler(use_threads=True, display_pid=False).shutdown_event)
    bundle = Bundle.read(cfg.bundle.path)
    logger.info("Loading the bundle from '%s'...", bundle.path)
    engine = BundleEngine(bundle, task=cfg.task, rename_map=cfg.rename_map, stop_event=stop)
    robot = make_robot_from_config(cfg.robot)
    robot_frame = RobotFrame.of(robot, bundle.manifest.action_names)
    engine.check_robot(robot_frame.shapes(), robot_frame.actions)
    interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
    timer = CycleTimer(cfg.fps, interpolator.multiplier, records_data=False)

    def send(target: np.ndarray) -> None:
        robot.send_action(robot_frame.action(target))

    logger.info("Connecting robot (%s)...", robot.name)
    robot.connect()
    try:
        initial_position = {
            key: value for key, value in robot.get_observation().items() if key.endswith(".pos")
        }
        start = time.perf_counter()
        logger.info("Running the bundle on the robot")
        while not stop.is_set():
            timer.tick(new_cycle=interpolator.needs_new_action())
            if cfg.duration > 0 and time.perf_counter() - start >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break
            with timer.section("observe"):
                observation = robot.get_observation()
            dispatch_tick(
                engine, interpolator, partial(robot_frame.frame, observation), send=send, timer=timer
            )
            timer.wait()
        timer.log_run_summary()
        # Only after a clean stop: after an error, the robot is not moved again.
        if cfg.return_to_initial_position and initial_position:
            logger.info("Returning robot to initial position before shutdown...")
            return_to_position(robot, initial_position)
    finally:
        logger.info("Disconnecting robot")
        robot.disconnect()


def main(args: Sequence[str] | None = None) -> None:
    """`lerobot-rollout --bundle.path`, with the arguments of the command."""
    init_logging()
    args = sys.argv[1:] if args is None else list(args)
    # Only robots and cameras: policy, environment and strategy plugins may import PyTorch.
    register_third_party_plugins(kinds=("robot", "camera"))
    for arg, package in parse_plugin_args(PLUGIN_DISCOVERY_SUFFIX, args).items():
        if not arg.startswith("robot."):
            raise ValueError(f"--{arg}: bundle mode loads robot and camera plugins only.")
        load_plugin(package)
        args = filter_arg(arg, args)
    rollout(draccus.parse(BundleRolloutConfig, args=args))
