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

"""Robot and camera drivers, and the loop helpers, load without PyTorch."""

import pytest

from tests.torch_free import assert_runs_without_torch

# Building an SO-101 needs the serial motor packages of the `hardware` extra.
pytestmark = pytest.mark.hardware


def test_drivers_and_loop_helpers_load_without_torch():
    assert_runs_without_torch(
        """
        import lerobot.cameras.opencv
        import lerobot.robots.koch_follower
        import lerobot.robots.so_follower
        import lerobot.teleoperators
        from lerobot.robots import make_robot_from_config
        from lerobot.robots.so_follower import SO101FollowerConfig
        from lerobot.utils import ActionInterpolator, get_safe_torch_device, init_logging
        from lerobot.utils.cycle_timer import CycleTimer
        from lerobot.utils.process import ProcessSignalHandler
        from lerobot.utils.robot_utils import policy_features, return_to_position

        robot = make_robot_from_config(SO101FollowerConfig(port="/dev/null"))
        policy_features(robot, None)
        """
    )
