#!/usr/bin/env python

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
"""Modules that must import without torch.

Each module is imported in a fresh interpreter where torch is installed but refuses to load, so a
failure's traceback points at the import that pulled torch in.
"""

import subprocess
import sys

import pytest

from lerobot.utils.import_utils import _datasets_available

# The spec is still found, so code that only checks whether torch is installed keeps working. Each
# load attempt is recorded, so an import wrapped in try/except still fails the test.
IMPORT_WITHOUT_TORCH = """
import importlib
import importlib.abc
import importlib.machinery
import sys
import traceback

attempts = []


class RefuseLoader(importlib.abc.Loader):
    def exec_module(self, module):
        attempts.append("".join(traceback.format_stack()))
        raise RuntimeError("torch must not be imported here")


class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "torch":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        spec.loader = RefuseLoader()
        return spec


sys.meta_path.insert(0, BlockTorch())
importlib.import_module(sys.argv[1])
if attempts:
    sys.exit(attempts[0])
"""


@pytest.mark.parametrize(
    "module",
    [
        "lerobot.scripts.lerobot_calibrate",
        "lerobot.scripts.lerobot_find_cameras",
        "lerobot.scripts.lerobot_find_joint_limits",
        "lerobot.scripts.lerobot_find_port",
        "lerobot.scripts.lerobot_setup_can",
        "lerobot.scripts.lerobot_setup_motors",
        "lerobot.scripts.lerobot_teleoperate",
        "lerobot.configs",
        "lerobot.policies",
        "lerobot.processor",
        "lerobot.utils.action_interpolator",
        "lerobot.rollout.inference.export.tensorrt",
        *(
            pytest.param(
                module, marks=pytest.mark.skipif(not _datasets_available, reason="datasets not installed")
            )
            for module in [
                "lerobot.scripts.lerobot_rollout",
                "lerobot.datasets",
                "lerobot.datasets.pipeline_features",
                "lerobot.rollout.ring_buffer",
                "lerobot.rollout.status_line",
            ]
        ),
    ],
)
def test_imports_without_torch(module):
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_WITHOUT_TORCH, module], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr


def test_act_export_config_and_engine_run_without_torch(tmp_path):
    code = r'''
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from safetensors.numpy import save_file

from lerobot.rollout.configs import RolloutConfig
from lerobot.rollout.inference.export import ExportInferenceEngine
from lerobot.robots.config import RobotConfig

folder = Path(sys.argv[1])
info = {
    "backend": "executorch_cuda", "file": "model.pte",
    "inputs": ["observation.state", "observation.images.camera"],
    "output": "action", "test_case": "case.safetensors", "tolerance": 0.0,
}
(folder / "export.json").write_text(json.dumps(info))
(folder / "config.json").write_text(json.dumps({
    "type": "act", "input_features": {
        "observation.images.camera": {"type": "VISUAL", "shape": [3, 2, 2]}
    }, "action_feature_names": ["joint.pos"],
}))
state = np.array([2.0], dtype=np.float32)
image = np.full((2, 2, 3), 255, dtype=np.uint8)
save_file({"observation.state": state, "observation.images.camera": image,
           "expected_actions": np.array([[2.0], [4.0]], dtype=np.float32)},
          folder / "case.safetensors")

class Method:
    def execute(self, inputs):
        state, image = inputs
        assert isinstance(state, np.ndarray)
        assert image.shape == (1, 3, 2, 2) and image.dtype == np.float32
        np.testing.assert_array_equal(image, 1.0)
        return [np.stack([state, 2 * state], axis=1)]

runtime = SimpleNamespace(load_program=lambda *a, **kw: SimpleNamespace(load_method=lambda n: Method()))
sys.modules["executorch.runtime"] = SimpleNamespace(Runtime=SimpleNamespace(get=lambda: runtime))

@RobotConfig.register_subclass("test_export_robot")
@dataclass
class ExportRobotConfig(RobotConfig):
    pass

sys.argv = ["lerobot-rollout", f"--policy.path={folder}"]
cfg = RolloutConfig(robot=ExportRobotConfig())
assert cfg.policy.type == "act" and cfg.policy.action_feature_names == ["joint.pos"]
engine = ExportInferenceEngine(folder, task="", robot_type="test_export_robot")
frame = {"observation.state": state, "observation.images.camera": image}
np.testing.assert_array_equal(engine.get_action(frame), [2.0])
np.testing.assert_array_equal(engine.get_action(frame), [4.0])
assert "torch" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
