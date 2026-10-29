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
"""Modules that must import without torch, and a bundle rollout that must run without it.

Each check runs in a fresh interpreter where torch is installed but refuses to load, so a
failure's traceback points at the import that pulled torch in.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from lerobot.inference.bundle import MANIFEST, BundleManifest, ProgramSpec
from lerobot.utils.import_utils import _datasets_available, is_package_available

NEEDS_DATASETS = [
    pytest.mark.dataset,
    pytest.mark.skipif(not _datasets_available, reason="datasets not installed"),
]

# The spec is still found, so code that only checks whether torch is installed keeps working. Each
# load attempt is recorded, so an import wrapped in try/except still fails the test.
BLOCK_TORCH = """
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
"""

IMPORT_WITHOUT_TORCH = (
    BLOCK_TORCH
    + """
importlib.import_module(sys.argv[1])
if attempts:
    sys.exit(attempts[0])
"""
)


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
        "lerobot.inference",
        "lerobot.inference.bundle",
        "lerobot.processor",
        "lerobot.utils.action_interpolator",
        pytest.param("lerobot.scripts.lerobot_rollout", marks=NEEDS_DATASETS),
    ],
)
def test_imports_without_torch(module):
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_WITHOUT_TORCH, module], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr


# `lerobot-rollout` on a mock robot, run on the bundle in argv[1] with the repository in argv[2] for the mock.
ROLLOUT_WITHOUT_TORCH = (
    BLOCK_TORCH
    + """
import json
from pathlib import Path

bundle, repo = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, repo)
from tests.mocks.mock_robot import MockRobot

sent = []
send_action = MockRobot.send_action
MockRobot.send_action = lambda self, action: sent.append(dict(action)) or send_action(self, action)

from lerobot.scripts import lerobot_rollout

sys.argv = [
    "lerobot-rollout", f"--policy.path={bundle}", "--robot.type=mock_robot", "--robot.random_values=false",
    "--robot.static_values=[1,2,3]", "--strategy.type=base", "--duration=0.3", "--fps=20",
]
lerobot_rollout.main()
if attempts:
    sys.exit(attempts[0])
assert "torch" not in sys.modules
maps = Path("/proc/self/maps")
if maps.exists():
    assert "libtorch" not in maps.read_text(), "a torch library is mapped into the process"
print(json.dumps(sent[:2]))
"""
)

STATE = np.array([1, 2, 3], dtype=np.float32)
ACTIONS = np.stack([STATE, 2 * STATE])
ACTION_NAMES = ["motor_1.pos", "motor_2.pos", "motor_3.pos"]

# A runtime whose programs are NumPy functions, registered the way a compiler's module registers its kernels.
NUMPY_RUNTIME = """
import numpy as np

from lerobot.inference.bundle import RUNTIMES, Runtime


class TwoActions:
    def __init__(self, path, data=None):
        pass

    def __call__(self, state):
        return (np.stack([state, 2 * state]),)

    def to_numpy(self, value):
        return value

    def place(self, index, value):
        return value


RUNTIMES["numpy_test"] = Runtime(TwoActions)
"""


def write_bundle(folder: Path, runtime: str, runtime_imports: list[str], file: str) -> Path:
    folder.mkdir(exist_ok=True)
    save_file({"observation.state": STATE, "action": ACTIONS}, folder / "test.safetensors")
    manifest = BundleManifest(
        backend=runtime,
        runtime=runtime,
        runtime_imports=runtime_imports,
        programs=[ProgramSpec("model", file, ["observation.state"], ["action"])],
        action_names=ACTION_NAMES,
        action_scale=[1.0] * 3,
        tolerance=1e-6,
    )
    (folder / MANIFEST).write_text(json.dumps(manifest.to_dict()))
    return folder


def run_rollout_without_torch(bundle: Path, pythonpath: Path | None = None) -> list[dict]:
    env = None
    if pythonpath is not None:
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(pythonpath), os.environ.get("PYTHONPATH", "")]),
        }
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", ROLLOUT_WITHOUT_TORCH, str(bundle), str(repo)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def expected_actions() -> list[dict]:
    return [dict(zip(ACTION_NAMES, row.tolist(), strict=True)) for row in ACTIONS]


@pytest.mark.dataset
@pytest.mark.skipif(not _datasets_available, reason="datasets not installed")
def test_rollout_runs_a_bundle_without_torch(tmp_path):
    """`lerobot-rollout` checks a bundle, drives a robot with its chunk and never imports torch."""
    (tmp_path / "numpy_runtime.py").write_text(NUMPY_RUNTIME)
    bundle = write_bundle(tmp_path / "bundle", "numpy_test", ["numpy_runtime"], "model.bin")
    sent = run_rollout_without_torch(bundle, pythonpath=tmp_path)
    assert sent == expected_actions()


@pytest.mark.dataset
@pytest.mark.skipif(not _datasets_available, reason="datasets not installed")
@pytest.mark.skipif(not is_package_available("executorch"), reason="executorch not installed")
def test_rollout_runs_an_executorch_bundle_without_torch(tmp_path):
    """The same with a real ExecuTorch program, and with the TensorRT companion when it is installed."""
    import torch
    from executorch.exir import to_edge

    class TwoActions(torch.nn.Module):
        def forward(self, state):
            return torch.stack([state, 2 * state])

    program = to_edge(torch.export.export(TwoActions(), (torch.from_numpy(STATE),))).to_executorch()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.pte").write_bytes(program.buffer)
    companion = "torch_tensorrt_executorch_runtime"
    imports = [companion] if is_package_available(companion) else []
    sent = run_rollout_without_torch(write_bundle(bundle, "executorch", imports, "model.pte"))
    assert sent == expected_actions()
