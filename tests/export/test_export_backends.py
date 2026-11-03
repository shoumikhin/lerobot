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

"""Each policy exported with each backend that ExecuTorch runs, end to end on tiny checkpoints.

The export passes only when the bundle reproduces the PyTorch policy's test case. Then each program must run where
its backend runs it, and `lerobot-rollout --bundle.path` must drive a robot with the bundle without PyTorch.
"""

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest
from packaging.version import Version
from safetensors.numpy import load_file

from lerobot.bundle.manifest import Bundle
from lerobot.scripts.lerobot_export import export
from lerobot.utils.constants import OBS_STATE
from tests.export import tiny_groot, tiny_pi0, tiny_pi05, tiny_smolvla
from tests.export.backends import BACKENDS, GPU_BACKENDS, INSTALLED
from tests.export.plugins import use_plugins
from tests.torch_free import assert_runs_without_torch

TASK = "pick up the cube"
DELEGATES = {"executorch_xnnpack": "XnnpackBackend", "executorch_cuda": "CudaBackend"}
TRANSFORMERS = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None, reason="transformers not installed"
)
DIFFUSERS = pytest.mark.skipif(
    importlib.util.find_spec("diffusers") is None, reason="diffusers not installed"
)
GROOT_EXTRAS = pytest.mark.skipif(
    not all(importlib.util.find_spec(name) for name in ("diffusers", "cv2")),
    reason="GR00T needs diffusers and OpenCV",
)


def _release(package: str) -> tuple[int, ...]:
    try:
        return Version(version(package)).release
    except PackageNotFoundError:
        return ()


# Each skips up to the newest release that has the bug, so a newer wheel runs the row again and shows the fix.
XNNPACK_8D_PERMUTE = pytest.mark.skipif(
    _release("executorch") <= (1, 6, 0),
    reason="ExecuTorch 1.6.0's XNNPACK cannot load GR00T's vision program, which permutes an 8-D tensor "
    f"(pytorch/executorch#24380) ({INSTALLED})",
)
TENSORRT_TINY_CONCAT = pytest.mark.skipif(
    _release("tensorrt-cu13") <= (11, 4, 0, 106),
    reason="TensorRT 11.4.0.106 cannot build an engine for the tiny policy's concatenations (Myelin "
    f"CHECK(def_tactic_desc) failed), which the full-size policy builds ({INSTALLED})",
)
KNOWN_BUGS = {
    ("groot", "executorch_xnnpack"): (XNNPACK_8D_PERMUTE,),
    ("pi0", "executorch_tensorrt"): (TENSORRT_TINY_CONCAT,),
    ("pi05", "executorch_tensorrt"): (TENSORRT_TINY_CONCAT,),
}


@dataclass
class Policy:
    """A tiny checkpoint that `save` writes into a folder and returns the path of, and how it exports.

    `plugins` make the policy tiny in the export's own processes, as `save` does in this one.
    """

    save: Callable[[Path, pytest.MonkeyPatch], Path]
    plugins: tuple[str, ...] = ()
    args: tuple[str, ...] = ()
    marks: tuple[pytest.MarkDecorator, ...] = ()


def _act(folder: Path, patch: pytest.MonkeyPatch) -> Path:
    from tests.export.tiny_act import IMAGE, save_checkpoint

    save_checkpoint(folder / "checkpoint", IMAGE)
    return folder / "checkpoint"


def _diffusion(folder: Path, patch: pytest.MonkeyPatch) -> Path:
    from tests.export.test_export_history import diffusion_checkpoint

    diffusion_checkpoint(folder / "checkpoint")
    return folder / "checkpoint"


def _tiny(module) -> Callable[[Path, pytest.MonkeyPatch], Path]:
    def save(folder: Path, patch: pytest.MonkeyPatch) -> Path:
        module.save_checkpoint(folder / "checkpoint", patch)
        return folder / "checkpoint"

    return save


POLICIES = {
    "act": Policy(_act),
    "diffusion": Policy(_diffusion, marks=(DIFFUSERS,)),
    "pi0": Policy(_tiny(tiny_pi0), plugins=("tiny_pi0",), marks=(TRANSFORMERS,)),
    "pi05": Policy(_tiny(tiny_pi05), plugins=("tiny_pi05",), marks=(TRANSFORMERS,)),
    "smolvla": Policy(_tiny(tiny_smolvla), plugins=("tiny_smolvla",), marks=(TRANSFORMERS,)),
    # The tolerance of this random tiny GR00T's chain in tests/export/test_policy_chunks.py.
    "groot": Policy(
        lambda folder, patch: tiny_groot.save_checkpoint(folder, "torchvision"),
        args=("--tolerance=1e-4",),
        marks=(TRANSFORMERS, GROOT_EXTRAS),
    ),
}


@pytest.fixture(
    scope="module",
    params=[
        pytest.param(
            (name, backend),
            marks=[*policy.marks, mark, *KNOWN_BUGS.get((name, backend), ())],
            id=f"{name}-{backend}",
        )
        for name, policy in POLICIES.items()
        for backend, mark in BACKENDS.items()
    ],
)
def exported(request, tmp_path_factory) -> tuple[str, Bundle]:
    name, backend = request.param
    policy, folder = POLICIES[name], tmp_path_factory.mktemp(f"{name}-{backend}")
    with pytest.MonkeyPatch.context() as patch:
        checkpoint = policy.save(folder, patch)
        use_plugins(patch, *policy.plugins)
        export(
            [
                f"--backend.type={backend}",
                f"--policy.path={checkpoint}",
                f"--output_dir={folder / 'bundle'}",
                f"--task={TASK}",
                *policy.args,
            ]
        )
    return backend, Bundle.read(folder / "bundle")


def test_each_program_runs_on_the_backend(exported):
    from executorch.exir._serialize._program import deserialize_pte_binary

    backend, bundle = exported
    delegates = set()
    for files in bundle.manifest.programs.values():
        (plan,) = deserialize_pte_binary((bundle.path / files.file).read_bytes()).program.execution_plan
        delegates |= {delegate.id for delegate in plan.delegates}
        if backend in GPU_BACKENDS:
            # Nothing runs on the CPU between the GPU's parts.
            assert [operator.name for operator in plan.operators] == [], files.file
    if backend == "executorch_tensorrt":
        # An operator TensorRT does not take runs on ExecuTorch's CUDA backend.
        assert "TensorRTBackend" in delegates and delegates <= {"TensorRTBackend", "CudaBackend"}, delegates
    else:
        assert delegates == {DELEGATES[backend]}, delegates


ROLLOUT = """
import json
import os
import sys

from lerobot.scripts.lerobot_rollout_entry import main

log, actions = sys.argv[1], int(sys.argv[2])
if os.path.exists(log):
    os.remove(log)
sys.argv = ["lerobot-rollout", *sys.argv[3:], f"--robot.log={log}"]
main()
events = [json.loads(line) for line in open(log)]
sent = [value for event, value in events if event == "action"]
assert events[0][0] == "connect" and events[-1][0] == "disconnect", events
assert len(sent) >= actions, len(sent)
"""


def test_the_bundle_drives_a_robot_without_torch(exported, tmp_path, monkeypatch):
    """The rollout first replays the bundle's test case, then sends the bundle's actions, at least three chunks of
    the tiny policies' few actions each."""
    _, bundle = exported
    frames = bundle.path / bundle.manifest.test_case
    state = load_file(frames)[OBS_STATE]
    # A test case of several control ticks holds a state per tick, as the robot's images do.
    state = (state if state.ndim == 1 else state[-1]).tolist()
    use_plugins(monkeypatch, "recording_robot")
    assert_runs_without_torch(
        ROLLOUT,
        str(tmp_path / "robot.jsonl"),
        "12",
        f"--bundle.path={bundle.path}",
        "--robot.type=recording_robot",
        f"--robot.frames={frames}",
        f"--robot.n_motors={len(state)}",
        f"--robot.static_values={state}",
        "--fps=50",
        "--duration=2",
        "--return_to_initial_position=false",
        "--status_interval_s=0",
    )
