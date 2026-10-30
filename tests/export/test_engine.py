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

"""The engine that runs a bundle, on bundles of small modules run by the test runtime."""

import json
import os
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file, save_file
from torch import Tensor, nn

from lerobot.bundle.engine import BundleEngine
from lerobot.bundle.manifest import MANIFEST, NOISE, TEST_CASE, Bundle
from lerobot.bundle.runtime import RUNTIMES
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.export.torch_backend import make_bundle, record_places

FRAME = {OBS_STATE: np.array([1.0, 2.0, 3.0], np.float32)}
REPO = Path(__file__).parents[2]


class Chunk(nn.Module):
    """Three actions from a state: the state scaled by 1, 2 and 3."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([[1.0], [2.0], [3.0]]))

    def forward(self, state: Tensor) -> Tensor:
        return state[None] * self.scale


class CameraChunk(Chunk):
    """The chunk of `Chunk`, from a frame that also holds a camera image."""

    def forward(self, state: Tensor, image: Tensor) -> Tensor:
        return super().forward(state)


class NoisyChunk(Chunk):
    """The chunk of `Chunk`, plus the starting noise."""

    def forward(self, state: Tensor, noise: Tensor) -> Tensor:
        return super().forward(state) + noise


def _edit(bundle: Bundle, **changes) -> None:
    manifest = json.loads((bundle.path / MANIFEST).read_text())
    (bundle.path / MANIFEST).write_text(json.dumps({**manifest, **changes}))


def _edit_test_case(bundle: Bundle, **changes) -> None:
    case = load_file(bundle.path / TEST_CASE)
    save_file({**case, **changes}, bundle.path / TEST_CASE)


def _chunk_bundle(folder: Path) -> Bundle:
    return make_bundle(folder, Chunk(), FRAME)


def _noisy_bundle(folder: Path) -> Bundle:
    return make_bundle(folder, NoisyChunk(), FRAME, noise_shape=(3, 3))


def test_engine_plays_a_chunk_then_runs_again(tmp_path):
    engine = BundleEngine(_chunk_bundle(tmp_path / "bundle"))
    actions = [engine.get_action({OBS_STATE: np.ones(3, np.float32)}) for _ in range(4)]
    np.testing.assert_array_equal(np.stack(actions), [[1, 1, 1], [2, 2, 2], [3, 3, 3], [1, 1, 1]])


def test_engine_measures_the_difference_in_normalized_action_units(tmp_path):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME, expected_offset=1.0)
    _edit(bundle, action_scale=[100.0, 100.0, 100.0], tolerance=0.05)
    BundleEngine(Bundle.read(bundle.path))
    _edit(bundle, action_scale=[100.0, 100.0, 10.0])
    with pytest.raises(RuntimeError, match=r"does not reproduce its test case \(1.00e-01 > 0.05\)"):
        BundleEngine(Bundle.read(bundle.path))


def test_the_bundle_check_imports_nothing_the_bundle_ships(tmp_path):
    bundle = _chunk_bundle(tmp_path / "bundle")
    _edit(bundle, runtime_imports=["payload"])
    ran = tmp_path / "ran"
    (bundle.path / "payload.py").write_text(f"open({str(ran)!r}, 'w').close()\n")
    # From inside the bundle folder, which `python -m` puts on the module search path.
    result = subprocess.run(
        [sys.executable, "-m", "lerobot.bundle", "."],
        cwd=bundle.path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert not ran.exists()
    assert "needs the Python module 'payload'" in result.stderr


def test_the_check_command_imports_nothing_from_the_folder_it_starts_in(tmp_path):
    """Unlike `python -m`, which imports `lerobot` and the standard library from that folder first."""
    module, function = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"][
        "lerobot-check-bundle"
    ].split(":")
    bundle = _chunk_bundle(tmp_path / "bundle")
    ran = tmp_path / "ran"
    for name in ("csv", "numpy", "safetensors"):
        (bundle.path / f"{name}.py").write_text(f"open({str(ran)!r}, 'w').close()\n")
    # The script an installer writes for the command; Python puts its folder, not the current one, on the path.
    command = tmp_path / "bin" / "lerobot-check-bundle"
    command.parent.mkdir()
    command.write_text(f"import sys\nfrom {module} import {function}\nsys.exit({function}())\n")
    result = subprocess.run(
        [sys.executable, str(command), "."],
        cwd=bundle.path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])),
        },
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert not ran.exists()
    assert result.returncode == 0, result.stderr


def test_engine_feeds_the_test_case_noise_and_samples_its_own_after(tmp_path):
    bundle = _noisy_bundle(tmp_path / "bundle")
    engine = BundleEngine(bundle)
    first = np.stack([engine.get_action(FRAME) for _ in range(3)])
    second = np.stack([engine.get_action(FRAME) for _ in range(3)])
    assert not np.array_equal(first, second)
    _edit_test_case(bundle, **{NOISE: np.zeros((3, 3), np.float32)})
    with pytest.raises(RuntimeError, match="does not reproduce its test case"):
        BundleEngine(bundle)


class DrawNoise(nn.Module):
    def forward(self) -> Tensor:
        return torch.randn(3, 3)


def test_engine_runs_the_noise_draw_and_checks_with_the_recorded_noise(tmp_path, monkeypatch):
    """A GPU bundle draws its noise in a run: a chunk then copies only the frame."""
    bundle = _noisy_bundle(tmp_path / "bundle")
    torch.export.save(torch.export.export(DrawNoise(), ()), bundle.path / "noise.pt2")
    manifest = json.loads((bundle.path / MANIFEST).read_text())
    _edit(
        bundle,
        programs={**manifest["programs"], NOISE: {"file": "noise.pt2", "data": None}},
        runs=[{"name": NOISE, "program": NOISE, "inputs": [], "outputs": [NOISE]}, *manifest["runs"]],
    )
    placed = record_places(monkeypatch)
    # The check passes only with the test case's noise, so the draw did not run for it.
    engine = BundleEngine(Bundle.read(bundle.path))
    loaded = len(placed)
    first = np.stack([engine.get_action(FRAME) for _ in range(3)])
    second = np.stack([engine.get_action(FRAME) for _ in range(3)])
    assert not np.array_equal(first, second)
    assert placed[loaded:] == [(3,)] * 2


def test_engine_applies_the_runtime_options_of_its_bundle(tmp_path, monkeypatch):
    applied = []
    monkeypatch.setitem(RUNTIMES, "torch_test", replace(RUNTIMES["torch_test"], configure=applied.append))
    bundle = _chunk_bundle(tmp_path / "bundle")
    _edit(bundle, runtime_options={"TestBackend": {"shared": True}})
    BundleEngine(Bundle.read(bundle.path))
    assert applied == [{"TestBackend": {"shared": True}}]


def test_engine_loads_the_data_file_a_program_names(tmp_path):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME, data="model.data")
    BundleEngine(bundle)
    (bundle.path / "model.data").unlink()
    with pytest.raises(FileNotFoundError, match="model.data"):
        BundleEngine(bundle)


def test_engine_refuses_non_finite_actions(tmp_path):
    engine = BundleEngine(_chunk_bundle(tmp_path / "bundle"))
    with pytest.raises(RuntimeError, match="NaN or infinite"):
        engine.get_action({OBS_STATE: np.array([np.nan, 0.0, 0.0], np.float32)})


def test_engine_runs_only_the_task_it_was_exported_for(tmp_path):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME, task="pick")
    with pytest.raises(ValueError, match="runs only the task"):
        BundleEngine(bundle, task="place")
    engine = BundleEngine(bundle)
    assert engine.task == "pick"
    with pytest.raises(AttributeError):
        engine.task = "place"


def test_engine_permits_no_action_once_stopped(tmp_path):
    stop = Event()
    engine = BundleEngine(_chunk_bundle(tmp_path / "bundle"), stop_event=stop)
    assert engine.dispatch_allowed()
    stop.set()
    assert not engine.dispatch_allowed()


@pytest.mark.parametrize(
    ("action_names", "rename_map", "shapes", "names", "error"),
    [
        (None, {}, {OBS_STATE: (4,)}, ["a", "b", "c"], r"\(3,\) \(the robot gives \(4,\)\)"),
        (None, {}, {OBS_STATE: (3,)}, ["a", "b"], "3 action components"),
        (["a", "b", "c"], {}, {OBS_STATE: (3,)}, ["x", "y", "z"], r"the actions \['a', 'b', 'c'\]"),
        (None, {"a": OBS_STATE}, {"a": (3,), OBS_STATE: (3,)}, ["a", "b", "c"], "several of the robot's"),
    ],
)
def test_engine_refuses_a_robot_that_does_not_match(tmp_path, action_names, rename_map, shapes, names, error):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME, action_names=action_names)
    engine = BundleEngine(bundle, rename_map=rename_map)
    with pytest.raises(ValueError, match=error):
        engine.check_robot(shapes, names)


@pytest.mark.parametrize("dtype", [np.float32, np.uint16])
def test_engine_refuses_a_camera_frame_of_another_dtype(tmp_path, dtype):
    camera = "observation.images.front"
    image = np.zeros((4, 6, 3), np.uint8)
    engine = BundleEngine(make_bundle(tmp_path / "bundle", CameraChunk(), {**FRAME, camera: image}))
    engine.check_frame({**FRAME, camera: image})
    with pytest.raises(ValueError, match=rf"takes {camera} as uint8 \(the robot gives {np.dtype(dtype)}\)"):
        engine.check_frame({**FRAME, camera: image.astype(dtype)})


def test_engine_renames_the_robot_frame_for_the_check_and_the_run(tmp_path):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME, action_names=["a.pos", "b.pos", "c.pos"])
    engine = BundleEngine(bundle, rename_map={"observation.joints": OBS_STATE})
    engine.check_robot({"observation.joints": (3,)}, ["a.pos", "b.pos", "c.pos"])
    np.testing.assert_array_equal(
        engine.get_action({"observation.joints": np.ones(3, np.float32)}), [1, 1, 1]
    )


class Scale(nn.Module):
    def forward(self, x: Tensor, factor: Tensor) -> Tensor:
        return x * factor


def _repeat_bundle(folder: Path, chunk_inputs=("scaled", "chunk_factor")) -> Bundle:
    """`Chunk`, then one `Scale` program run twice with factors from the constants."""
    bundle = _chunk_bundle(folder)
    torch.export.save(
        torch.export.export(Scale(), (torch.ones(3, 3), torch.ones(1))), bundle.path / "scale.pt2"
    )
    save_file(
        {"factor_0": np.array([2.0], np.float32), "factor_1": np.array([0.5], np.float32)},
        bundle.path / "constants.safetensors",
    )
    runs = [
        {"name": "model", "program": "model", "inputs": [OBS_STATE], "outputs": ["chunk"]},
        {"name": "scale_0", "program": "scale", "inputs": ["chunk", "factor_0"], "outputs": ["scaled"]},
        {"name": "scale_1", "program": "scale", "inputs": list(chunk_inputs), "outputs": [ACTION]},
    ]
    programs = {"model": {"file": "model.pt2", "data": None}, "scale": {"file": "scale.pt2", "data": None}}
    _edit(bundle, programs=programs, runs=runs, constants="constants.safetensors")
    return Bundle.read(bundle.path)


def test_engine_runs_a_program_several_times_with_constants(tmp_path, monkeypatch):
    from tests.export import torch_backend

    loads = []
    init = torch_backend.TorchProgram.__init__
    monkeypatch.setattr(
        torch_backend.TorchProgram,
        "__init__",
        lambda self, path, data=None: loads.append(path.name) or init(self, path, data),
    )
    engine = BundleEngine(_repeat_bundle(tmp_path / "bundle", ("scaled", "factor_1")))
    assert loads == ["model.pt2", "scale.pt2"]
    np.testing.assert_array_equal(engine.get_action({OBS_STATE: np.ones(3, np.float32)}), [1, 1, 1])


def test_engine_places_the_constants_at_load_and_a_frame_value_once_per_chunk(tmp_path, monkeypatch):
    """Two runs read the state; the engine copies it once per chunk."""
    bundle = _repeat_bundle(tmp_path / "bundle")
    torch.export.save(
        torch.export.export(Scale(), (torch.ones(3, 3), torch.ones(3))), bundle.path / "scale_state.pt2"
    )
    runs = [
        {"name": "model", "program": "model", "inputs": [OBS_STATE], "outputs": ["chunk"]},
        {"name": "scale", "program": "scale", "inputs": ["chunk", "factor_0"], "outputs": ["scaled"]},
        {"name": "by_state", "program": "by_state", "inputs": ["scaled", OBS_STATE], "outputs": [ACTION]},
    ]
    programs = {
        "model": {"file": "model.pt2", "data": None},
        "scale": {"file": "scale.pt2", "data": None},
        "by_state": {"file": "scale_state.pt2", "data": None},
    }
    _edit(bundle, programs=programs, runs=runs)
    state = FRAME[OBS_STATE]
    _edit_test_case(bundle, **{ACTION: 2 * np.outer([1.0, 2.0, 3.0], state) * state})
    placed = record_places(monkeypatch)
    engine = BundleEngine(Bundle.read(bundle.path))
    for _ in range(2):
        engine.reset()
        engine.get_action(dict(FRAME))
    # The one constant a run reads, at load; then the state once for the load check and once per chunk.
    assert placed == [(1,), (3,), (3,), (3,)]


def test_engine_refuses_a_live_chunk_of_the_wrong_width(tmp_path, monkeypatch):
    engine = BundleEngine(_chunk_bundle(tmp_path / "bundle"))
    program = engine._runs[-1][1]
    monkeypatch.setattr(program, "to_numpy", lambda value: np.ones((3, 2), np.float32))
    with pytest.raises(RuntimeError, match=r"shaped \(3, 2\), not \[execution_steps, 3\]"):
        engine.get_action(FRAME)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"format_version": 2}, "format version 2.*Upgrade LeRobot"),
        ({"format_version": 0}, "format version 0.*Export the policy again"),
        ({"steps": ["model"]}, r"\['steps'\].*Upgrade LeRobot"),
        ({"programs": {"model": {"file": "model.pt2", "files": []}}}, r"\['files'\].*Upgrade LeRobot"),
        (
            {"runs": [{"name": "model", "program": "model", "inputs": [], "outputs": [], "device": "cuda"}]},
            r"\['device'\].*Upgrade LeRobot",
        ),
        (
            {"runs": [{"name": "model", "program": "other", "inputs": [], "outputs": []}]},
            r"runs the programs \['other'\]",
        ),
        ({"tolerance": float("nan")}, "tolerance must be a finite number"),
        ({"tolerance": float("inf")}, "tolerance must be a finite number"),
        ({"tolerance": -1.0}, "tolerance must be a finite number"),
    ],
)
def test_bundle_refuses_a_manifest_it_cannot_run(tmp_path, change, error):
    bundle = _chunk_bundle(tmp_path / "bundle")
    _edit(bundle, **change)
    with pytest.raises(ValueError, match=error):
        Bundle.read(bundle.path)


@pytest.mark.parametrize(
    ("make", "change", "case", "error"),
    [
        (_chunk_bundle, {"action_scale": [1e-100, 1.0, 1.0]}, {}, "does not fit in 32-bit floats"),
        (
            _chunk_bundle,
            {},
            {ACTION: np.array([[np.nan, 0.0, 0.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]], np.float32)},
            "no valid expected actions",
        ),
        (_chunk_bundle, {}, {ACTION: np.array([1.0, 2.0, 3.0], np.float32)}, "no valid expected actions"),
        (_chunk_bundle, {}, {ACTION: np.zeros((3, 2), np.float32)}, "no valid expected actions"),
        (_noisy_bundle, {}, {NOISE: np.zeros((1, 3), np.float32)}, r"noise shaped \(1, 3\).*\(3, 3\)"),
        (_repeat_bundle, {}, {}, r"'scale_1' reads \['chunk_factor'\]"),
        (
            _repeat_bundle,
            {"runs": [{"name": "model", "program": "model", "inputs": [OBS_STATE], "outputs": ["factor_1"]}]},
            {},
            r"writes the constants \['factor_1'\]",
        ),
    ],
)
def test_engine_refuses_a_bundle_it_cannot_run(tmp_path, make, change, case, error):
    bundle = make(tmp_path / "bundle")
    _edit(bundle, **change)
    _edit_test_case(bundle, **case)
    with pytest.raises(ValueError, match=error):
        BundleEngine(Bundle.read(bundle.path))
