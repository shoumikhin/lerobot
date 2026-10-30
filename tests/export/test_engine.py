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
from threading import Event

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file, save_file
from torch import Tensor, nn

from lerobot.bundle.engine import BundleEngine
from lerobot.bundle.manifest import MANIFEST, NOISE, TEST_CASE, Bundle
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.export.torch_backend import make_bundle

FRAME = {OBS_STATE: np.array([1.0, 2.0, 3.0], np.float32)}


class Chunk(nn.Module):
    """Three actions from a state: the state scaled by 1, 2 and 3."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([[1.0], [2.0], [3.0]]))

    def forward(self, state: Tensor) -> Tensor:
        return state[None] * self.scale


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


def test_engine_plays_a_chunk_then_runs_again(tmp_path):
    engine = BundleEngine(make_bundle(tmp_path / "bundle", Chunk(), FRAME))
    actions = [engine.get_action({OBS_STATE: np.ones(3, np.float32)}) for _ in range(4)]
    np.testing.assert_array_equal(np.stack(actions), [[1, 1, 1], [2, 2, 2], [3, 3, 3], [1, 1, 1]])


def test_engine_refuses_a_bundle_that_misses_its_test_case(tmp_path):
    with pytest.raises(RuntimeError, match="does not reproduce its test case"):
        BundleEngine(make_bundle(tmp_path / "bundle", Chunk(), FRAME, expected_offset=1.0))


def test_engine_measures_the_difference_in_normalized_action_units(tmp_path):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME, expected_offset=1.0)
    _edit(bundle, action_scale=[100.0, 100.0, 100.0], tolerance=0.05)
    BundleEngine(Bundle.read(bundle.path))
    _edit(bundle, action_scale=[100.0, 100.0, 10.0])
    with pytest.raises(RuntimeError, match="1.00e-01 > 0.05"):
        BundleEngine(Bundle.read(bundle.path))


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1.0])
def test_bundle_refuses_a_tolerance_that_passes_anything(tmp_path, tolerance):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME)
    _edit(bundle, tolerance=tolerance)
    with pytest.raises(ValueError, match="tolerance must be a finite number"):
        Bundle.read(bundle.path)


@pytest.mark.parametrize(
    "expected",
    [
        np.array([[np.nan, 0.0, 0.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]], np.float32),
        np.array([1.0, 2.0, 3.0], np.float32),
        np.zeros((3, 2), np.float32),
    ],
)
def test_engine_refuses_a_test_case_without_valid_actions(tmp_path, expected):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME)
    _edit_test_case(bundle, **{ACTION: expected})
    with pytest.raises(ValueError, match="no valid expected actions"):
        BundleEngine(bundle)


def test_engine_feeds_the_test_case_noise_and_samples_its_own_after(tmp_path):
    bundle = make_bundle(tmp_path / "bundle", NoisyChunk(), FRAME, noise_shape=(3, 3))
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
    from tests.export import torch_backend

    bundle = make_bundle(tmp_path / "bundle", NoisyChunk(), FRAME, noise_shape=(3, 3))
    torch.export.save(torch.export.export(DrawNoise(), ()), bundle.path / "noise.pt2")
    manifest = json.loads((bundle.path / MANIFEST).read_text())
    _edit(
        bundle,
        programs={**manifest["programs"], NOISE: {"file": "noise.pt2", "data": None}},
        runs=[{"name": NOISE, "program": NOISE, "inputs": [], "outputs": [NOISE]}, *manifest["runs"]],
    )
    placed = []
    place = torch_backend.TorchProgram.place
    monkeypatch.setattr(
        torch_backend.TorchProgram,
        "place",
        lambda self, index, value: placed.append(value.shape) or place(self, index, value),
    )
    # The check passes only with the test case's noise, so the draw did not run for it.
    engine = BundleEngine(Bundle.read(bundle.path))
    loaded = len(placed)
    first = np.stack([engine.get_action(FRAME) for _ in range(3)])
    second = np.stack([engine.get_action(FRAME) for _ in range(3)])
    assert not np.array_equal(first, second)
    assert placed[loaded:] == [(3,)] * 2


def test_engine_loads_the_data_file_a_program_names(tmp_path):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME, data="model.data")
    BundleEngine(bundle)
    (bundle.path / "model.data").unlink()
    with pytest.raises(FileNotFoundError, match="model.data"):
        BundleEngine(bundle)


def test_engine_refuses_non_finite_actions(tmp_path):
    engine = BundleEngine(make_bundle(tmp_path / "bundle", Chunk(), FRAME))
    with pytest.raises(RuntimeError, match="NaN or infinite"):
        engine.get_action({OBS_STATE: np.array([np.nan, 0.0, 0.0], np.float32)})


def test_engine_runs_only_the_task_it_was_exported_for(tmp_path):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME, task="pick")
    with pytest.raises(ValueError, match="runs only the task"):
        BundleEngine(bundle, task="place")
    engine = BundleEngine(bundle)
    assert engine.task == "pick"


def test_engine_permits_no_action_once_stopped(tmp_path):
    stop = Event()
    engine = BundleEngine(make_bundle(tmp_path / "bundle", Chunk(), FRAME), stop_event=stop)
    assert engine.dispatch_allowed()
    stop.set()
    assert not engine.dispatch_allowed()


def test_engine_refuses_a_robot_that_does_not_match(tmp_path):
    engine = BundleEngine(make_bundle(tmp_path / "bundle", Chunk(), FRAME))
    engine.check_robot({OBS_STATE: (3,)}, ["a", "b", "c"])
    with pytest.raises(ValueError, match=r"observation.state \(3,\) \(the robot gives \(4,\)\)"):
        engine.check_robot({OBS_STATE: (4,)}, ["a", "b", "c"])
    with pytest.raises(ValueError, match="3 action components"):
        engine.check_robot({OBS_STATE: (3,)}, ["a", "b"])


def test_engine_renames_the_robot_frame_for_the_check_and_the_run(tmp_path):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME)
    engine = BundleEngine(bundle, rename_map={"observation.joints": OBS_STATE})
    engine.check_robot({"observation.joints": (3,)}, ["a", "b", "c"])
    np.testing.assert_array_equal(
        engine.get_action({"observation.joints": np.ones(3, np.float32)}), [1, 1, 1]
    )


def test_engine_checks_the_joint_names_the_bundle_records(tmp_path):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME, action_names=["a.pos", "b.pos", "c.pos"])
    engine = BundleEngine(bundle)
    engine.check_robot({OBS_STATE: (3,)}, ["a.pos", "b.pos", "c.pos"])
    with pytest.raises(ValueError, match=r"the actions \['a.pos', 'b.pos', 'c.pos'\]"):
        engine.check_robot({OBS_STATE: (3,)}, ["x.pos", "y.pos", "z.pos"])


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
    ],
)
def test_bundle_refuses_a_manifest_it_cannot_run(tmp_path, change, error):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME)
    _edit(bundle, **change)
    with pytest.raises(ValueError, match=error):
        Bundle.read(bundle.path)


class Scale(nn.Module):
    def forward(self, x: Tensor, factor: Tensor) -> Tensor:
        return x * factor


def _repeat_bundle(tmp_path, chunk_inputs=("scaled", "chunk_factor")):
    """`Chunk`, then one `Scale` program run twice with factors from the constants."""
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME)
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
    engine = BundleEngine(_repeat_bundle(tmp_path, ("scaled", "factor_1")))
    assert loads == ["model.pt2", "scale.pt2"]
    np.testing.assert_array_equal(engine.get_action({OBS_STATE: np.ones(3, np.float32)}), [1, 1, 1])


def test_engine_places_the_constants_once_at_load(tmp_path, monkeypatch):
    from tests.export import torch_backend

    placed = []
    place = torch_backend.TorchProgram.place
    monkeypatch.setattr(
        torch_backend.TorchProgram,
        "place",
        lambda self, index, value: placed.append(value.tolist()) or place(self, index, value),
    )
    engine = BundleEngine(_repeat_bundle(tmp_path, ("scaled", "factor_1")))
    for _ in range(3):
        engine.reset()
        engine.get_action({OBS_STATE: np.ones(3, np.float32)})
    # The two factors once at load; the rest is the state, once for the load check and once per chunk.
    assert placed[:2] == [[2.0], [0.5]] and len(placed) == 2 + 1 + 3


def test_engine_places_a_frame_value_once_per_chunk(tmp_path, monkeypatch):
    """Two runs read the state; the engine copies it once per chunk."""
    from tests.export import torch_backend

    bundle = _repeat_bundle(tmp_path)
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
    placed = []
    place = torch_backend.TorchProgram.place
    monkeypatch.setattr(
        torch_backend.TorchProgram,
        "place",
        lambda self, index, value: placed.append(value.shape) or place(self, index, value),
    )
    engine = BundleEngine(Bundle.read(bundle.path))
    for _ in range(2):
        engine.reset()
        engine.get_action(dict(FRAME))
    # The one constant a run reads, at load; then the state once for the load check and once per chunk.
    assert placed == [(1,), (3,), (3,), (3,)]


def test_engine_refuses_a_program_that_reads_what_nothing_gives(tmp_path):
    with pytest.raises(ValueError, match=r"'scale_1' reads \['chunk_factor'\]"):
        BundleEngine(_repeat_bundle(tmp_path))


def test_engine_refuses_a_live_chunk_of_the_wrong_width(tmp_path, monkeypatch):
    engine = BundleEngine(make_bundle(tmp_path / "bundle", Chunk(), FRAME))
    program = engine._runs[-1][1]
    monkeypatch.setattr(program, "to_numpy", lambda value: np.ones((3, 2), np.float32))
    with pytest.raises(RuntimeError, match=r"shaped \(3, 2\), not \[execution_steps, 3\]"):
        engine.get_action(FRAME)


def test_engine_refuses_a_test_case_noise_of_another_shape(tmp_path):
    bundle = make_bundle(tmp_path / "bundle", NoisyChunk(), FRAME, noise_shape=(3, 3))
    _edit_test_case(bundle, **{NOISE: np.zeros((1, 3), np.float32)})
    with pytest.raises(ValueError, match=r"noise shaped \(1, 3\).*\(3, 3\)"):
        BundleEngine(bundle)


def test_engine_refuses_an_action_scale_that_float32_rounds_to_zero(tmp_path):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME)
    _edit(bundle, action_scale=[1e-100, 1.0, 1.0])
    with pytest.raises(ValueError, match="does not fit in 32-bit floats"):
        BundleEngine(Bundle.read(bundle.path))


def test_engine_refuses_a_rename_that_merges_two_observations(tmp_path):
    engine = BundleEngine(make_bundle(tmp_path / "bundle", Chunk(), FRAME), rename_map={"a": OBS_STATE})
    with pytest.raises(ValueError, match="several of the robot's observations"):
        engine.check_robot({"a": (3,), OBS_STATE: (3,)}, ["a", "b", "c"])


def test_engine_refuses_a_program_that_writes_a_constant(tmp_path):
    bundle = _repeat_bundle(tmp_path)
    manifest = bundle.manifest.to_dict()
    manifest["runs"][1]["outputs"] = ["factor_1"]
    _edit(bundle, runs=manifest["runs"])
    with pytest.raises(ValueError, match=r"writes the constants \['factor_1'\]"):
        BundleEngine(Bundle.read(bundle.path))
