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

import importlib
import io
import json
import pickle
import subprocess
import sys
import tomllib
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file, save_file
from torch import Tensor, nn

from lerobot.bundle.engine import BundleEngine
from lerobot.bundle.manifest import MANIFEST, NOISE, TEST_CASE, Bundle, BundleManifest
from lerobot.bundle.runtime import RUNTIMES
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.export.torch_backend import TorchProgram, make_bundle, record_places

FRAME = {OBS_STATE: np.array([1.0, 2.0, 3.0], np.float32)}
TASK_TOKENS = {
    "prompt_ops": [],
    "tokenizer": "tokenizer.json",
    "embeddings": "embeddings.safetensors",
    "known_tasks": "known_tasks.json",
    "max_tokens": 4,
}
ACTION_TOKENS = {
    "tokenizer": "fast.json",
    "clean_up_spaces": False,
    "id_offset": 0,
    "end": 1,
    "prefix": [],
    "start": [],
    "min_token": 0,
    "scale": 1.0,
    "horizon": 2,
    "action_dim": 3,
}
MODEL_RUN = {"name": "model", "program": "model", "inputs": [OBS_STATE], "outputs": [ACTION]}


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


def _chunk_bundle(folder: Path) -> Bundle:
    return make_bundle(folder, Chunk(), FRAME)


def _noisy_bundle(folder: Path) -> Bundle:
    return make_bundle(folder, NoisyChunk(), FRAME, noise_shape=(3, 3))


def test_engine_plays_a_chunk_then_runs_again(tmp_path):
    engine = BundleEngine(_chunk_bundle(tmp_path / "bundle"))
    # No frame yet, so the control loop holds; no chunk runs.
    assert engine.get_action(None) is None
    actions = [engine.get_action({OBS_STATE: np.ones(3, np.float32)}) for _ in range(4)]
    np.testing.assert_array_equal(np.stack(actions), [[1, 1, 1], [2, 2, 2], [3, 3, 3], [1, 1, 1]])


def test_engine_measures_the_difference_in_normalized_action_units(tmp_path):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME, expected_offset=1.0)
    _edit(bundle, action_scale=[100.0, 100.0, 100.0], tolerance=0.05)
    BundleEngine(Bundle.read(bundle.path))
    _edit(bundle, action_scale=[100.0, 100.0, 10.0])
    with pytest.raises(RuntimeError, match=r"does not reproduce its test case \(1.00e-01 > 0.05\)"):
        BundleEngine(Bundle.read(bundle.path))


def test_engine_refuses_a_test_case_with_another_number_of_actions(tmp_path):
    # Every action is 0, so one expected action broadcast against the three would match them.
    bundle = _chunk_bundle(tmp_path / "bundle")
    _edit_test_case(bundle, **{OBS_STATE: np.zeros(3, np.float32), ACTION: np.zeros((1, 3), np.float32)})
    with pytest.raises(RuntimeError, match=r"shaped \(3, 3\), but its test case expects \(1, 3\)"):
        BundleEngine(Bundle.read(bundle.path))


ANY_HOST = (
    "from lerobot.bundle.engine import BundleEngine\nfrom lerobot.bundle.manifest import Bundle\n"
    "BundleEngine(Bundle.read({!r}))"
)


@pytest.mark.parametrize(
    "command, shipped",
    [
        (["-m", "lerobot.bundle", "."], ["payload", "numpy"]),
        # As the installed command starts: without the current folder on the module search path.
        (["-P", "-m", "lerobot.scripts.lerobot_check_bundle", "."], ["payload", "numpy"]),
        # Another host imports its own modules from where it starts, but none that the bundle names.
        (["-c", ANY_HOST.format(".")], ["payload"]),
    ],
    ids=["bundle", "check-command", "any-host"],
)
def test_the_bundle_check_imports_nothing_the_bundle_ships(tmp_path, command, shipped):
    bundle = _chunk_bundle(tmp_path / "bundle")
    _edit(bundle, runtime_imports=["payload"])
    for name in shipped:
        (bundle.path / f"{name}.py").write_text(f"open({str(tmp_path / name)!r}, 'w').close()\n")
    # From inside the bundle folder, which `python -m` and `python -c` put on the module search path.
    result = subprocess.run(
        [sys.executable, *command],
        cwd=bundle.path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert not [name for name in shipped if (tmp_path / name).exists()]
    assert "could not import the Python module 'payload'" in result.stderr


def test_a_bundle_loaded_inside_its_folder_runs_no_executorch_it_ships(tmp_path):
    """ExecuTorch's bindings import while its runtime gets ready, before any module the manifest names."""
    bundle = _chunk_bundle(tmp_path / "bundle")
    _edit(bundle, runtime="executorch", runtime_imports=[])
    ran = tmp_path / "ran"
    (bundle.path / "executorch").mkdir()
    (bundle.path / "executorch" / "__init__.py").write_text("")
    (bundle.path / "executorch" / "runtime.py").write_text(f"open({str(ran)!r}, 'w').close()\n")
    subprocess.run(
        [sys.executable, "-c", ANY_HOST.format(".")], cwd=bundle.path, capture_output=True, timeout=300
    )
    assert not ran.exists()


def test_a_module_the_bundle_ships_is_refused_through_the_folder_above(tmp_path):
    bundle = _chunk_bundle(tmp_path / "bundle")
    _edit(bundle, runtime_imports=["bundle.payload"])
    ran = tmp_path / "ran"
    (bundle.path / "payload.py").write_text(f"open({str(ran)!r}, 'w').close()\n")
    # A script beside the bundle puts the folder above it on the module search path.
    result = subprocess.run(
        [sys.executable, "-c", ANY_HOST.format("bundle")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert not ran.exists()
    assert "the Python module 'bundle.payload', which would load from the bundle folder" in result.stderr


def test_the_bundle_check_prints_its_usage_and_names_a_folder_that_is_no_bundle(
    tmp_path, monkeypatch, capsys
):
    from lerobot.scripts import lerobot_check_bundle

    monkeypatch.setattr(sys, "argv", ["lerobot-check-bundle", "--help"])
    lerobot_check_bundle.main()
    assert capsys.readouterr().out == "usage: lerobot-check-bundle <bundle folder>\n"
    for arguments in ([], ["one", "two"]):
        monkeypatch.setattr(sys, "argv", ["lerobot-check-bundle", *arguments])
        with pytest.raises(SystemExit, match="usage: lerobot-check-bundle <bundle folder>"):
            lerobot_check_bundle.main()
    monkeypatch.setattr(sys, "argv", ["lerobot-check-bundle", str(tmp_path)])
    with pytest.raises(SystemExit, match="is not a bundle: it has no export.json"):
        lerobot_check_bundle.main()


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


def _draw_noise_in_a_run(bundle: Bundle, draw: nn.Module, inputs: tuple[str, ...] = ()) -> None:
    """Make the bundle draw its noise with `draw` in a run of its own, as a bundle on a CUDA device does."""
    torch.export.save(torch.export.export(draw, ()), bundle.path / "noise.pt2")
    manifest = json.loads((bundle.path / MANIFEST).read_text())
    _edit(
        bundle,
        programs={**manifest["programs"], NOISE: {"file": "noise.pt2", "data": None}},
        runs=[
            {"name": NOISE, "program": NOISE, "inputs": list(inputs), "outputs": [NOISE]},
            *manifest["runs"],
        ],
    )


def test_engine_runs_the_noise_draw_and_checks_with_the_recorded_noise(tmp_path, monkeypatch):
    """A bundle on a CUDA device draws its noise in a run: a chunk then copies only the frame."""
    bundle = _noisy_bundle(tmp_path / "bundle")
    _draw_noise_in_a_run(bundle, DrawNoise())
    placed = record_places(monkeypatch)
    # The check passes only with the test case's noise, so the draw did not run for it.
    engine = BundleEngine(Bundle.read(bundle.path))
    loaded = len(placed)
    first = np.stack([engine.get_action(FRAME) for _ in range(3)])
    second = np.stack([engine.get_action(FRAME) for _ in range(3)])
    assert not np.array_equal(first, second)
    assert placed[loaded:] == [(3,)] * 2


class ConstantNoise(nn.Module):
    def forward(self) -> Tensor:
        return torch.full((3, 3), 1000.0)


class ZeroNoise(nn.Module):
    def forward(self) -> Tensor:
        return torch.zeros(3, 3)


class WideNoise(nn.Module):
    def forward(self) -> Tensor:
        return torch.randn(3, 4)


class DoubleNoise(nn.Module):
    def forward(self) -> Tensor:
        return torch.randn(3, 3, dtype=torch.float64)


@pytest.mark.parametrize("draw", [ConstantNoise, ZeroNoise, WideNoise, DoubleNoise])
def test_engine_refuses_a_noise_draw_that_is_not_standard_normal_noise(tmp_path, draw):
    """The test case's noise stands in for the draw when the chunks replay, so the check runs it alone."""
    bundle = _noisy_bundle(tmp_path / "bundle")
    _draw_noise_in_a_run(bundle, draw())
    with pytest.raises(RuntimeError, match=r"not standard normal noise shaped \[3, 3\]"):
        BundleEngine(Bundle.read(bundle.path))


def test_engine_refuses_a_noise_draw_that_reads_a_value(tmp_path):
    bundle = _noisy_bundle(tmp_path / "bundle")
    _draw_noise_in_a_run(bundle, DrawNoise(), inputs=(OBS_STATE,))
    with pytest.raises(ValueError, match=r"draws the noise from \['observation.state'\]"):
        BundleEngine(Bundle.read(bundle.path))


def test_engine_refuses_a_noise_draw_when_the_bundle_gives_no_noise_shape(tmp_path):
    """Without a shape the draw cannot be checked, and a live draw would decide the test case."""
    bundle = _noisy_bundle(tmp_path / "bundle")
    _draw_noise_in_a_run(bundle, DrawNoise())
    case = load_file(bundle.path / TEST_CASE)
    del case[NOISE]
    save_file(case, bundle.path / TEST_CASE)
    _edit(bundle, noise_shape=None, tolerance=1e9)
    with pytest.raises(RuntimeError, match="not standard normal noise shaped None"):
        BundleEngine(Bundle.read(bundle.path))


def test_engine_refuses_a_noise_draw_that_writes_more_than_the_noise(tmp_path):
    bundle = _noisy_bundle(tmp_path / "bundle")
    manifest = json.loads((bundle.path / MANIFEST).read_text())
    _edit(
        bundle,
        programs={**manifest["programs"], NOISE: {"file": "noise.pt2", "data": None}},
        runs=[
            {"name": NOISE, "program": NOISE, "inputs": [], "outputs": [NOISE, "extra"]},
            *manifest["runs"],
        ],
    )
    with pytest.raises(ValueError, match=r"writes 'noise' and \['extra'\]"):
        BundleEngine(Bundle.read(bundle.path))


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


class _OpensAFile:
    """Unpickling it calls `open`, as a pickle can call anything it names."""

    def __init__(self, path: Path):
        self.path = path

    def __reduce__(self):
        return open, (str(self.path), "w")


def test_the_test_runtime_runs_no_code_from_a_program_file(tmp_path):
    program = torch.export.export(Chunk(), tuple(map(torch.from_numpy, FRAME.values())))
    torch.export.save(program, tmp_path / "a.pt2")
    pickled = io.BytesIO()
    torch.save(_OpensAFile(tmp_path / "ran"), pickled)
    with zipfile.ZipFile(tmp_path / "a.pt2") as source, zipfile.ZipFile(tmp_path / "b.pt2", "w") as target:
        for item in source.infolist():
            data = source.read(item)
            if item.filename.endswith("sample_inputs/model.pt"):
                data = pickled.getvalue()
            target.writestr(item, data)
    with pytest.raises(pickle.UnpicklingError):
        TorchProgram(tmp_path / "b.pt2")
    assert not (tmp_path / "ran").exists()


def test_the_test_runtime_refuses_a_bundle_whose_program_holds_an_object_constant(tmp_path):
    """PyTorch reads an object constant with plain pickle, even when weights-only loading is forced."""
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME)
    program = bundle.path / "model.pt2"
    with zipfile.ZipFile(program) as source:
        files = {item.filename: source.read(item) for item in source.infolist()}
    (config,) = (name for name in files if name.endswith("data/constants/model_constants_config.json"))
    constants = json.loads(files[config])
    constants["config"]["held"] = {
        "path_name": "opaque_obj_0",
        "is_param": False,
        "use_pickle": True,
        "tensor_meta": None,
    }
    files[config] = json.dumps(constants).encode()
    opaque = config.replace("model_constants_config.json", "opaque_obj_0")
    files[opaque] = pickle.dumps(_OpensAFile(tmp_path / "ran"))
    with zipfile.ZipFile(program, "w") as target:
        for name, data in files.items():
            target.writestr(name, data)
    with pytest.raises(ValueError, match="model.pt2 holds object constants"):
        BundleEngine(Bundle.read(bundle.path))
    assert not (tmp_path / "ran").exists()


def test_engine_runs_only_the_task_it_was_exported_for(tmp_path):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME, task="pick")
    with pytest.raises(ValueError, match="runs only the task"):
        BundleEngine(bundle, task="place")
    assert BundleEngine(bundle).task == "pick"


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


def _repeat_bundle(folder: Path, chunk_inputs=("scaled", "factor_1")) -> Bundle:
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
    loads = []
    init = TorchProgram.__init__
    monkeypatch.setattr(
        TorchProgram,
        "__init__",
        lambda self, path, data=None: loads.append(path.name) or init(self, path, data),
    )
    engine = BundleEngine(_repeat_bundle(tmp_path / "bundle"))
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
        ({"programs": {"model": {"file": "model.pt2", "shards": 2}}}, r"\['shards'\].*Upgrade LeRobot"),
        ({"runs": [{**MODEL_RUN, "device": "cuda"}]}, r"\['device'\].*Upgrade LeRobot"),
        (
            {"target": {"device": "cuda", "gpu": "A100", "compute_capability": "8.0", "driver": "580"}},
            r"\['driver'\].*Upgrade LeRobot",
        ),
        (
            {"runs": [{"name": "model", "program": "other", "inputs": [], "outputs": []}]},
            r"runs the programs \['other'\]",
        ),
        ({"tolerance": float("nan")}, "tolerance must be a finite number"),
        # An infinite tolerance would let any program pass its check.
        ({"tolerance": float("inf")}, "tolerance must be a finite number"),
        ({"tolerance": -1.0}, "tolerance must be a finite number of at least 0"),
    ],
)
def test_bundle_refuses_a_manifest_it_cannot_run(tmp_path, change, error):
    bundle = _chunk_bundle(tmp_path / "bundle")
    _edit(bundle, **change)
    with pytest.raises(ValueError, match=error):
        Bundle.read(bundle.path)


@pytest.mark.parametrize(
    ("change", "wrong"),
    [
        # A program named by its file alone would read as the unknown keys of its letters.
        ({"programs": {"model": "model.pt2"}}, "programs"),
        ({"runs": ["model"]}, "runs"),
        ({"runtime_imports": [123]}, "runtime_imports"),
        ({"runtime_options": "TensorRTBackend"}, "runtime_options"),
        ({"runtime_options": {"XnnpackBackend": 5}}, "runtime_options"),
        ({"runtime_options": {"XnnpackBackend": {"x": 1.5}}}, "runtime_options"),
        ({"target": "cuda"}, "target"),
        ({"target": {"device": "cuda", "gpu": "Orin", "compute_capability": 8.7}}, "target"),
        ({"runtime_versions": None}, "runtime_versions"),
        ({"runtime_versions": {"executorch": 1.6}}, "runtime_versions"),
        ({"exact_runtime_modules": "tensorrt"}, "exact_runtime_modules"),
    ],
)
def test_bundle_refuses_a_manifest_in_a_form_it_does_not_read(tmp_path, change, wrong):
    bundle = _chunk_bundle(tmp_path / "bundle")
    _edit(bundle, **change)
    with pytest.raises(ValueError, match=rf"gives \['{wrong}'\] in a form LeRobot does not read"):
        Bundle.read(bundle.path)


def test_bundle_refuses_a_manifest_that_lacks_a_field_it_must_have(tmp_path):
    bundle = _chunk_bundle(tmp_path / "bundle")
    manifest = json.loads((bundle.path / MANIFEST).read_text())
    del manifest["tolerance"]
    (bundle.path / MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"lacks \['tolerance'\]"):
        Bundle.read(bundle.path)


def test_a_manifest_leaves_out_the_fields_at_their_default(tmp_path):
    # So a field a later version adds, which this one refuses, stops it only for a bundle that sets the field.
    bundle = _chunk_bundle(tmp_path / "bundle")
    data = json.loads((bundle.path / MANIFEST).read_text())
    assert {"runtime_options", "first_runs", "target", "history", "carry", "test_case"}.isdisjoint(data)
    assert "data" not in data["programs"]["model"]
    assert data["format_version"] == 1
    assert BundleManifest.from_dict(data) == bundle.manifest


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
        (_noisy_bundle, {}, {NOISE: np.zeros((1, 3), np.float32)}, r"noise shaped \(1, 3\).*\(3, 3\)"),
        (
            lambda folder: _repeat_bundle(folder, ("scaled", "chunk_factor")),
            {},
            {},
            r"'scale_1' reads \['chunk_factor'\]",
        ),
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


def test_engine_refuses_runs_that_read_a_missing_value_before_any_program_loads(tmp_path):
    bundle = _chunk_bundle(tmp_path / "bundle")
    (bundle.path / "model.pt2").unlink()
    _edit(bundle, runs=[{**MODEL_RUN, "inputs": ["missing"]}])
    with pytest.raises(ValueError, match=r"reads \['missing'\]"):
        BundleEngine(Bundle.read(bundle.path))


@pytest.mark.parametrize("name", [OBS_STATE, NOISE])
def test_engine_refuses_a_constant_named_as_a_value_of_each_chunk(tmp_path, name):
    # Placed once at load, it would stand in for the robot's value or the drawn noise in every chunk.
    bundle = _repeat_bundle(tmp_path / "bundle")
    constants = load_file(bundle.path / "constants.safetensors")
    save_file({**constants, name: np.zeros(3, np.float32)}, bundle.path / "constants.safetensors")
    with pytest.raises(ValueError, match=rf"constants \['{name}'\] have the name of a value"):
        BundleEngine(Bundle.read(bundle.path))


def test_lerobot_check_bundle_is_the_installed_command():
    from lerobot.scripts import lerobot_check_bundle

    scripts = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())["project"]["scripts"]
    module, _, name = scripts["lerobot-check-bundle"].partition(":")
    assert getattr(importlib.import_module(module), name) is lerobot_check_bundle.main
