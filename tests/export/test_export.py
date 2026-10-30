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

"""`lerobot-export` and the bundle it writes, end to end on a tiny ACT, with the test backend."""

import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file, save_file

from lerobot.bundle.engine import BundleEngine
from lerobot.bundle.manifest import POLICY_CONFIG, Bundle
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.export import worker
from lerobot.inference.sync import SyncInferenceEngine
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.scripts import lerobot_export
from lerobot.scripts.lerobot_export import export
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE
from tests.export.plugins import use_plugins
from tests.export.torch_backend import TorchTestBackend
from tests.torch_free import assert_runs_without_torch

IMAGE = "observation.images.front"


@pytest.fixture(scope="module", autouse=True)
def torch_test_backend():
    with pytest.MonkeyPatch.context() as monkeypatch:
        use_plugins(monkeypatch, "torch_backend")
        yield


def save_checkpoint(path, image: str) -> None:
    config = ACTConfig(
        input_features={
            image: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        chunk_size=4,
        n_action_steps=4,
        use_vae=False,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        pretrained_backbone_weights=None,
        device="cpu",
    )
    stats = {
        image: {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.2)},
        OBS_STATE: {"mean": torch.zeros(3), "std": torch.ones(3)},
        ACTION: {"mean": torch.full((3,), 10.0), "std": torch.full((3,), 30.0)},
    }
    ACTPolicy(config).save_pretrained(path)
    for processor in make_pre_post_processors(config, dataset_stats=stats):
        processor.save_pretrained(path)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("act")
    save_checkpoint(path, IMAGE)
    return path


def run_export(*args: str, backend: str = "torch_test") -> None:
    export([f"--backend.type={backend}", *args])


def robot_frame(image: str = IMAGE) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        image: rng.integers(0, 256, (32, 32, 3), dtype=np.uint8),
        OBS_STATE: rng.random(3, dtype=np.float32),
    }


def assert_plays_like_pytorch(checkpoint, bundle: Bundle, image: str) -> None:
    policy = ACTPolicy.from_pretrained(checkpoint)
    reference = SyncInferenceEngine(
        policy, *make_pre_post_processors(policy.config, checkpoint), {}, [], "", "cpu", ""
    )
    engine = BundleEngine(bundle)
    frame = robot_frame(image)
    scale = np.asarray(bundle.manifest.action_scale, dtype=np.float32)
    for _ in range(policy.config.n_action_steps + 1):
        expected = reference.get_action(dict(frame)).numpy()
        # A compiled backend sums in another order than PyTorch, so hold it to the tolerance the bundle was checked
        # with, in normalized action units, as the engine does.
        drift = float((np.abs(engine.get_action(dict(frame)) - expected) / scale).max())
        assert drift <= bundle.manifest.tolerance, drift


def test_export_takes_a_singular_image_key_a_task_and_a_backend_from_discover_packages_path(tmp_path):
    save_checkpoint(tmp_path / "act", OBS_IMAGE)
    # No installed package adds this backend, so each step must load it from the command line.
    run_export(
        f"--policy.path={tmp_path / 'act'}",
        f"--output_dir={tmp_path / 'bundle'}",
        "--task=pick",
        "--backend.discover_packages_path=tests.export.discovered_backend",
        backend="torch_test_discovered",
    )
    bundle = Bundle.read(tmp_path / "bundle")
    assert (bundle.manifest.backend, bundle.manifest.task) == ("torch_test_discovered", "pick")
    assert_plays_like_pytorch(tmp_path / "act", bundle, OBS_IMAGE)


# Robot units of the test checkpoint's actions, whose spread is 30: float32's measured drift between eager
# PyTorch and correct compilers (8.3e-6 normalized), and ten times that.
@pytest.mark.parametrize(("offset", "passes"), [(2.5e-4, True), (2.5e-3, False)])
def test_float32_tolerance_catches_a_tenfold_drift(checkpoint, tmp_path, offset, passes):
    args = (
        f"--policy.path={checkpoint}",
        f"--output_dir={tmp_path / 'bundle'}",
        f"--backend.offset={offset}",
    )
    if passes:
        run_export(*args)
    else:
        with pytest.raises(RuntimeError, match="failed its check"):
            run_export(*args)
        # Nothing is written, not even the folders the export made beside the output folder.
        assert list(tmp_path.iterdir()) == []


def test_a_backend_refuses_a_pytorch_older_than_it_needs(monkeypatch):
    monkeypatch.setattr(TorchTestBackend, "min_torch", "2.0")
    TorchTestBackend().check_torch()
    monkeypatch.setattr(TorchTestBackend, "min_torch", "99.0")
    with pytest.raises(ValueError, match="needs PyTorch 99.0 or newer to export") as refusal:
        TorchTestBackend().check_torch()
    assert "torch-overrides.txt" not in str(refusal.value)

    monkeypatch.setattr(TorchTestBackend, "packages", ("compiler>=99.0,<99.1",))
    monkeypatch.setattr(TorchTestBackend, "torch_overrides", ("torch>=99.0,<99.1", "torchvision>=9.0,<9.1"))
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ValueError) as refusal:
        TorchTestBackend().check_torch()
    message = str(refusal.value)
    assert "printf 'torch>=99.0,<99.1\\ntorchvision>=9.0,<9.1\\n' > torch-overrides.txt" in message
    assert 'uv pip install --override torch-overrides.txt lerobot "compiler>=99.0,<99.1"' in message

    # LeRobot pins an older PyTorch on macOS only.
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(ValueError) as refusal:
        TorchTestBackend().check_torch()
    message = str(refusal.value)
    assert 'pip install "torch>=99.0,<99.1" "torchvision>=9.0,<9.1" "compiler>=99.0,<99.1"' in message
    assert "torch-overrides.txt" not in message


@pytest.mark.parametrize("tolerance", ["nan", "inf", "-1"])
def test_export_refuses_a_tolerance_that_passes_anything(checkpoint, tmp_path, tolerance, capfd):
    with pytest.raises(RuntimeError, match="trace step failed"):
        run_export(
            f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}", f"--tolerance={tolerance}"
        )
    assert "--tolerance must be a finite number" in capfd.readouterr().err


def test_export_refuses_a_checkpoint_that_leaves_a_weight_unfilled(checkpoint, tmp_path, capfd):
    partial = tmp_path / "partial"
    shutil.copytree(checkpoint, partial)
    weights = load_file(partial / "model.safetensors")
    del weights["model.action_head.weight"]
    save_file(weights, partial / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="trace step failed"):
        run_export(f"--policy.path={partial}", f"--output_dir={tmp_path / 'b'}")
    assert "model.action_head.weight" in capfd.readouterr().err


def test_the_trace_step_loads_the_requested_revision_without_tf32(checkpoint, tmp_path, monkeypatch):
    import lerobot.policies.factory as factory

    revisions, tf32 = [], []
    load, make = factory.load_pretrained_policy, factory.make_pre_post_processors

    def load_pretrained_policy(config, **kwargs):
        revisions.append(config.pretrained_revision)
        tf32.append((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
        return load(config, **kwargs)

    def make_pre_post_processors(*args, **kwargs):
        revisions.append(kwargs["pretrained_revision"])
        return make(*args, **kwargs)

    monkeypatch.setattr(factory, "load_pretrained_policy", load_pretrained_policy)
    monkeypatch.setattr(factory, "make_pre_post_processors", make_pre_post_processors)
    args = [
        "--backend.type=torch_test",
        f"--policy.path={checkpoint}",
        f"--output_dir={tmp_path / 'b'}",
        "--policy.pretrained_revision=pin123",
    ]
    request = tmp_path / "trace.json"
    folders, result = str(tmp_path / "folders.json"), str(tmp_path / "result.json")
    request.write_text(json.dumps({"step": "trace", "args": args, "folders": folders, "result": result}))
    argv = ["worker", str(request)]
    monkeypatch.setattr(sys, "argv", list(argv))
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    try:
        worker.main()
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
    assert revisions == ["pin123", "pin123"] and tf32 == [(False, False)]
    # Later configs in this process read the command line, so the step gives it back.
    assert sys.argv == argv


def test_the_compile_step_runs_without_autograd(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        "tests.export.torch_backend.TorchTestBackend.compile",
        lambda self, program, path: seen.append(torch.is_grad_enabled()),
    )
    with torch.no_grad():
        program = torch.export.export(torch.nn.Linear(2, 2), (torch.zeros(1, 2),))
    torch.export.save(program, tmp_path / "part.pt2")
    request = {"backend": {"type": "torch_test"}, "program": str(tmp_path / "part.pt2"), "plugins": []}
    worker.compile_step({**request, "path": str(tmp_path / "out.pt2")})
    assert seen == [False]


def test_a_killed_trace_step_leaves_no_folders_behind(tmp_path, monkeypatch):
    made = [tmp_path / ".b-1", tmp_path / ".b-traced-1"]

    def killed_trace(args):
        request = json.loads(Path(args[-1]).read_text())
        for folder in made:
            folder.mkdir()
        Path(request["folders"]).write_text(json.dumps([str(folder) for folder in made]))
        return subprocess.CompletedProcess(args, -9)

    monkeypatch.setattr(lerobot_export.subprocess, "run", killed_trace)
    with pytest.raises(RuntimeError, match="trace step crashed with SIGKILL"):
        run_export(f"--output_dir={tmp_path / 'b'}")
    assert not [folder for folder in made if folder.exists()]


def test_export_names_the_signal_that_killed_the_check(checkpoint, tmp_path, monkeypatch):
    run = subprocess.run
    monkeypatch.setattr(
        lerobot_export.subprocess,
        "run",
        lambda args: subprocess.CompletedProcess(args, -11) if "lerobot.bundle" in args else run(args),
    )
    with pytest.raises(RuntimeError, match="crashed with SIGSEGV"):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}")


BACKENDS = ["torch_test"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_recipe_runs_a_part_several_times_with_constants(checkpoint, tmp_path, monkeypatch, backend):
    """ACT exported as a chain whose last part runs twice, reading its factor from the constants."""
    use_plugins(monkeypatch, "chain_recipe")
    run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}", backend=backend)
    bundle = Bundle.read(tmp_path / "b")
    assert_plays_like_pytorch(checkpoint, bundle, IMAGE)
    manifest = bundle.manifest
    assert [(run.name, run.program) for run in manifest.runs] == [
        ("prepare", "prepare"),
        ("model", "model"),
        ("scale_0", "scale"),
        ("scale_1", "scale"),
    ]
    assert sorted(manifest.programs) == ["model", "prepare", "scale"]
    constants = load_file(bundle.path / bundle.manifest.constants)
    np.testing.assert_allclose([constants["factor_0"][0], constants["factor_1"][0]], [2.0, 0.5])


EXPORT_WITHOUT_TORCH = """
import sys
import tempfile
from lerobot.scripts.lerobot_export import export

export([*sys.argv[2:], f"--output_dir={tempfile.mkdtemp(dir=sys.argv[1])}/b"])
"""


def test_export_writes_a_bundle_that_plays_like_pytorch_without_importing_torch(checkpoint, tmp_path):
    """Only the steps import PyTorch, so what one step used is freed before the next starts."""
    args = ["--backend.type=torch_test", f"--policy.path={checkpoint}"]
    assert_runs_without_torch(EXPORT_WITHOUT_TORCH, str(tmp_path), *args)
    runs = list(tmp_path.iterdir())
    (tmp_path / "new").mkdir()
    for run in runs:
        assert sorted(path.name for path in run.iterdir()) == ["b"]
        bundle = Bundle.read(run / "b")
        np.testing.assert_allclose(bundle.manifest.action_scale, [30.0, 30.0, 30.0], rtol=1e-6)
        assert (bundle.path / POLICY_CONFIG).is_file() and not (bundle.path / "config.json").exists()
        # The permissions of a new folder, not the private ones of the temporary folder the bundle was.
        assert stat.S_IMODE(bundle.path.stat().st_mode) == stat.S_IMODE((tmp_path / "new").stat().st_mode)
        assert_plays_like_pytorch(checkpoint, bundle, IMAGE)
