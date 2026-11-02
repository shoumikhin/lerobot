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

"""`lerobot-export` and the bundle it writes, end to end on a tiny ACT, with the test backend and, where they
are installed, real backends."""

import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from packaging.version import Version
from safetensors.numpy import load_file, save_file

from lerobot.bundle.engine import BundleEngine
from lerobot.bundle.manifest import NOISE, POLICY_CONFIG, Bundle, ProgramRun
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.export import worker
from lerobot.inference.sync import SyncInferenceEngine
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.scripts import lerobot_export
from lerobot.scripts.lerobot_export import export
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.utils.import_utils import is_package_available
from tests.export.plugins import use_plugins
from tests.export.torch_backend import TorchTestBackend
from tests.torch_free import CHECK_BUNDLE, assert_runs_without_torch

IMAGE = "observation.images.front"


@pytest.fixture(scope="module", autouse=True)
def torch_test_backend():
    """Add the test backend to the export's steps, which run in processes of their own."""
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
        # Held to the bundle's tolerance, in normalized action units, as the engine holds it, so that a backend
        # that sums in another order than PyTorch passes too.
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
    # A nightly or a release candidate of the version a backend needs counts as that version.
    monkeypatch.setattr(TorchTestBackend, "min_torch", "2.15")
    monkeypatch.setattr(
        "lerobot.export.configs.importlib.metadata.version", lambda name: "2.15.0.dev20260901"
    )
    TorchTestBackend().check_torch()
    monkeypatch.setattr("lerobot.export.configs.importlib.metadata.version", lambda name: "2.15.0rc1")
    TorchTestBackend().check_torch()
    monkeypatch.setattr(TorchTestBackend, "min_torch", "99.0")
    with pytest.raises(ValueError, match="needs PyTorch 99.0 or newer to export") as refusal:
        TorchTestBackend().check_torch()
    assert "pip install" not in str(refusal.value)

    monkeypatch.setattr(TorchTestBackend, "packages", ("compiler>=99.0,<99.1",))
    monkeypatch.setattr(TorchTestBackend, "torch_pins", ("torch>=99.0,<99.1", "torchvision>=9.0,<9.1"))
    with pytest.raises(ValueError) as refusal:
        TorchTestBackend().check_torch()
    message = str(refusal.value)
    assert 'pip install "torch>=99.0,<99.1" "torchvision>=9.0,<9.1" "compiler>=99.0,<99.1"' in message


def test_export_refuses_a_tolerance_that_passes_anything(checkpoint, tmp_path, capfd):
    with pytest.raises(RuntimeError, match="trace step failed"):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}", "--tolerance=nan")
    assert "--tolerance must be a finite number" in capfd.readouterr().err


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_executorch_coreml_is_refused_before_tracing_off_a_mac(monkeypatch, platform):
    from lerobot.export.backends.executorch_coreml import ExecuTorchCoreMLConfig

    monkeypatch.setattr(sys, "platform", platform)
    with pytest.raises(ValueError, match="executorch_coreml exports only on a Mac"):
        ExecuTorchCoreMLConfig()
    monkeypatch.setattr(sys, "platform", "darwin")
    ExecuTorchCoreMLConfig()


def test_export_refuses_a_checkpoint_that_leaves_a_weight_unfilled(checkpoint, tmp_path, capfd):
    partial = tmp_path / "partial"
    shutil.copytree(checkpoint, partial)
    weights = load_file(partial / "model.safetensors")
    del weights["model.action_head.weight"]
    save_file(weights, partial / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="trace step failed"):
        run_export(f"--policy.path={partial}", f"--output_dir={tmp_path / 'b'}")
    assert "model.action_head.weight" in capfd.readouterr().err


def test_export_loads_no_module_from_the_current_folder(checkpoint, tmp_path, monkeypatch):
    """Run from a folder holding .py files named like modules the steps and the bundle check import."""
    folder = tmp_path / "current"
    folder.mkdir()
    for name in ("csv", "draccus"):
        (folder / f"{name}.py").write_text(f"raise ImportError('{name}.py from the current folder')\n")
    monkeypatch.chdir(folder)
    run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}")
    assert Bundle.read(tmp_path / "b").manifest.backend == "torch_test"


@pytest.mark.parametrize("made", ["bundle", "broken link"])
def test_export_keeps_an_output_folder_made_while_it_ran(checkpoint, tmp_path, monkeypatch, made):
    """Another export to the same folder finished first, or a link to nothing appeared there: this export
    leaves it and removes its own folders."""
    output_dir, run = tmp_path / "b", subprocess.run

    def check_after_another_export(args, **kwargs):
        if "lerobot.bundle" in args and made == "bundle":
            output_dir.mkdir()
            (output_dir / "export.json").write_text("{}")
        elif "lerobot.bundle" in args:
            output_dir.symlink_to(tmp_path / "nothing")
        return run(args, **kwargs)

    monkeypatch.setattr(lerobot_export.subprocess, "run", check_after_another_export)
    with pytest.raises(RuntimeError, match="was made while the export ran"):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={output_dir}")
    assert list(tmp_path.iterdir()) == [output_dir]
    if made == "bundle":
        assert [path.name for path in output_dir.iterdir()] == ["export.json"]


@pytest.mark.parametrize("step", ["check"])
def test_a_failed_export_removes_the_parents_it_made_for_the_output_folder(
    checkpoint, tmp_path, monkeypatch, step
):
    """A first export into `outputs/export` makes both folders, and a failed one leaves neither behind. A parent
    that was there before stays."""
    (tmp_path / "kept").mkdir()
    run = subprocess.run
    monkeypatch.setattr(
        lerobot_export.subprocess,
        "run",
        lambda args, **kwargs: (
            subprocess.CompletedProcess(args, 1) if "lerobot.bundle" in args else run(args, **kwargs)
        ),
    )
    args = [f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'kept' / 'made' / 'deeper' / 'b'}"]
    with pytest.raises(RuntimeError, match="trace step failed" if step == "trace" else "failed its check"):
        run_export(*args, *(["--max_task_tokens=8"] if step == "trace" else []))
    assert list(tmp_path.iterdir()) == [tmp_path / "kept"]
    assert list((tmp_path / "kept").iterdir()) == []


def test_a_trace_step_that_cannot_make_its_folders_has_named_them_already(checkpoint, tmp_path, monkeypatch):
    """Making the second folder fails: the parents made for the output folder are named, so they are removed."""
    mkdir = Path.mkdir

    def no_traced_folder(path, mode=0o777, parents=False, exist_ok=False):
        if mode == 0o700:
            raise PermissionError(path)
        return mkdir(path, mode, parents, exist_ok)

    monkeypatch.setattr(Path, "mkdir", no_traced_folder)
    folders = tmp_path / "folders.json"
    args = [
        "--backend.type=torch_test",
        f"--policy.path={checkpoint}",
        f"--output_dir={tmp_path / 'made' / 'b'}",
    ]
    with pytest.raises(PermissionError):
        worker.trace_step({"step": "trace", "args": args, "folders": str(folders)})
    lerobot_export._remove_made(json.loads(folders.read_text()))
    assert list(tmp_path.iterdir()) == [folders]


def test_a_stop_between_the_trace_and_the_compile_steps_removes_what_the_trace_made(
    checkpoint, tmp_path, monkeypatch
):
    def stopped(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(lerobot_export, "parse_plugin_args", stopped)
    with pytest.raises(KeyboardInterrupt):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}")
    assert list(tmp_path.iterdir()) == []


def test_lerobot_export_is_the_installed_command():
    import importlib
    import tomllib

    scripts = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())["project"]["scripts"]
    module, _, name = scripts["lerobot-export"].partition(":")
    assert getattr(importlib.import_module(module), name) is lerobot_export.main


def test_export_prints_its_help_and_writes_nothing(tmp_path, monkeypatch, capfd):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stop:
        run_export("--help")
    assert stop.value.code == 0 and "usage: lerobot-export" in capfd.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_export_keeps_an_empty_output_folder_made_while_it_ran(checkpoint, tmp_path, monkeypatch):
    """A rename onto an empty folder would replace it without an error, so the export refuses to."""
    output_dir, run = tmp_path / "b", subprocess.run

    def check_after_a_folder_was_made(args, **kwargs):
        if "lerobot.bundle" in args:
            output_dir.mkdir()
        return run(args, **kwargs)

    monkeypatch.setattr(lerobot_export.subprocess, "run", check_after_a_folder_was_made)
    with pytest.raises(RuntimeError, match="was made while the export ran"):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={output_dir}")
    assert list(tmp_path.iterdir()) == [output_dir] and list(output_dir.iterdir()) == []


@pytest.mark.usefixtures("restore_torch_backend_flags")
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
    # The step sets up logging and loads the installed plugins for a process of its own; here both would stay
    # for every later test.
    monkeypatch.setattr(worker, "init_logging", lambda: None)
    monkeypatch.setattr(worker, "register_third_party_plugins", lambda **kwargs: None)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    worker.main()
    assert revisions == ["pin123", "pin123"] and tf32 == [(False, False)]
    # Later configs in this process read the command line, so the step gives it back.
    assert sys.argv == argv


@pytest.mark.usefixtures("restore_torch_backend_flags")
@pytest.mark.parametrize(
    ("names", "recorded"),
    [
        ([], None),
        (["pos_0", "pos_1", "gripper_0"], None),
        (["a.pos", "b.pos", "c.vel"], ["a.pos", "b.pos", "c.vel"]),
    ],
)
def test_the_trace_step_records_only_action_names_a_robot_can_have(
    checkpoint, tmp_path, monkeypatch, names, recorded
):
    import lerobot.policies.factory as factory

    load = factory.load_pretrained_policy

    def load_pretrained_policy(config, **kwargs):
        policy = load(config, **kwargs)
        policy.config.action_feature_names = names
        return policy

    monkeypatch.setattr(factory, "load_pretrained_policy", load_pretrained_policy)
    args = ["--backend.type=torch_test", f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}"]
    traced = worker.trace_step({"args": args, "folders": str(tmp_path / "folders.json")})
    assert traced["manifest"].get("action_names") == recorded


@pytest.mark.usefixtures("restore_torch_backend_flags")
def test_the_compile_step_runs_without_autograd_and_tf32(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        "tests.export.torch_backend.TorchTestBackend.compile",
        lambda self, program, path: seen.append(
            (torch.is_grad_enabled(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        ),
    )
    with torch.no_grad():
        program = torch.export.export(torch.nn.Linear(2, 2), (torch.zeros(1, 2),))
    torch.export.save(program, tmp_path / "part.pt2")
    request = tmp_path / "compile.json"
    request.write_text(
        json.dumps(
            {
                "step": "compile",
                "backend": {"type": "torch_test"},
                "program": str(tmp_path / "part.pt2"),
                "path": str(tmp_path / "out.pt2"),
                "plugins": [],
                "result": str(tmp_path / "result.json"),
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["worker", str(request)])
    # The step sets up logging and loads the installed plugins for a process of its own.
    monkeypatch.setattr(worker, "init_logging", lambda: None)
    monkeypatch.setattr(worker, "register_third_party_plugins", lambda **kwargs: None)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    worker.main()
    assert seen == [(False, False, False)]


def test_the_noise_program_draws_another_standard_normal_noise_each_run():
    program = torch.export.export(worker._DrawNoise((4000,), torch.float32, "cpu"), ()).module()
    first, second = program(), program()
    assert first.shape == (4000,) and not torch.equal(first, second)
    assert abs(first.mean()) < 0.1 and abs(first.std() - 1) < 0.1


def test_the_compile_step_names_what_its_backend_needs_installed(tmp_path, monkeypatch):
    def compile_without_its_compiler(self, program, path):
        raise ModuleNotFoundError("No module named 'compiler'")

    monkeypatch.setattr("tests.export.torch_backend.TorchTestBackend.compile", compile_without_its_compiler)
    monkeypatch.setattr("tests.export.torch_backend.TorchTestBackend.packages", ("compiler>=1.0",))
    with torch.no_grad():
        program = torch.export.export(torch.nn.Linear(2, 2), (torch.zeros(1, 2),))
    torch.export.save(program, tmp_path / "part.pt2")
    request = {
        "backend": {"type": "torch_test"},
        "program": str(tmp_path / "part.pt2"),
        "path": str(tmp_path / "out.pt2"),
        "plugins": [],
    }
    with pytest.raises(
        ImportError, match='The torch_test backend needs: pip install "compiler>=1.0"'
    ) as error:
        worker.compile_step(request)
    assert isinstance(error.value.__cause__, ModuleNotFoundError)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no SIGKILL")
def test_a_trace_step_killed_while_it_named_its_folders_is_reported_as_killed(tmp_path, monkeypatch):
    def killed_trace(args, **kwargs):
        Path(json.loads(Path(args[-1]).read_text())["folders"]).write_text('{"made": ["')
        return subprocess.CompletedProcess(args, -9)

    monkeypatch.setattr(lerobot_export.subprocess, "run", killed_trace)
    with pytest.raises(RuntimeError, match="trace step crashed with SIGKILL"):
        run_export(f"--output_dir={tmp_path / 'b'}")


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no SIGKILL")
# 40 is a Linux real-time signal, which Python does not name, and no signal on macOS.
@pytest.mark.parametrize(("code", "name"), [(-9, "SIGKILL"), (-40, "signal 40")])
def test_a_killed_trace_step_leaves_no_folders_behind(tmp_path, monkeypatch, code, name):
    made = [tmp_path / ".b-1", tmp_path / ".b-traced-1"]

    def killed_trace(args, **kwargs):
        request = json.loads(Path(args[-1]).read_text())
        for folder in made:
            folder.mkdir()
        folders = {"made": [str(folder) for folder in made], "parents": []}
        Path(request["folders"]).write_text(json.dumps(folders))
        return subprocess.CompletedProcess(args, code)

    monkeypatch.setattr(lerobot_export.subprocess, "run", killed_trace)
    with pytest.raises(RuntimeError, match=f"trace step crashed with {name}"):
        run_export(f"--output_dir={tmp_path / 'b'}")
    assert not [folder for folder in made if folder.exists()]


def test_a_bundle_keeps_arrays_that_are_not_contiguous_in_files_others_can_read(tmp_path):
    """safetensors writes an array's memory in order, and its save_file makes files only the owner reads."""
    from lerobot.export.bundle import write_bundle

    frame = {OBS_STATE: np.arange(6, dtype=np.float32).reshape(2, 3).T}
    constants = {"table": np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]}
    manifest = SimpleNamespace(
        test_case="test.safetensors", constants="constants.safetensors", task_tokens=None, action_tokens=None
    )
    config = SimpleNamespace(save_pretrained=lambda path: (path / "config.json").write_text("{}"))
    case = SimpleNamespace(frame=frame, actions=np.zeros((1, 3), np.float32), noise=None)
    write_bundle(SimpleNamespace(path=tmp_path, manifest=manifest), case, config, constants)
    np.testing.assert_array_equal(load_file(tmp_path / "test.safetensors")[OBS_STATE], frame[OBS_STATE])
    np.testing.assert_array_equal(load_file(tmp_path / "constants.safetensors")["table"], constants["table"])
    (tmp_path / "new").touch()
    usual = stat.S_IMODE((tmp_path / "new").stat().st_mode)
    for name in ("test.safetensors", "constants.safetensors"):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == usual, name


def test_export_names_the_signal_that_killed_the_check(checkpoint, tmp_path, monkeypatch):
    run = subprocess.run
    monkeypatch.setattr(
        lerobot_export.subprocess,
        "run",
        lambda args, **kwargs: (
            subprocess.CompletedProcess(args, -11) if "lerobot.bundle" in args else run(args, **kwargs)
        ),
    )
    with pytest.raises(RuntimeError, match="The bundle crashed with SIGSEGV"):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}")


# ExecuTorch 1.6 reads programs traced by PyTorch 2.15 or newer.
EXECUTORCH = is_package_available("executorch") and Version(torch.__version__).release >= (2, 15)
needs_executorch = pytest.mark.skipif(not EXECUTORCH, reason="needs ExecuTorch and PyTorch 2.15")
needs_executorch_cuda = pytest.mark.skipif(
    not (EXECUTORCH and torch.cuda.is_available()), reason="needs ExecuTorch, PyTorch 2.15 and a CUDA GPU"
)


@needs_executorch_cuda
def test_a_gpu_bundle_draws_its_noise_on_the_gpu(checkpoint, tmp_path, monkeypatch):
    """ACT exported to start from a noise: each chunk copies only the robot's frame to the GPU, and only the
    actions back."""
    from lerobot.bundle.executorch import ExecuTorchProgram, prepare

    # Before ExecuTorch loads: it reads the setting prepare() makes only once, when it loads.
    prepare()
    from executorch.runtime import ExecuTorchResult

    use_plugins(monkeypatch, "noise_recipe")
    run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}", backend="executorch_cuda")
    bundle = Bundle.read(tmp_path / "b")
    assert bundle.manifest.runs[0] == ProgramRun(NOISE, NOISE, [], [NOISE])
    # Every copy between the host and the GPU, also one that passes a program's output to the next program.
    copies = []
    to_gpu, to_host = ExecuTorchResult.from_host, ExecuTorchResult.to_host
    monkeypatch.setattr(
        ExecuTorchResult,
        "from_host",
        lambda value, device: copies.append(("to GPU", value.shape)) or to_gpu(value, device),
    )
    monkeypatch.setattr(
        ExecuTorchResult,
        "to_host",
        lambda value: copies.append(("to host", tuple(value.shape))) or to_host(value),
    )
    engine = BundleEngine(bundle)
    loaded = len(copies)
    first = np.stack([engine.get_action(robot_frame()) for _ in range(4)])
    second = np.stack([engine.get_action(robot_frame()) for _ in range(4)])
    assert not np.array_equal(first, second)
    assert copies[loaded:] == [("to GPU", (32, 32, 3)), ("to GPU", (3,)), ("to host", (4, 3))] * 2
    # The policy samples from a standard normal, so the noise program must too.
    files = bundle.manifest.programs[NOISE]
    noise = ExecuTorchProgram(
        bundle.path / files.file, None if files.data is None else bundle.path / files.data
    )
    count = 4000 // int(np.prod(bundle.manifest.noise_shape)) + 1
    draws = np.concatenate([noise.to_numpy(noise()[0]).ravel() for _ in range(count)])
    assert abs(draws.mean()) < 0.1 and abs(draws.std() - 1) < 0.1


@needs_executorch
def test_the_compiled_noise_program_draws_another_noise_each_run(tmp_path):
    """A compiler that folded the draw into a constant would play one noise forever."""
    from lerobot.bundle.executorch import ExecuTorchProgram
    from lerobot.export.backends.executorch_xnnpack import ExecuTorchXNNPACKConfig

    program = torch.export.export(worker._DrawNoise((4000,), torch.float32, "cpu"), ())
    with torch.no_grad():
        ExecuTorchXNNPACKConfig().compile(program, tmp_path / "noise.pte")
    noise = ExecuTorchProgram(tmp_path / "noise.pte")
    first, second = (noise.to_numpy(noise()[0]) for _ in range(2))
    assert first.shape == (4000,) and not np.array_equal(first, second)
    assert abs(first.mean()) < 0.1 and abs(first.std() - 1) < 0.1


# ExecuTorch installs coremltools only on the Pythons it supports.
@pytest.mark.skipif(
    not (sys.platform == "darwin" and EXECUTORCH and is_package_available("coremltools")),
    reason="needs ExecuTorch, PyTorch 2.15 and coremltools on macOS",
)
def test_executorch_coreml_compiles_a_program_that_core_ml_runs(tmp_path):
    from lerobot.bundle.executorch import ExecuTorchProgram
    from lerobot.export.backends.executorch_coreml import ExecuTorchCoreMLConfig

    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).eval()
    x = torch.randn(3, 4)
    path = tmp_path / "model.pte"
    # Test machines may have no GPU or Neural Engine to give.
    ExecuTorchCoreMLConfig(compute_units="cpu_only").compile(torch.export.export(model, (x,)), path)
    assert b"CoreMLBackend" in path.read_bytes()
    program = ExecuTorchProgram(path)
    (actual,) = program(x.numpy())
    np.testing.assert_allclose(program.to_numpy(actual), model(x).detach().numpy(), rtol=1e-5, atol=1e-5)


@needs_executorch
def test_an_executorch_whose_cuda_backend_cannot_name_weights_files_is_refused(monkeypatch):
    # ExecuTorch's CUDA backend imports Triton, which comes with PyTorch's CUDA builds and not its CPU builds.
    # The test needs no GPU, so its skip reason names none: tests/conftest.py fails a skip that names one in a
    # test not marked `gpu`.
    pytest.importorskip(
        "executorch.backends.cuda.cuda_partitioner", reason="needs Triton, which a CPU build of PyTorch lacks"
    )
    from lerobot.export.backends import executorch

    monkeypatch.setattr(executorch, "cuda_backend_names_its_weights", lambda: False)
    with pytest.raises(ValueError, match="pytorch/executorch#24314"):
        executorch.cuda_partitioner("part")


class MaskedPositions(torch.nn.Module):
    """Adds embeddings of position ids picked with a boolean mask made from constants alone, as SmolVLM does."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(8, dtype=torch.bool)
        ids = torch.zeros(8, dtype=torch.long)
        ids[mask] = torch.arange(8)[mask]
        return x + self.embedding(ids)


@needs_executorch
def test_a_backend_folds_what_a_program_computes_from_constants(tmp_path, monkeypatch):
    """The in-place write of the masked ids is folded too, so no indexing reaches the backend. On a GPU such
    indexing makes a tensor whose size the GPU would read back to the host on every run."""
    import executorch.exir

    from lerobot.export.backends.executorch_xnnpack import ExecuTorchXNNPACKConfig

    lowered = []
    to_edge = executorch.exir.to_edge_transform_and_lower
    monkeypatch.setattr(
        executorch.exir,
        "to_edge_transform_and_lower",
        lambda program, **kwargs: lowered.append(program) or to_edge(program, **kwargs),
    )
    with torch.no_grad():
        program = torch.export.export(MaskedPositions().eval(), (torch.randn(8, 4),))
    ExecuTorchXNNPACKConfig().compile(program, tmp_path / "model.pte")
    ops = [str(node.target) for node in lowered[0].graph.nodes if node.op == "call_function"]
    assert not [op for op in ops if "nonzero" in op or "index" in op], ops


@needs_executorch
def test_executorch_xnnpack_takes_the_linear_layers_of_a_bfloat16_program(tmp_path, monkeypatch):
    """XNNPACK takes bfloat16 ops only when asked to, and the portable kernels run them many times slower."""
    import executorch.exir
    from executorch.runtime import Runtime

    from lerobot.export.backends.executorch_xnnpack import ExecuTorchXNNPACKConfig

    ops = []
    to_edge = executorch.exir.to_edge_transform_and_lower

    def lower(program, **kwargs):
        edge = to_edge(program, **kwargs)
        ops.extend(
            str(node.target) for node in edge.exported_program().graph.nodes if node.op == "call_function"
        )
        return edge

    monkeypatch.setattr(executorch.exir, "to_edge_transform_and_lower", lower)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
    model = model.to(torch.bfloat16).eval()
    x = torch.randn(3, 4, dtype=torch.bfloat16)
    path = tmp_path / "model.pte"
    with torch.no_grad():
        ExecuTorchXNNPACKConfig().compile(torch.export.export(model, (x,)), path)
        expected = model(x)
    assert not [op for op in ops if "executorch_call_delegate" not in op and "getitem" not in op], ops
    (actual,) = Runtime.get().load_program(path).load_method("forward").execute([x])
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


EXPORT_WITHOUT_TORCH = """
import sys
import tempfile
from lerobot.scripts.lerobot_export import export

export([*sys.argv[2:], f"--output_dir={tempfile.mkdtemp(dir=sys.argv[1])}/b"])
"""


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("executorch_xnnpack", marks=needs_executorch),
        pytest.param("executorch_cuda", marks=needs_executorch_cuda),
    ],
)
def test_an_executorch_bundle_runs_without_torch(checkpoint, tmp_path, backend):
    """The bundle check loads the programs, on the CPU or the GPU, and replays the test case, without PyTorch."""
    run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}", backend=backend)
    assert_runs_without_torch(CHECK_BUNDLE, str(tmp_path / "b"))


# ExecuTorch for macOS has no OpenVINO backend.
@pytest.mark.skipif(
    not (
        EXECUTORCH
        and sys.platform != "darwin"
        and is_package_available("openvino")
        and is_package_available("nncf")
    ),
    reason="needs ExecuTorch, PyTorch 2.15 and OpenVINO, off a Mac: pip install openvino nncf",
)
def test_executorch_openvino_compiles_a_program_that_openvino_runs(tmp_path):
    from lerobot.bundle.executorch import ExecuTorchProgram
    from lerobot.export.backends.executorch_openvino import ExecuTorchOpenVINOConfig

    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).eval()
    x = torch.randn(3, 4)
    path = tmp_path / "model.pte"
    ExecuTorchOpenVINOConfig().compile(torch.export.export(model, (x,)), path)
    assert b"OpenvinoBackend" in path.read_bytes()
    program = ExecuTorchProgram(path)
    (actual,) = program(x.numpy())
    np.testing.assert_allclose(program.to_numpy(actual), model(x).detach().numpy(), rtol=1e-5, atol=1e-5)


def test_executorch_openvino_is_refused_before_tracing_on_a_mac(monkeypatch):
    from lerobot.export.backends.executorch_openvino import ExecuTorchOpenVINOConfig

    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(ValueError, match="executorch_openvino does not export on a Mac"):
        ExecuTorchOpenVINOConfig()
    monkeypatch.setattr(sys, "platform", "linux")
    ExecuTorchOpenVINOConfig()


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


@pytest.mark.parametrize(
    "backend",
    [
        "torch_test",
        pytest.param("executorch_xnnpack", marks=needs_executorch),
        pytest.param("executorch_cuda", marks=needs_executorch_cuda),
    ],
)
def test_a_recipe_runs_its_first_chunk_with_a_program_of_its_own(checkpoint, tmp_path, monkeypatch, backend):
    """ACT exported to start the sum of its noises in a first chunk of its own: the check plays three chunks."""
    use_plugins(monkeypatch, "first_runs_recipe")
    run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}", backend=backend)
    manifest = Bundle.read(tmp_path / "b").manifest
    assert manifest.carry == {"noise_total": None} and manifest.noise_shape == [4, 1]
    (first,) = manifest.first_runs
    assert first[-1].program == "begin" and manifest.runs[-1].program == "model"
    # On a GPU every chunk draws its noise first, in a run of its own.
    draws = [runs[0].outputs == [NOISE] for runs in (first, manifest.runs)]
    assert draws == [backend == "executorch_cuda"] * 2
    case = load_file(tmp_path / "b" / manifest.test_case)
    assert (
        case[IMAGE].shape == (12, 32, 32, 3)
        and case[ACTION].shape == (12, 3)
        and case[NOISE].shape == (3, 4, 1)
    )
    engine = BundleEngine(Bundle.read(tmp_path / "b"))
    actions = np.stack([engine.get_action(robot_frame()) for _ in range(8)])
    assert actions.shape == (8, 3) and np.isfinite(actions).all()
