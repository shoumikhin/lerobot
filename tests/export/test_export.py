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

import contextlib
import json
import logging
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file, save_file

from lerobot.bundle.engine import BundleEngine
from lerobot.bundle.manifest import NOISE, POLICY_CONFIG, Bundle, ProgramRun
from lerobot.configs import parser
from lerobot.export import worker
from lerobot.export.configs import ExportBackendConfig, ExportConfig
from lerobot.export.recipe import ExportRecipe
from lerobot.inference.sync import SyncInferenceEngine
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.scripts import lerobot_export
from lerobot.scripts.lerobot_export import export
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE
from tests.export.backends import (
    GPU_BACKENDS,
    needs_executorch,
    needs_executorch_coreml,
    needs_executorch_cuda,
    needs_executorch_openvino,
    needs_executorch_tensorrt,
)
from tests.export.plugins import use_plugins
from tests.export.tiny_act import IMAGE, save_checkpoint
from tests.export.torch_backend import TorchTestBackend
from tests.torch_free import assert_runs_without_torch


@pytest.fixture(scope="module", autouse=True)
def torch_test_backend():
    """Add the test backend to the export's steps, which run in processes of their own."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        use_plugins(monkeypatch, "torch_backend")
        yield


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("act")
    save_checkpoint(path)
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


@pytest.mark.parametrize(
    ("offset", "data_file", "passes"), [(0.0, False, True), (0.0, True, True), (2.5e-4, False, False)]
)
def test_a_part_checked_on_its_own_refuses_a_program_that_computes_something_else(
    checkpoint, tmp_path, monkeypatch, capfd, offset, data_file, passes
):
    """A drift the float32 test case lets through is far more than rounding moves the program's own outputs. A
    program with a data file is checked with it."""
    use_plugins(monkeypatch, "checked_recipe")
    args = (
        f"--policy.path={checkpoint}",
        f"--output_dir={tmp_path / 'bundle'}",
        f"--backend.offset={offset}",
        f"--backend.data_file={data_file}",
    )
    if passes:
        run_export(*args)
        assert (tmp_path / "bundle" / "model.bin").is_file() == data_file
    else:
        with pytest.raises(RuntimeError, match="check-model step failed"):
            run_export(*args)
        assert "computes something else than PyTorch on its own inputs" in capfd.readouterr().err
        assert list(tmp_path.iterdir()) == []


def test_a_backend_that_cannot_run_its_programs_says_so():
    with pytest.raises(NotImplementedError, match="The torch_test backend cannot run its programs at export"):
        ExportBackendConfig.run(TorchTestBackend(), Path("model.pt2"), None, ())


def test_a_backend_refuses_a_pytorch_older_than_it_needs(monkeypatch):
    monkeypatch.setattr(TorchTestBackend, "min_torch", "2.0")
    TorchTestBackend().check_torch()
    # A nightly or a release candidate of the version a backend needs counts as that version.
    monkeypatch.setattr(TorchTestBackend, "min_torch", "2.15")
    monkeypatch.setattr(torch, "__version__", "2.15.0.dev20260901")
    TorchTestBackend().check_torch()
    monkeypatch.setattr(torch, "__version__", "2.15.0rc1")
    TorchTestBackend().check_torch()
    monkeypatch.setattr(TorchTestBackend, "min_torch", "99.0")
    with pytest.raises(
        ValueError, match="needs PyTorch 99.0 or newer to export, but PyTorch 2.15.0rc1 is installed"
    ) as refusal:
        TorchTestBackend().check_torch()
    assert "pip install" not in str(refusal.value)

    monkeypatch.setattr(TorchTestBackend, "packages", ("compiler>=99.0,<99.1",))
    monkeypatch.setattr(TorchTestBackend, "torch_pins", ("torch>=99.0,<99.1", "torchvision>=9.0,<9.1"))
    with pytest.raises(ValueError) as refusal:
        TorchTestBackend().check_torch()
    message = str(refusal.value)
    assert 'pip install "torch>=99.0,<99.1" "torchvision>=9.0,<9.1" "compiler>=99.0,<99.1"' in message


def test_export_refuses_a_task_token_budget_for_a_policy_that_takes_no_task(checkpoint, tmp_path, capfd):
    with pytest.raises(RuntimeError, match="trace step failed"):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}", "--max_task_tokens=8")
    assert "--max_task_tokens applies only to a policy whose bundle takes any task" in capfd.readouterr().err


@pytest.mark.parametrize(
    ("args", "min_torch", "error"),
    [
        (["--output_dir={b}"], None, "--policy.path is required"),
        (["--policy.path={checkpoint}", "--output_dir={b}", "--tolerance=nan"], None, "not nan"),
        (["--policy.path={checkpoint}", "--output_dir={b}", "--tolerance=inf"], None, "not inf"),
        (["--policy.path={checkpoint}", "--output_dir={b}", "--tolerance=-1"], None, "not -1"),
        (
            ["--policy.path={checkpoint}", "--output_dir={b}", "--max_task_tokens=0"],
            None,
            "at least 1, not 0",
        ),
        (["--policy.path={checkpoint}", "--output_dir={tmp}"], None, "already exists. Pick another"),
        (["--policy.path={checkpoint}", "--output_dir={link}"], None, "already exists. Pick another"),
        (["--policy.path={checkpoint}", "--output_dir={b}"], "99.0", "needs PyTorch 99.0 or newer"),
    ],
)
def test_export_refuses_a_command_line_before_it_traces(
    checkpoint, tmp_path, monkeypatch, args, min_torch, error
):
    @parser.wrap()
    def parse(cfg: ExportConfig) -> ExportConfig:
        return cfg

    if min_torch is not None:
        monkeypatch.setattr("tests.export.torch_backend.TorchTestBackend.min_torch", min_torch)
    # A link to nothing is taken as there too: the export could not rename its bundle onto it.
    (tmp_path / "link").symlink_to(tmp_path / "nothing")
    values = {"checkpoint": checkpoint, "b": tmp_path / "b", "tmp": tmp_path, "link": tmp_path / "link"}
    argv = ["lerobot-export", "--backend.type=torch_test", *(arg.format(**values) for arg in args)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(Exception) as refusal:
        parse()
    # draccus raises its own error from a ValueError the config's checks raise.
    assert error in str(refusal.value.__cause__ or refusal.value)


def test_export_refuses_a_quantization_its_backend_does_not_offer(checkpoint, tmp_path, capfd):
    with pytest.raises(RuntimeError, match="trace step failed"):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}", "--quantize=fp8")
    assert "The torch_test backend cannot --quantize=fp8. It offers: none." in capfd.readouterr().err


def test_export_refuses_a_quantization_without_a_tolerance(tmp_path, monkeypatch):
    monkeypatch.setattr(TorchTestBackend, "quantizations", ("fp8",))
    with pytest.raises(ValueError, match="--quantize needs --tolerance"):
        ExportConfig(TorchTestBackend(), output_dir=tmp_path / "b", quantize="fp8")


def test_a_part_checked_on_its_own_is_refused_quantization():
    """Its check holds the program to rounding, which quantization exceeds by design."""
    from types import SimpleNamespace

    from lerobot.export.recipe import Part, ProgramCheck

    linear = torch.nn.Linear(2, 2)
    x = torch.zeros(1, 2)
    check = ProgramCheck((x,), (linear(x),), [(0.0, 0.0)])
    cfg = SimpleNamespace(quantize="fp8", policy=SimpleNamespace(type="tiny"), backend=None)
    with pytest.raises(ValueError, match="checks its model program to within what rounding moves it"):
        worker._quantized(cfg, Part("model", linear, (x,), ["x"], ["y"], check))


@pytest.mark.parametrize(("option", "value"), [("compute_units", "tpu"), ("compute_precision", "int8")])
def test_executorch_coreml_refuses_an_unknown_option_on_any_platform(option, value):
    from lerobot.export.backends.executorch_coreml import ExecuTorchCoreMLConfig

    with pytest.raises(ValueError, match=f"--backend.{option} must be one of"):
        ExecuTorchCoreMLConfig(**{option: value})


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_executorch_coreml_is_refused_before_tracing_off_a_mac(monkeypatch, platform):
    from lerobot.export.backends.executorch_coreml import ExecuTorchCoreMLConfig

    monkeypatch.setattr(sys, "platform", platform)
    with pytest.raises(ValueError, match="executorch_coreml exports only on a Mac"):
        ExecuTorchCoreMLConfig()
    monkeypatch.setattr(sys, "platform", "darwin")
    ExecuTorchCoreMLConfig()


SKIP_MARKS = """
import subprocess
import sys
from types import SimpleNamespace

import torch

import lerobot.utils.import_utils

torch.__version__ = "2.15.0"
torch.cuda.is_available = lambda: True
lerobot.utils.import_utils.is_package_available = lambda *args, **kwargs: True
# ExecuTorch's CUDA compiler does not import, as without Triton, though ExecuTorch says its CUDA backend runs.
sys.modules["executorch.backends.cuda.cuda_partitioner"] = None
subprocess.run = lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="True", stderr="")

from tests.export import backends

assert not backends.EXECUTORCH_CUDA
"""


def test_the_cuda_skip_mark_skips_when_the_cuda_compiler_cannot_be_imported():
    """On a machine with a GPU, the marks import ExecuTorch's CUDA compiler as they load, so an import that
    fails must skip the CUDA tests, not stop the tests from loading."""
    subprocess.run([sys.executable, "-c", SKIP_MARKS], check=True, timeout=300)


FAILED_PROBE = """
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import torch

import lerobot.utils.import_utils

torch.__version__ = "2.15.0"
torch.cuda.is_available = lambda: True
lerobot.utils.import_utils.is_package_available = lambda *args, **kwargs: True
sys.modules["executorch.backends.cuda.cuda_partitioner"] = ModuleType("cuda_partitioner")
# The process that asks ExecuTorch fails, as in a broken install, even though it printed "True".
subprocess.run = lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="True", stderr="broken")

from tests.export import backends

assert not backends.EXECUTORCH_CUDA
"""


def test_the_cuda_skip_mark_skips_when_executorch_cannot_say_whether_cuda_runs():
    """On a machine with a GPU, a probe process that fails is not trusted, even when it printed "True"."""
    subprocess.run([sys.executable, "-c", FAILED_PROBE], check=True, timeout=300)


@needs_executorch
def test_the_backend_probe_says_which_backends_run():
    """Every ExecuTorch package runs XNNPACK programs; a probe that never says yes skips the GPU tests."""
    from tests.export import backends

    assert backends._backend_runs("XnnpackBackend")
    assert not backends._backend_runs("NoSuchBackend")


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


@pytest.mark.parametrize("made", ["bundle", "empty folder", "broken link"])
def test_export_keeps_an_output_folder_made_while_it_ran(checkpoint, tmp_path, monkeypatch, made):
    """Another export to the same folder finished first, or an empty folder or a link to nothing appeared there:
    this export leaves it and removes its own folders. A rename onto an empty folder would replace it without an
    error, so the export refuses to."""
    output_dir, run = tmp_path / "b", subprocess.run

    def check_after_another_export(args, **kwargs):
        if "lerobot.bundle" in args and made == "broken link":
            output_dir.symlink_to(tmp_path / "nothing")
        elif "lerobot.bundle" in args:
            output_dir.mkdir()
            if made == "bundle":
                (output_dir / "export.json").write_text("{}")
        return run(args, **kwargs)

    monkeypatch.setattr(lerobot_export.subprocess, "run", check_after_another_export)
    with pytest.raises(RuntimeError, match="was made while the export ran"):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={output_dir}")
    assert list(tmp_path.iterdir()) == [output_dir]
    if made != "broken link":
        assert [path.name for path in output_dir.iterdir()] == (["export.json"] if made == "bundle" else [])


@pytest.mark.parametrize("step", ["trace", "check"])
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


def test_export_logs_each_step_with_its_time_and_leaves_out_the_compilers_logs(
    checkpoint, tmp_path, monkeypatch, caplog, capfd
):
    # Each step takes seconds, so each says at least once that it is still running.
    monkeypatch.setattr(lerobot_export, "HEARTBEAT_SECONDS", 0.1)
    with caplog.at_level(logging.INFO, logger=lerobot_export.__name__):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}")
    lines = [
        re.sub(r"\d+ s$", "N s", record.getMessage())
        for record in caplog.records
        if record.name == lerobot_export.__name__
    ]
    assert [line for line in lines if "still running" not in line] == [
        "Traced the policy in N s",
        "[1/2] Compiling model with torch_test",
        "[1/2] Compiling model with torch_test: done in N s",
        "[2/2] Checking the bundle",
        "[2/2] Checking the bundle: done in N s",
        f"Wrote {tmp_path / 'b'} in N s",
    ]
    for doing in ("Tracing the policy", "[1/2] Compiling model with torch_test", "[2/2] Checking the bundle"):
        assert f"{doing}: still running after N s" in lines
    assert "Lowering model" not in capfd.readouterr().err


def test_export_at_log_level_debug_shows_the_compilers_and_pytorchs_logs(checkpoint, tmp_path, capfd):
    run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}", "--log_level=DEBUG")
    err = capfd.readouterr().err
    assert "Lowering model" in err
    # PyTorch's own format, such as `I1104 09:27:58.123000 4242 torch/fx/experimental/symbolic_shapes.py:3712]`.
    assert re.search(r"^I\d{4} [\d:.]+ +\d+ torch/", err, re.MULTILINE)


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
    monkeypatch.setattr(worker, "set_log_level", lambda level: None)
    loaded = []
    monkeypatch.setattr(worker, "register_third_party_plugins", lambda kinds: loaded.append(kinds))
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    worker.main()
    assert revisions == ["pin123", "pin123"] and tf32 == [(False, False)]
    # Only the plugins an export uses, not every installed robot, camera or env.
    assert loaded == [("policy", "backend")]
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
    # The step sets up logging for a process of its own; here it would stay for every later test.
    monkeypatch.setattr(worker, "set_log_level", lambda level: None)
    args = ["--backend.type=torch_test", f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}"]
    traced = worker.trace_step({"args": args, "folders": str(tmp_path / "folders.json")})
    assert traced["manifest"].get("action_names") == recorded


def test_the_trace_step_quantizes_each_part_and_traces_it_in_the_backends_context(
    checkpoint, tmp_path, monkeypatch
):
    events = []

    @contextlib.contextmanager
    def context():
        events.append("enter")
        yield
        events.append("exit")

    def quantized(self, module, example_inputs, mode):
        events.append(f"quantize {mode}")
        return context()

    export = torch.export.export
    monkeypatch.setattr(TorchTestBackend, "quantizations", ("fp8",))
    monkeypatch.setattr(TorchTestBackend, "quantized", quantized)
    monkeypatch.setattr(
        torch.export, "export", lambda *args, **kwargs: events.append("export") or export(*args, **kwargs)
    )
    args = [
        "--backend.type=torch_test",
        f"--policy.path={checkpoint}",
        f"--output_dir={tmp_path / 'b'}",
        "--quantize=fp8",
        "--tolerance=1",
    ]
    request = tmp_path / "trace.json"
    folders, result = str(tmp_path / "folders.json"), str(tmp_path / "result.json")
    request.write_text(json.dumps({"step": "trace", "args": args, "folders": folders, "result": result}))
    monkeypatch.setattr(sys, "argv", ["worker", str(request)])
    monkeypatch.setattr(worker, "set_log_level", lambda level: None)
    recipe_tolerances, load = [], ExportRecipe.load.__func__
    monkeypatch.setattr(
        ExportRecipe,
        "load",
        classmethod(lambda cls, cfg: recipe_tolerances.append(cfg.tolerance) or load(cls, cfg)),
    )
    worker.main()
    assert events == ["quantize fp8", "enter", "export", "exit"]
    # The recipe checks PyTorch against PyTorch with its own tolerance; the bundle's test case takes --tolerance.
    assert recipe_tolerances == [None]
    traced = json.loads(Path(result).read_text())
    assert traced["manifest"]["tolerance"] == 1
    # The compile steps load the parts knowing how they were quantized.
    assert traced["quantize"] == "fp8"


def test_the_trace_step_records_what_the_programs_were_built_for_and_with(checkpoint, tmp_path, monkeypatch):
    from lerobot.bundle.manifest import Target

    monkeypatch.setattr(TorchTestBackend, "target", lambda self: Target("cuda", "Orin", "8.7"))
    monkeypatch.setattr(TorchTestBackend, "runtime_modules", ("numpy", "not_a_module"))
    monkeypatch.setattr(TorchTestBackend, "exact_runtime_modules", ("numpy",))
    monkeypatch.setattr(worker, "set_log_level", lambda level: None)
    args = ["--backend.type=torch_test", f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}"]
    manifest = worker.trace_step({"args": args, "folders": str(tmp_path / "folders.json")})["manifest"]
    assert manifest["target"] == {"device": "cuda", "gpu": "Orin", "compute_capability": "8.7"}
    assert manifest["runtime_versions"] == {"numpy": np.__version__}
    assert manifest["exact_runtime_modules"] == ["numpy"]


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
    monkeypatch.setattr(worker, "set_log_level", lambda level: None)
    monkeypatch.setattr(worker, "register_third_party_plugins", lambda **kwargs: None)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    worker.main()
    assert seen == [(False, False, False)]


def test_the_compile_step_leaves_the_backend_the_only_reference_to_the_program(tmp_path, monkeypatch):
    """A backend frees the traced program, and with it the part's weights, once it no longer needs it."""
    freed = []

    def compile(self, program, path):
        reference = weakref.ref(program)
        del program
        freed.append(reference() is None)

    monkeypatch.setattr(TorchTestBackend, "compile", compile)
    compile_part(tmp_path, None)
    assert freed == [True]


def compile_part(tmp_path, quantize):
    with torch.no_grad():
        program = torch.export.export(torch.nn.Linear(2, 2), (torch.zeros(1, 2),))
    torch.export.save(program, tmp_path / "part.pt2")
    worker.compile_step(
        {
            "backend": {"type": "torch_test"},
            "program": str(tmp_path / "part.pt2"),
            "path": str(tmp_path / "out.pt2"),
            "plugins": [],
            "quantize": quantize,
        }
    )


@pytest.mark.parametrize("quantize", [None, "fp8"])
def test_the_compile_step_imports_what_a_quantized_part_needs_before_it_loads_it(
    tmp_path, monkeypatch, quantize
):
    """A quantized part holds operators that only a module's import registers, and loading needs them."""
    order = []
    monkeypatch.setattr(TorchTestBackend, "quantization_imports", {"fp8": "registers_ops"})
    monkeypatch.setattr(worker.importlib, "import_module", lambda name: order.append(f"import {name}"))
    load = torch.export.load
    monkeypatch.setattr(torch.export, "load", lambda path: order.append("load") or load(path))
    monkeypatch.setattr(TorchTestBackend, "compile", lambda self, program, path: None)
    compile_part(tmp_path, quantize)
    assert order == (["load"] if quantize is None else ["import registers_ops", "load"])


def test_the_compile_step_says_which_quantization_needs_a_module_that_does_not_import(tmp_path, monkeypatch):
    monkeypatch.setattr(TorchTestBackend, "quantization_imports", {"fp8": "missing_module_for_quantized_ops"})
    with pytest.raises(
        ImportError, match="--quantize=fp8 holds operators that missing_module_for_quantized_ops"
    ):
        compile_part(tmp_path, "fp8")


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
    message = (
        r"could not import what it needs \(No module named 'compiler'\)\. "
        r'If it is not installed: pip install "compiler>=1.0"'
    )
    with pytest.raises(ImportError, match=message) as error:
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


class SavedConfig(SimpleNamespace):
    def save_pretrained(self, path: Path) -> None:
        def encode(value: object) -> object:
            return vars(value) if isinstance(value, SimpleNamespace) else str(value)

        (path / "config.json").write_text(json.dumps(vars(self), default=encode))


def test_a_bundle_keeps_no_absolute_path(tmp_path):
    """An absolute path keeps only its last part, also inside a list, a dict or a nested config, one this
    machine lacks, and a Windows one."""
    from lerobot.export.bundle import write_bundle

    (tmp_path / "SmolVLM2").mkdir()
    config = SavedConfig(
        pretrained_path=str(tmp_path),
        vlm_model_name=str(tmp_path / "SmolVLM2"),
        tokenizer="org/model",
        base_model_path=Path("/home/alice/models/base"),
        dataset_paths=["/home/alice/data/one", "org/two"],
        paths={"output": "/fsx/alice/run", "rooted": "\\Users\\alice\\ckpt"},
        windows_path="C:\\Users\\alice\\checkpoints\\last",
        backbone=SimpleNamespace(_name_or_path="/home/alice/models/backbone"),
    )
    manifest = SimpleNamespace(
        test_case="test.safetensors", constants=None, task_tokens=None, action_tokens=None
    )
    case = SimpleNamespace(frame={}, actions=np.zeros((1, 3), np.float32), noise=None)
    (tmp_path / "b").mkdir()
    write_bundle(SimpleNamespace(path=tmp_path / "b", manifest=manifest), case, config)
    saved = json.loads((tmp_path / "b" / POLICY_CONFIG).read_text())
    assert saved == {
        "pretrained_path": None,
        "vlm_model_name": "SmolVLM2",
        "tokenizer": "org/model",
        "base_model_path": "base",
        "dataset_paths": ["one", "org/two"],
        "paths": {"output": "run", "rooted": "ckpt"},
        "windows_path": "last",
        "backbone": {"_name_or_path": "backbone"},
    }


def test_a_bundle_keeps_arrays_that_are_not_contiguous_or_0d_in_files_others_can_read(tmp_path):
    """safetensors writes an array's memory in order, and its save_file makes files only the owner reads.

    A 0-d array keeps its shape.
    """
    from lerobot.export.bundle import write_bundle

    frame = {OBS_STATE: np.arange(6, dtype=np.float32).reshape(2, 3).T, "scalar": np.array(0.5, np.float32)}
    constants = {
        "table": np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2],
        "step": np.array(3, np.int64),
    }
    manifest = SimpleNamespace(
        test_case="test.safetensors", constants="constants.safetensors", task_tokens=None, action_tokens=None
    )
    config = SimpleNamespace(save_pretrained=lambda path: (path / "config.json").write_text("{}"))
    case = SimpleNamespace(frame=frame, actions=np.zeros((1, 3), np.float32), noise=None)
    write_bundle(SimpleNamespace(path=tmp_path, manifest=manifest), case, config, constants)
    np.testing.assert_array_equal(load_file(tmp_path / "test.safetensors")[OBS_STATE], frame[OBS_STATE])
    np.testing.assert_array_equal(load_file(tmp_path / "constants.safetensors")["table"], constants["table"])
    assert load_file(tmp_path / "test.safetensors")["scalar"].shape == ()
    assert load_file(tmp_path / "constants.safetensors")["step"].shape == ()
    (tmp_path / "new").touch()
    usual = stat.S_IMODE((tmp_path / "new").stat().st_mode)
    for name in ("test.safetensors", "constants.safetensors"):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == usual, name


STOPPED_EXPORT = """
import json, os, signal, subprocess, sys, threading, time
from pathlib import Path
from lerobot.scripts import lerobot_export

stage, stop, mode, run = sys.argv[1], getattr(signal, sys.argv[2]), sys.argv[3], subprocess.run
output_dir = Path(next(arg for arg in sys.argv if arg.startswith("--output_dir=")).split("=", 1)[1])
started = os.environ.get("STARTED")
# Writes its pid, then waits as a compile step would.
COMPILE = "import os, sys, time; open(sys.argv[1] + '.new', 'w').write(str(os.getpid())); "
COMPILE += "os.rename(sys.argv[1] + '.new', sys.argv[1]); time.sleep(60)"

def stop_once(found):
    while not found():
        time.sleep(0.01)
    os.kill(os.getpid(), stop)

def run_step(args, **kwargs):
    request = json.loads(Path(args[-1]).read_text()) if args[-1].endswith(".json") else {}
    if request.get("step") != stage:
        return run(args, **kwargs)
    if mode == "ignore":
        os.kill(os.getpid(), stop)
        return run(args, **kwargs)
    if stage == "trace":
        # Once the first part is traced: the trace step names its folders before it fills them, and the export
        # removes what it names.
        traced = lambda: list(output_dir.parent.glob(f".{output_dir.name}-traced-*/*.pt2"))
        threading.Thread(target=stop_once, args=(traced,), daemon=True).start()
        return run(args, **kwargs)
    threading.Thread(target=stop_once, args=(Path(started).exists,), daemon=True).start()
    return run([sys.executable, "-c", COMPILE, started], **kwargs)

lerobot_export.subprocess.run = run_step
# Ignored as nohup starts it, or handled by default even when the test runs under nohup.
signal.signal(stop, signal.SIG_IGN if mode == "ignore" else signal.SIG_DFL)
sys.argv = ["lerobot-export", *sys.argv[4:]]
lerobot_export.main()
"""


@pytest.mark.skipif(sys.platform == "win32", reason="On Windows, os.kill with SIGTERM ends a process at once")
# One handler takes both signals, so each signal is sent once and each step is stopped once.
@pytest.mark.parametrize(("stage", "stop"), [("trace", "SIGTERM"), ("compile", "SIGHUP")])
def test_lerobot_export_stopped_with_a_signal_removes_what_it_made(
    checkpoint, tmp_path, tmp_path_factory, stage, stop
):
    """A signal while a step runs: the export stops it and removes the folders beside the output folder."""
    started = tmp_path_factory.mktemp("compile") / "pid"
    args = ["--backend.type=torch_test", f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}"]
    result = subprocess.run(
        [sys.executable, "-c", STOPPED_EXPORT, stage, stop, "handle", *args],
        env={**os.environ, "STARTED": str(started)},
        timeout=600,
    )
    assert result.returncode == 128 + getattr(signal, stop)
    assert list(tmp_path.iterdir()) == []
    if stage == "compile":
        # The compile step had started, and is gone.
        with pytest.raises(ProcessLookupError):
            os.kill(int(started.read_text()), 0)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no SIGHUP")
@pytest.mark.parametrize("stop", ["SIGHUP", "SIGTERM"])
def test_lerobot_export_keeps_running_on_a_signal_it_was_started_ignoring(checkpoint, tmp_path, stop):
    """Under nohup SIGHUP is ignored, so closing the ssh session must not stop the export."""
    args = ["--backend.type=torch_test", f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}"]
    result = subprocess.run(
        [sys.executable, "-c", STOPPED_EXPORT, "compile", stop, "ignore", *args], timeout=600
    )
    assert result.returncode == 0
    assert [path.name for path in tmp_path.iterdir()] == ["b"]


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


@needs_executorch_cuda
def test_a_gpu_bundle_draws_its_noise_on_the_gpu(checkpoint, tmp_path, monkeypatch):
    """ACT exported to start from a noise: each chunk copies only the robot's frame to the GPU, and only the
    actions back."""
    from executorch.runtime import ExecuTorchResult

    from lerobot.bundle.executorch import ExecuTorchProgram

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


@needs_executorch_coreml
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
    # ExecuTorch's own tensor, since the tests load ExecuTorch as a bundle does (see conftest.py).
    torch.testing.assert_close(torch.from_dlpack(actual), expected, rtol=2e-2, atol=2e-2)


@needs_executorch
def test_executorch_xnnpack_resizes_long_wam_frames_to_the_same_pixels(tmp_path):
    from lerobot.bundle.executorch import ExecuTorchProgram
    from lerobot.export.backends.executorch_xnnpack import ExecuTorchXNNPACKConfig
    from lerobot.policies.longwam.processor_longwam import _pil_bilinear_resize

    class Resize(torch.nn.Module):
        def forward(self, image):
            return _pil_bilinear_resize(image, 16, 16)

    generator = torch.Generator().manual_seed(0)
    image = torch.randint(0, 256, (2, 3, 20, 28), dtype=torch.uint8, generator=generator)
    path = tmp_path / "resize.pte"
    ExecuTorchXNNPACKConfig().compile(torch.export.export(Resize(), (image,)), path)
    program = ExecuTorchProgram(path)
    (actual,) = program(image.numpy())
    np.testing.assert_array_equal(program.to_numpy(actual), Resize()(image).numpy())


EXPORT_WITHOUT_TORCH = """
import sys
import tempfile
from lerobot.scripts.lerobot_export import export

export([*sys.argv[2:], f"--output_dir={tempfile.mkdtemp(dir=sys.argv[1])}/b"])
"""


# In a process of its own: the backend returns PyTorch's tensors only from an ExecuTorch loaded without
# `prepare()`, which conftest.py calls in this process.
RUN_PART = """
import sys
from pathlib import Path

import torch

from lerobot.export.backends.executorch_xnnpack import ExecuTorchXNNPACKConfig

model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).eval()
x = torch.randn(3, 4)
backend = ExecuTorchXNNPACKConfig()
path = Path(sys.argv[1]) / "model.pte"
backend.compile(torch.export.export(model, (x,)), path)
(actual,) = backend.run(path, None, [x])
torch.testing.assert_close(actual, model(x).detach(), rtol=1e-5, atol=1e-5)
# Options for a backend this machine lacks are refused with the bundle's message, not ExecuTorch's code.
ExecuTorchXNNPACKConfig.runtime_options = {"NoSuchBackend": {"option": True}}
try:
    backend.run(path, None, [x])
except ValueError as error:
    assert "ExecuTorch backends this machine does not have" in str(error), error
else:
    raise AssertionError("an option for a missing backend was not refused")
"""


@needs_executorch
def test_an_executorch_part_runs_at_export_the_way_a_bundle_loads_it(tmp_path):
    subprocess.run([sys.executable, "-c", RUN_PART, str(tmp_path)], check=True, timeout=300)


@needs_executorch_openvino
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
        assert json.loads((bundle.path / POLICY_CONFIG).read_text())["pretrained_path"] is None
        # The permissions of a new folder, not the private ones of the temporary folder the bundle was.
        assert stat.S_IMODE(bundle.path.stat().st_mode) == stat.S_IMODE((tmp_path / "new").stat().st_mode)
        assert_plays_like_pytorch(checkpoint, bundle, IMAGE)


@pytest.mark.parametrize(
    "backend",
    [
        "torch_test",
        pytest.param("executorch_xnnpack", marks=needs_executorch),
        pytest.param("executorch_cuda", marks=needs_executorch_cuda),
        pytest.param("executorch_tensorrt", marks=needs_executorch_tensorrt),
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
    assert draws == [backend in GPU_BACKENDS] * 2
    case = load_file(tmp_path / "b" / manifest.test_case)
    assert (
        case[IMAGE].shape == (12, 32, 32, 3)
        and case[ACTION].shape == (12, 3)
        and case[NOISE].shape == (3, 4, 1)
    )
    engine = BundleEngine(Bundle.read(tmp_path / "b"))
    actions = np.stack([engine.get_action(robot_frame()) for _ in range(8)])
    assert actions.shape == (8, 3) and np.isfinite(actions).all()


def test_the_install_hints_name_an_extra_that_lerobot_defines():
    import tomllib

    from packaging.requirements import Requirement

    from lerobot.export.backends.executorch import EXPORT_EXTRA

    project = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())["project"]
    extra = Requirement(EXPORT_EXTRA)
    assert extra.name == project["name"] and extra.extras <= set(project["optional-dependencies"])
