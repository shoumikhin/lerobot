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
from lerobot.bundle.manifest import NOISE, POLICY_CONFIG, Bundle, ProgramRun
from lerobot.export import worker
from lerobot.export.backends.executorch_xnnpack import ExecuTorchXNNPACKConfig
from lerobot.inference.sync import SyncInferenceEngine
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.scripts import lerobot_export
from lerobot.scripts.lerobot_export import export
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.utils.import_utils import is_package_available
from tests.export.backends import EXECUTORCH, needs_executorch, needs_executorch_cuda
from tests.export.plugins import use_plugins
from tests.export.tiny_act import IMAGE, save_checkpoint
from tests.export.torch_backend import TorchTestBackend
from tests.torch_free import assert_runs_without_torch


@pytest.fixture(scope="module", autouse=True)
def torch_test_backend():
    with pytest.MonkeyPatch.context() as monkeypatch:
        use_plugins(monkeypatch, "torch_backend")
        yield


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


@pytest.mark.parametrize(("offset", "passes"), [(0.0, True), (2.5e-4, False)])
def test_a_part_checked_on_its_own_refuses_a_program_that_computes_something_else(
    checkpoint, tmp_path, monkeypatch, offset, passes
):
    """A drift the float32 test case lets through is far more than rounding moves the program's own outputs."""
    use_plugins(monkeypatch, "checked_recipe")
    args = (
        f"--policy.path={checkpoint}",
        f"--output_dir={tmp_path / 'bundle'}",
        f"--backend.offset={offset}",
    )
    if passes:
        run_export(*args)
    else:
        with pytest.raises(RuntimeError, match="check-model step failed"):
            run_export(*args)
        assert list(tmp_path.iterdir()) == []


def test_a_backend_refuses_a_pytorch_older_than_it_needs(monkeypatch):
    monkeypatch.setattr(TorchTestBackend, "min_torch", "2.0")
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


def test_export_refuses_a_task_token_budget_for_a_policy_that_takes_no_task(checkpoint, tmp_path, capfd):
    with pytest.raises(RuntimeError, match="trace step failed"):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}", "--max_task_tokens=8")
    assert "--max_task_tokens applies only to a policy whose bundle takes any task" in capfd.readouterr().err


def test_export_refuses_a_task_token_budget_below_one(checkpoint, tmp_path, capfd):
    with pytest.raises(RuntimeError, match="trace step failed"):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}", "--max_task_tokens=0")
    assert "--max_task_tokens must be at least 1" in capfd.readouterr().err


@pytest.mark.parametrize(("option", "value"), [("compute_units", "tpu"), ("compute_precision", "int8")])
def test_executorch_coreml_refuses_an_unknown_option_on_any_platform(option, value):
    from lerobot.export.backends.executorch_coreml import ExecuTorchCoreMLConfig

    with pytest.raises(ValueError, match=f"--backend.{option} must be one of"):
        ExecuTorchCoreMLConfig(**{option: value})


def test_export_refuses_a_checkpoint_that_leaves_a_weight_unfilled(checkpoint, tmp_path, capfd):
    partial = tmp_path / "partial"
    shutil.copytree(checkpoint, partial)
    weights = load_file(partial / "model.safetensors")
    del weights["model.action_head.weight"]
    save_file(weights, partial / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="trace step failed"):
        run_export(f"--policy.path={partial}", f"--output_dir={tmp_path / 'b'}")
    assert "model.action_head.weight" in capfd.readouterr().err


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
    # The step sets up logging for a process of its own; here it would change the logs of every later test.
    monkeypatch.setattr(worker, "init_logging", lambda: None)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    worker.main()
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
    worker.compile_step(
        {
            "backend": {"type": "torch_test"},
            "program": str(tmp_path / "part.pt2"),
            "path": str(tmp_path / "out.pt2"),
            "plugins": [],
        }
    )
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


SIGTERMED_EXPORT = """
import json, os, signal, subprocess, sys, threading
from pathlib import Path
from lerobot.scripts import lerobot_export

stage, run = sys.argv[1], subprocess.run

def run_step(args, **kwargs):
    request = json.loads(Path(args[-1]).read_text()) if args[-1].endswith(".json") else {}
    if request.get("step") != stage:
        return run(args, **kwargs)
    if stage == "trace":
        # A trace step makes its folders beside the output folder and lists them as it goes.
        output_dir = Path(next(arg for arg in request["args"] if arg.startswith("--output_dir="))[13:])
        made = [output_dir.parent / ".b-1", output_dir.parent / ".b-traced-1"]
        for folder in made:
            folder.mkdir()
        Path(request["folders"]).write_text(json.dumps([str(folder) for folder in made]))
    threading.Timer(1, os.kill, (os.getpid(), signal.SIGTERM)).start()
    return run([sys.executable, "-c", "import time; time.sleep(60)"], **kwargs)

lerobot_export.subprocess.run = run_step
sys.argv = ["lerobot-export", *sys.argv[2:]]
lerobot_export.main()
"""


@pytest.mark.skipif(sys.platform == "win32", reason="On Windows, os.kill with SIGTERM ends a process at once")
@pytest.mark.parametrize("stage", ["trace", "compile"])
def test_lerobot_export_stopped_with_sigterm_removes_what_it_made(checkpoint, tmp_path, stage):
    """SIGTERM while a step runs: the export stops it and removes the folders beside the output folder."""
    args = ["--backend.type=torch_test", f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}"]
    result = subprocess.run([sys.executable, "-c", SIGTERMED_EXPORT, stage, *args], timeout=600)
    assert result.returncode == 128 + 15
    assert list(tmp_path.iterdir()) == []


def test_export_names_the_signal_that_killed_the_check(checkpoint, tmp_path, monkeypatch):
    run = subprocess.run
    monkeypatch.setattr(
        lerobot_export.subprocess,
        "run",
        lambda args: subprocess.CompletedProcess(args, -11) if "lerobot.bundle" in args else run(args),
    )
    with pytest.raises(RuntimeError, match="crashed with SIGSEGV"):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}")


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


@pytest.mark.skipif(
    not (sys.platform == "darwin" and EXECUTORCH), reason="needs ExecuTorch and PyTorch 2.15 on macOS"
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


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(ExecuTorchXNNPACKConfig, marks=needs_executorch),
    ],
)
def test_a_backend_folds_what_a_program_computes_from_constants(tmp_path, monkeypatch, backend):
    """The in-place write of the masked ids is folded too, so no indexing reaches the backend. On a GPU such
    indexing makes a tensor whose size the GPU would read back to the host on every run."""
    import executorch.exir

    lowered = []
    to_edge = executorch.exir.to_edge_transform_and_lower
    monkeypatch.setattr(
        executorch.exir,
        "to_edge_transform_and_lower",
        lambda program, **kwargs: lowered.append(program) or to_edge(program, **kwargs),
    )
    with torch.no_grad():
        program = torch.export.export(MaskedPositions().eval(), (torch.randn(8, 4),))
    backend().compile(program, tmp_path / "model.pte")
    ops = [str(node.target) for node in lowered[0].graph.nodes if node.op == "call_function"]
    assert not [op for op in ops if "nonzero" in op or "index" in op], ops


EXPORT_WITHOUT_TORCH = """
import sys
import tempfile
from lerobot.scripts.lerobot_export import export

export([*sys.argv[2:], f"--output_dir={tempfile.mkdtemp(dir=sys.argv[1])}/b"])
"""


@pytest.mark.skipif(
    not (EXECUTORCH and is_package_available("openvino") and is_package_available("nncf")),
    reason="needs ExecuTorch, PyTorch 2.15 and OpenVINO: pip install openvino nncf",
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
