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
import subprocess
import sys

import numpy as np
import pytest
import torch
from packaging.version import Version
from safetensors.numpy import load_file

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
from lerobot.utils.import_utils import is_package_available
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


@pytest.fixture(scope="module")
def bundle(checkpoint, tmp_path_factory):
    path = tmp_path_factory.mktemp("export") / "act_torch_test"
    run_export(f"--policy.path={checkpoint}", f"--output_dir={path}")
    return Bundle.read(path)


def assert_plays_like_pytorch(checkpoint, bundle: Bundle, image: str) -> None:
    policy = ACTPolicy.from_pretrained(checkpoint)
    reference = SyncInferenceEngine(
        policy, *make_pre_post_processors(policy.config, checkpoint), {}, [], "", "cpu", ""
    )
    engine = BundleEngine(bundle)
    rng = np.random.default_rng(0)
    frame = {
        image: rng.integers(0, 256, (32, 32, 3), dtype=np.uint8),
        OBS_STATE: rng.random(3, dtype=np.float32),
    }
    scale = np.asarray(bundle.manifest.action_scale, dtype=np.float32)
    for _ in range(policy.config.n_action_steps + 1):
        expected = reference.get_action(dict(frame)).numpy()
        # A compiled backend sums in another order than PyTorch, so hold it to the tolerance the bundle was checked
        # with, in normalized action units, as the engine does.
        drift = float((np.abs(engine.get_action(dict(frame)) - expected) / scale).max())
        assert drift <= bundle.manifest.tolerance, drift


def test_bundle_plays_the_actions_of_the_pytorch_policy(checkpoint, bundle):
    manifest = bundle.manifest
    assert (manifest.backend, manifest.runtime, manifest.task, manifest.noise_shape) == (
        "torch_test",
        "torch_test",
        None,
        None,
    )
    assert [(r.name, r.program, r.inputs, r.outputs) for r in manifest.runs] == [
        ("model", "model", [IMAGE, OBS_STATE], [ACTION])
    ]
    assert {name: (f.file, f.data) for name, f in manifest.programs.items()} == {"model": ("model.pt2", None)}
    np.testing.assert_allclose(manifest.action_scale, [30.0, 30.0, 30.0], rtol=1e-6)
    assert (bundle.path / POLICY_CONFIG).is_file() and not (bundle.path / "config.json").exists()
    assert_plays_like_pytorch(checkpoint, bundle, IMAGE)


def test_export_takes_a_singular_image_key(tmp_path):
    save_checkpoint(tmp_path / "act", OBS_IMAGE)
    run_export(f"--policy.path={tmp_path / 'act'}", f"--output_dir={tmp_path / 'bundle'}")
    assert_plays_like_pytorch(tmp_path / "act", Bundle.read(tmp_path / "bundle"), OBS_IMAGE)


def test_export_keeps_a_task_given(checkpoint, tmp_path):
    run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'bundle'}", "--task=pick")
    assert Bundle.read(tmp_path / "bundle").manifest.task == "pick"


def test_export_writes_nothing_over_tolerance(checkpoint, tmp_path):
    output_dir = tmp_path / "bundle"
    with pytest.raises(RuntimeError, match="failed its check"):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={output_dir}", "--backend.offset=10.0")
    assert list(tmp_path.iterdir()) == []


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


def test_a_backend_takes_a_pytorch_as_new_as_it_needs(monkeypatch):
    monkeypatch.setattr(TorchTestBackend, "min_torch", "2.0")
    TorchTestBackend().check_torch()


def test_a_backend_refuses_a_pytorch_older_than_it_needs(monkeypatch):
    monkeypatch.setattr(TorchTestBackend, "min_torch", "99.0")
    with pytest.raises(ValueError, match="needs PyTorch 99.0 or newer to export") as refusal:
        TorchTestBackend().check_torch()
    assert "torch-overrides.txt" not in str(refusal.value)

    monkeypatch.setattr(TorchTestBackend, "packages", ("compiler>=99.0,<99.1",))
    monkeypatch.setattr(TorchTestBackend, "torch_overrides", ("torch>=99.0,<99.1", "torchvision>=9.0,<9.1"))
    with pytest.raises(ValueError) as refusal:
        TorchTestBackend().check_torch()
    message = str(refusal.value)
    assert "printf 'torch>=99.0,<99.1\\ntorchvision>=9.0,<9.1\\n' > torch-overrides.txt" in message
    assert 'uv pip install --override torch-overrides.txt lerobot "compiler>=99.0,<99.1"' in message


@pytest.mark.parametrize("tolerance", ["nan", "inf", "-1"])
def test_export_refuses_a_tolerance_that_passes_anything(checkpoint, tmp_path, tolerance, capfd):
    with pytest.raises(RuntimeError, match="trace step failed"):
        run_export(
            f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}", f"--tolerance={tolerance}"
        )
    assert "--tolerance must be a finite number" in capfd.readouterr().err


def test_export_loads_the_requested_revision(checkpoint, tmp_path, monkeypatch):
    import lerobot.policies.factory as factory

    seen = []
    load, make = factory.load_pretrained_policy, factory.make_pre_post_processors

    def load_pretrained_policy(config):
        seen.append(config.pretrained_revision)
        return load(config)

    def make_pre_post_processors(*args, **kwargs):
        seen.append(kwargs["pretrained_revision"])
        return make(*args, **kwargs)

    monkeypatch.setattr(factory, "load_pretrained_policy", load_pretrained_policy)
    monkeypatch.setattr(factory, "make_pre_post_processors", make_pre_post_processors)
    monkeypatch.setattr(sys, "argv", ["worker"])
    args = [
        "--backend.type=torch_test",
        f"--policy.path={checkpoint}",
        f"--output_dir={tmp_path / 'b'}",
        "--policy.pretrained_revision=pin123",
    ]
    worker.trace_step({"step": "trace", "args": args})
    assert seen == ["pin123", "pin123"]


def test_export_steps_run_without_tf32(checkpoint, tmp_path, monkeypatch):
    import lerobot.policies.factory as factory

    seen = []
    load = factory.load_pretrained_policy

    def load_pretrained_policy(config):
        seen.append((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
        return load(config)

    monkeypatch.setattr(factory, "load_pretrained_policy", load_pretrained_policy)
    request = tmp_path / "trace.json"
    args = ["--backend.type=torch_test", f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}"]
    request.write_text(json.dumps({"step": "trace", "args": args, "result": str(tmp_path / "result.json")}))
    monkeypatch.setattr(sys, "argv", ["worker", str(request)])
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    try:
        worker.main()
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
    assert seen == [(False, False)]


def test_the_compile_step_runs_without_autograd(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        "tests.export.torch_backend.TorchTestBackend.compile",
        lambda self, program, path: seen.append(torch.is_grad_enabled()),
    )
    with torch.no_grad():
        program = torch.export.export(torch.nn.Linear(2, 2), (torch.zeros(1, 2),))
    torch.export.save(program, tmp_path / "part.pt2")
    request = {"backend": {"type": "torch_test"}, "program": str(tmp_path / "part.pt2")}
    worker.compile_step({**request, "path": str(tmp_path / "out.pt2")})
    assert seen == [False]


def test_export_names_the_signal_that_killed_the_check(checkpoint, tmp_path, monkeypatch):
    run = subprocess.run
    monkeypatch.setattr(
        lerobot_export.subprocess,
        "run",
        lambda args: subprocess.CompletedProcess(args, -11) if "lerobot.bundle" in args else run(args),
    )
    with pytest.raises(RuntimeError, match="crashed with SIGSEGV"):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}")


# ExecuTorch 1.6 reads programs traced by PyTorch 2.15 or newer.
EXECUTORCH = is_package_available("executorch") and Version(torch.__version__).release >= (2, 15)
BACKENDS = [
    "torch_test",
    pytest.param(
        "executorch_xnnpack",
        marks=pytest.mark.skipif(not EXECUTORCH, reason="needs ExecuTorch and PyTorch 2.15"),
    ),
    pytest.param(
        "executorch_cuda",
        marks=pytest.mark.skipif(
            not (EXECUTORCH and torch.cuda.is_available()),
            reason="needs ExecuTorch, PyTorch 2.15 and a CUDA GPU",
        ),
    ),
]


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


@pytest.mark.skipif(not EXECUTORCH, reason="needs ExecuTorch and PyTorch 2.15")
def test_lower_folds_what_a_program_computes_from_constants(monkeypatch):
    """The in-place write of the masked ids is folded too, so no indexing reaches the backend."""
    import executorch.exir
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    from lerobot.export.backends.executorch import lower

    lowered = []
    to_edge = executorch.exir.to_edge_transform_and_lower
    monkeypatch.setattr(
        executorch.exir,
        "to_edge_transform_and_lower",
        lambda program, **kwargs: lowered.append(program) or to_edge(program, **kwargs),
    )
    with torch.no_grad():
        lower(torch.export.export(MaskedPositions().eval(), (torch.randn(8, 4),)), XnnpackPartitioner())
    ops = [str(node.target) for node in lowered[0].graph.nodes if node.op == "call_function"]
    assert not [op for op in ops if "nonzero" in op or "index" in op], ops


@pytest.mark.skipif(
    not (EXECUTORCH and torch.cuda.is_available()), reason="needs ExecuTorch, PyTorch 2.15 and a CUDA GPU"
)
def test_executorch_cuda_folds_what_a_program_computes_from_constants(tmp_path, monkeypatch):
    """Indexing with a boolean mask makes a tensor whose size the GPU would read back to the host on every run."""
    import executorch.exir

    from lerobot.export.backends.executorch_cuda import ExecuTorchCUDAConfig

    lowered = []
    to_edge = executorch.exir.to_edge_transform_and_lower
    monkeypatch.setattr(
        executorch.exir,
        "to_edge_transform_and_lower",
        lambda program, **kwargs: lowered.append(program) or to_edge(program, **kwargs),
    )
    model = MaskedPositions().eval()
    x = torch.randn(8, 4)
    path = tmp_path / "model.pte"
    ExecuTorchCUDAConfig().compile(torch.export.export(model, (x,)), path)
    ops = [str(node.target) for node in lowered[0].graph.nodes if node.op == "call_function"]
    assert not [op for op in ops if "nonzero" in op or "index" in op], ops


class ConstantPositions(torch.nn.Module):
    """Adds embeddings of position ids that bucketize makes from constants alone, as SmolVLM's vision model does."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ids = torch.bucketize(torch.arange(0, 1 - 1e-6, 1 / 8), torch.arange(1 / 4, 1.0, 1 / 4), right=True)
        return x + self.embedding(ids)


@pytest.mark.skipif(
    not (sys.platform == "darwin" and EXECUTORCH), reason="needs ExecuTorch and PyTorch 2.15 on macOS"
)
def test_executorch_coreml_folds_what_a_program_computes_from_constants(tmp_path):
    """Core ML has no bucketize, so the position ids are computed at export."""
    from lerobot.bundle.executorch import ExecuTorchProgram
    from lerobot.export.backends.executorch_coreml import ExecuTorchCoreMLConfig

    model = ConstantPositions().eval()
    x = torch.randn(8, 8)
    path = tmp_path / "model.pte"
    ExecuTorchCoreMLConfig(compute_units="cpu_only").compile(torch.export.export(model, (x,)), path)
    program = ExecuTorchProgram(path)
    (actual,) = program(x.numpy())
    np.testing.assert_allclose(program.to_numpy(actual), model(x).detach().numpy(), rtol=1e-5, atol=1e-5)


EXPORT_WITHOUT_TORCH = """
import sys
import tempfile
from lerobot.scripts.lerobot_export import export

export([*sys.argv[2:], f"--output_dir={tempfile.mkdtemp(dir=sys.argv[1])}/b"])
"""


def test_export_runs_its_steps_without_importing_torch(checkpoint, tmp_path):
    """Only the steps import PyTorch, so what one step used is freed before the next starts."""
    args = ["--backend.type=torch_test", f"--policy.path={checkpoint}"]
    assert_runs_without_torch(EXPORT_WITHOUT_TORCH, str(tmp_path), *args)
    for run in tmp_path.iterdir():
        assert sorted(path.name for path in run.iterdir()) == ["b"]
        assert_plays_like_pytorch(checkpoint, Bundle.read(run / "b"), IMAGE)
