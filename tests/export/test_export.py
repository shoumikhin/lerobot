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

import subprocess
import sys

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file

import tests.export.torch_backend  # noqa: F401  # registers the test backend
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.export import worker
from lerobot.export.bundle import POLICY_CONFIG
from lerobot.inference.bundle import Bundle, BundleInferenceEngine
from lerobot.inference.sync import SyncInferenceEngine
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.scripts import lerobot_export
from lerobot.scripts.lerobot_export import export
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.utils.import_utils import is_package_available

IMAGE = "observation.images.front"
# The export steps run in processes of their own, which import this to know the test backend.
IMPORTS = ["tests.export.torch_backend"]


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


def run_export(*args: str, backend: str = "torch_test", imports: list[str] = IMPORTS) -> None:
    export([f"--backend.type={backend}", *args], imports)


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
    engine = BundleInferenceEngine(bundle)
    rng = np.random.default_rng(0)
    frame = {
        image: rng.integers(0, 256, (32, 32, 3), dtype=np.uint8),
        OBS_STATE: rng.random(3, dtype=np.float32),
    }
    for _ in range(policy.config.n_action_steps + 1):
        # A compiled backend sums in another order than PyTorch, so a few float32 ulps of drift are expected.
        np.testing.assert_allclose(
            engine.get_action(dict(frame)), reference.get_action(dict(frame)).numpy(), rtol=1e-5, atol=1e-5
        )


def test_bundle_plays_the_actions_of_the_pytorch_policy(checkpoint, bundle):
    manifest = bundle.manifest
    assert (manifest.backend, manifest.runtime, manifest.task, manifest.noise_shape) == (
        "torch_test",
        "torch_test",
        None,
        None,
    )
    assert [(p.inputs, p.outputs, p.data) for p in manifest.programs] == [
        ([IMAGE, OBS_STATE], [ACTION], None)
    ]
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
    worker.trace({"step": "trace", "args": args})
    assert seen == ["pin123", "pin123"]


def test_export_computes_the_test_case_without_tf32(checkpoint, tmp_path, monkeypatch):
    import lerobot.policies.factory as factory

    seen = []
    load = factory.load_pretrained_policy

    def load_pretrained_policy(config):
        seen.append((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
        return load(config)

    monkeypatch.setattr(factory, "load_pretrained_policy", load_pretrained_policy)
    monkeypatch.setattr(sys, "argv", ["worker"])
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    try:
        worker.trace(
            {
                "step": "trace",
                "args": [
                    "--backend.type=torch_test",
                    f"--policy.path={checkpoint}",
                    f"--output_dir={tmp_path / 'b'}",
                ],
            }
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
    assert seen == [(False, False)]


def test_export_names_the_signal_that_killed_the_check(checkpoint, tmp_path, monkeypatch):
    run = subprocess.run
    monkeypatch.setattr(
        lerobot_export.subprocess,
        "run",
        lambda args: subprocess.CompletedProcess(args, -11)
        if "lerobot.inference.bundle" in args
        else run(args),
    )
    with pytest.raises(RuntimeError, match="crashed with SIGSEGV"):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}")


BACKENDS = [
    "torch_test",
    pytest.param(
        "executorch_xnnpack",
        marks=pytest.mark.skipif(not is_package_available("executorch"), reason="executorch not installed"),
    ),
    pytest.param(
        "executorch_cuda",
        marks=pytest.mark.skipif(
            not (is_package_available("executorch") and torch.cuda.is_available()),
            reason="needs ExecuTorch and a CUDA GPU",
        ),
    ),
]


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_recipe_runs_a_part_several_times_with_constants(checkpoint, tmp_path, backend):
    """ACT exported as a chain whose last part runs twice, reading its factor from the constants."""
    run_export(
        f"--policy.path={checkpoint}",
        f"--output_dir={tmp_path / 'b'}",
        backend=backend,
        imports=[*IMPORTS, "tests.export.chain_recipe"],
    )
    bundle = Bundle.read(tmp_path / "b")
    assert_plays_like_pytorch(checkpoint, bundle, IMAGE)
    programs = bundle.manifest.programs
    suffix = programs[0].file.rsplit(".", 1)[1]
    assert [p.file for p in programs] == [
        f"prepare.{suffix}",
        f"model.{suffix}",
        f"scale.{suffix}",
        f"scale.{suffix}",
    ]
    assert [p.name for p in programs] == ["prepare", "model", "scale_0", "scale_1"]
    constants = load_file(bundle.path / bundle.manifest.constants)
    np.testing.assert_allclose([constants["factor_0"][0], constants["factor_1"][0]], [2.0, 0.5])


@pytest.mark.skipif(
    not (sys.platform == "darwin" and is_package_available("executorch")), reason="needs ExecuTorch on macOS"
)
def test_executorch_coreml_compiles_a_program_that_core_ml_runs(tmp_path):
    """Not end to end on ACT: Core ML takes no uint8 camera frames until pytorch/executorch#11686 is fixed."""
    from lerobot.export.backends.executorch_coreml import ExecuTorchCoreMLConfig
    from lerobot.inference.bundle.executorch import ExecuTorchProgram

    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).eval()
    x = torch.randn(3, 4)
    path = tmp_path / "model.pte"
    # Test machines may have no GPU or Neural Engine to give.
    ExecuTorchCoreMLConfig(compute_units="cpu_only").compile(torch.export.export(model, (x,)), path)
    assert b"CoreMLBackend" in path.read_bytes()
    program = ExecuTorchProgram(path)
    (actual,) = program(x.numpy())
    np.testing.assert_allclose(program.to_numpy(actual), model(x).detach().numpy(), rtol=1e-5, atol=1e-5)


EXPORT_WITHOUT_TORCH = """
import sys
from lerobot.scripts.lerobot_export import export

export(sys.argv[1:], {imports})
assert "torch" not in sys.modules
"""


def test_export_runs_its_steps_without_importing_torch(checkpoint, tmp_path):
    """Only the steps import PyTorch, so what one step used is freed before the next starts."""
    from tests.test_torch_free_imports import BLOCK_TORCH

    script = BLOCK_TORCH + EXPORT_WITHOUT_TORCH.format(imports=IMPORTS)
    args = ["--backend.type=torch_test", f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}"]
    subprocess.run([sys.executable, "-c", script, *args], check=True)
    assert_plays_like_pytorch(checkpoint, Bundle.read(tmp_path / "b"), IMAGE)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["b"]
