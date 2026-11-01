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

"""The TD-MPC recipe end to end on tiny checkpoints: a bundle plans like the PyTorch policy, warm start included.

Each bundle is played next to the PyTorch policy in `lerobot-rollout`'s engines, on the same frames and the same
random draws, for many chunks and across a reset.
"""

import math

import numpy as np
import pytest
import torch
from packaging.version import Version

from lerobot.bundle.engine import BundleEngine
from lerobot.bundle.manifest import NOISE, Bundle, ProgramRun
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.export.backends.executorch_tensorrt import tensorrt_takes_the_plugin_setting
from lerobot.export.recipe import ReplayDraws
from lerobot.inference.sync import SyncInferenceEngine
from lerobot.policies import make_pre_post_processors
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
from lerobot.scripts.lerobot_export import export
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_STATE
from lerobot.utils.import_utils import is_package_available
from tests.export import torch_backend  # noqa: F401  # registers torch_test for the in-process trace step
from tests.export.plugins import use_plugins

# TD-MPC's image encoder needs a square image of at least 64 pixels.
FRAME = (64, 64)
TICKS = 24
# Within the first chunk after it, so the bundle and the policy both start planning again from nothing.
RESET_TICK = 13
EXECUTORCH = is_package_available("executorch") and Version(torch.__version__).release >= (2, 15)
BACKENDS = [
    "torch_test",
    pytest.param(
        "executorch_xnnpack",
        marks=pytest.mark.skipif(not EXECUTORCH, reason="needs ExecuTorch and PyTorch 2.15"),
    ),
]


@pytest.fixture(scope="module", autouse=True)
def torch_test_backend():
    with pytest.MonkeyPatch.context() as monkeypatch:
        use_plugins(monkeypatch, "torch_backend")
        yield


def tdmpc_checkpoint(path, features: dict[str, PolicyFeature], **changes) -> None:
    config = TDMPCConfig(
        input_features=features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        image_encoder_hidden_dim=8,
        state_encoder_hidden_dim=16,
        latent_dim=8,
        mlp_dim=16,
        horizon=3,
        cem_iterations=2,
        n_gaussian_samples=16,
        n_pi_samples=4,
        n_elites=4,
        # More than two, so the planner also draws which of them value the end of each trajectory.
        q_ensemble_size=3,
        device="cpu",
        **changes,
    )
    torch.manual_seed(0)
    policy = TDMPCPolicy(config)
    # Untrained, the reward and value heads are zero, so every trajectory would score the same.
    for head in [policy.model._reward, *policy.model._Qs]:
        torch.nn.init.normal_(head[-1].weight)
    policy.save_pretrained(path)
    stats = {ACTION: {"min": torch.full((2,), -2.0), "max": torch.full((2,), 3.0)}}
    for processor in make_pre_post_processors(config, dataset_stats=stats):
        processor.save_pretrained(path)


CAMERA = {
    OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *FRAME)),
    OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
}
KEYPOINTS = {
    OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
    OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
}
CHECKPOINTS = {
    # The defaults: the first action of each plan, repeated.
    "repeats": lambda path: tdmpc_checkpoint(path, CAMERA),
    "steps": lambda path: tdmpc_checkpoint(path, KEYPOINTS, n_action_repeats=1, n_action_steps=2),
    "policy_only": lambda path: tdmpc_checkpoint(path, CAMERA, use_mpc=False),
}


@pytest.fixture(scope="module", params=list(CHECKPOINTS))
def checkpoint(request, tmp_path_factory):
    path = tmp_path_factory.mktemp(request.param)
    CHECKPOINTS[request.param](path)
    return path


def random_frame(features: dict[str, PolicyFeature], rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        name: rng.integers(0, 256, (*FRAME, 3), dtype=np.uint8)
        if feature.type is FeatureType.VISUAL
        else rng.random(feature.shape, dtype=np.float32)
        for name, feature in features.items()
    }


def play(checkpoint, bundle: Bundle, monkeypatch) -> list[float]:
    """The largest difference on each tick between the bundle and the PyTorch policy, on the same draws, in
    normalized action units."""
    policy = TDMPCPolicy.from_pretrained(checkpoint)
    reference = SyncInferenceEngine(
        policy, *make_pre_post_processors(policy.config, checkpoint), {}, [], "", "cpu", ""
    )
    rng = np.random.default_rng(0)
    frames = [random_frame(policy.config.input_features, rng) for _ in range(TICKS)]
    with ReplayDraws() as recorded:
        reference.get_action(dict(frames[0]))
    reference.reset()
    shapes = recorded.shapes
    engine = BundleEngine(bundle)
    draws: list[np.ndarray] = []
    # Seeded, so every run replays the same draws: the planner picks among candidates, and on another draw two of
    # them could score close enough for rounding to pick another one.
    standard_normal = np.random.default_rng(1).standard_normal
    monkeypatch.setattr(
        np.random, "standard_normal", lambda shape: draws.append(standard_normal(shape)) or draws[-1]
    )
    scale = np.asarray(bundle.manifest.action_scale, dtype=np.float32)
    differences = []
    for tick, frame in enumerate(frames):
        if tick == RESET_TICK:
            engine.reset()
            reference.reset()
        drawn = len(draws)
        actual = engine.get_action(dict(frame))
        if len(draws) == drawn:
            expected = reference.get_action(dict(frame))
        else:
            noise = torch.from_numpy(draws[-1].astype(np.float32))
            pieces = noise.split([math.prod(shape) for shape in shapes])
            with ReplayDraws(
                [piece.view(shape) for piece, shape in zip(pieces, shapes, strict=True)]
            ) as replay:
                expected = reference.get_action(dict(frame))
            assert len(replay.shapes) == len(shapes)
        differences.append(float((np.abs(actual - expected.numpy()) / scale).max()))
    return differences


@pytest.mark.parametrize("backend", BACKENDS)
def test_bundle_plans_like_the_pytorch_policy_over_many_chunks(checkpoint, tmp_path, backend, monkeypatch):
    export([f"--backend.type={backend}", f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}"])
    bundle = Bundle.read(tmp_path / "b")
    config = TDMPCConfig.from_pretrained(checkpoint)
    planned = {"plan_mean": [config.horizon, 1, 2]} if config.use_mpc else {}
    assert bundle.manifest.carry == planned and bundle.manifest.history == {}
    assert (bundle.manifest.noise_shape is None) == (not config.use_mpc)
    # A compiled backend sums in another order than PyTorch, so hold it to the tolerance the bundle was checked with.
    assert max(play(checkpoint, bundle, monkeypatch)) <= bundle.manifest.tolerance


@pytest.mark.gpu
@pytest.mark.skipif(not (EXECUTORCH and torch.cuda.is_available()), reason="needs ExecuTorch and a CUDA GPU")
@pytest.mark.parametrize(
    "backend",
    [
        "executorch_cuda",
        pytest.param(
            "executorch_tensorrt",
            marks=pytest.mark.skipif(
                not (is_package_available("torch_tensorrt") and tensorrt_takes_the_plugin_setting()),
                reason="needs a Torch-TensorRT with pytorch/TensorRT#4953",
            ),
        ),
    ],
)
def test_a_gpu_bundle_draws_its_noise_on_the_gpu_and_passes_its_check(tmp_path, backend):
    """The GPU draws the noise, which the PyTorch policy cannot replay, so the export's own check of the test
    case, made with the recorded noise of each chunk, is the comparison."""
    CHECKPOINTS["repeats"](tmp_path / "ckpt")
    export(
        [f"--backend.type={backend}", f"--policy.path={tmp_path / 'ckpt'}", f"--output_dir={tmp_path / 'b'}"]
    )
    manifest = Bundle.read(tmp_path / "b").manifest
    assert manifest.runs[0] == ProgramRun(NOISE, NOISE, [], [NOISE])
    assert manifest.carry == {"plan_mean": [3, 1, 2]}
