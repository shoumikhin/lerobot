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

"""The recipes of the policies that read past frames, end to end on tiny checkpoints.

Each bundle is played next to the PyTorch policy in `lerobot-rollout`'s engines, on the same frames and the
same random draws, for more ticks than one chunk, so both the history and the noise are exercised.
"""

import numpy as np
import pytest
import torch

import tests.export.torch_backend  # noqa: F401  # registers the test backend
from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.export.recipe import ReplayDraws
from lerobot.inference.bundle import Bundle, BundleInferenceEngine
from lerobot.inference.sync import SyncInferenceEngine
from lerobot.policies import make_pre_post_processors
from lerobot.policies.factory import get_policy_class
from lerobot.scripts.lerobot_export import export
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import is_package_available

IMAGE = "observation.images.front"
IMPORTS = ["tests.export.torch_backend"]
TICKS = 20
BACKENDS = [
    "torch_test",
    pytest.param(
        "executorch_xnnpack",
        marks=pytest.mark.skipif(not is_package_available("executorch"), reason="executorch not installed"),
    ),
]


def _min_max(size: int) -> dict[str, torch.Tensor]:
    return {"min": torch.full((size,), -2.0), "max": torch.full((size,), 3.0)}


def vqbet_checkpoint(path) -> None:
    from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig

    config = VQBeTConfig(
        input_features={
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        n_obs_steps=3,
        n_action_pred_token=2,
        action_chunk_size=4,
        crop_shape=(28, 28),
        pretrained_backbone_weights=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=8,
        vqvae_n_embed=8,
        vqvae_embedding_dim=16,
        vqvae_enc_hidden_dim=16,
        gpt_input_dim=32,
        gpt_output_dim=32,
        gpt_hidden_dim=32,
        gpt_n_layer=1,
        gpt_n_head=2,
        # Spread the code probabilities, so that the draws, not the logits, pick the codes.
        bet_softmax_temperature=10.0,
        device="cpu",
    )
    stats = {OBS_STATE: _min_max(2), ACTION: _min_max(2)}
    _save(config, stats, path)


def diffusion_checkpoint(path, **sampler) -> None:
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    config = DiffusionConfig(
        input_features={
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        n_obs_steps=3,
        horizon=8,
        n_action_steps=4,
        drop_n_last_frames=2,
        crop_shape=(28, 28),
        pretrained_backbone_weights=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=8,
        down_dims=(16, 32),
        kernel_size=3,
        n_groups=4,
        diffusion_step_embed_dim=16,
        num_train_timesteps=10,
        device="cpu",
        **sampler,
    )
    stats = {
        IMAGE: {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.2)},
        OBS_STATE: _min_max(2),
        ACTION: _min_max(2),
    }
    _save(config, stats, path)


def _save(config, stats, path) -> None:
    torch.manual_seed(0)
    get_policy_class(config.type)(config).save_pretrained(path)
    for processor in make_pre_post_processors(config, dataset_stats=stats):
        processor.save_pretrained(path)


CHECKPOINTS = {
    "vqbet": vqbet_checkpoint,
    "diffusion": diffusion_checkpoint,
    "diffusion_ddim": lambda path: diffusion_checkpoint(
        path, noise_scheduler_type="DDIM", prediction_type="sample", num_inference_steps=5
    ),
}


@pytest.fixture(scope="module", params=list(CHECKPOINTS))
def checkpoint(request, tmp_path_factory):
    path = tmp_path_factory.mktemp(request.param)
    CHECKPOINTS[request.param](path)
    return path


def play(checkpoint, bundle: Bundle, monkeypatch) -> list[float]:
    """The largest difference on each tick between the bundle and the PyTorch policy, on the same draws."""
    policy = get_policy_class(PreTrainedConfig.from_pretrained(checkpoint).type).from_pretrained(checkpoint)
    reference = SyncInferenceEngine(
        policy, *make_pre_post_processors(policy.config, checkpoint), {}, [], "", "cpu", ""
    )
    engine = BundleInferenceEngine(bundle)
    draws: list[np.ndarray] = []
    standard_normal = np.random.standard_normal
    monkeypatch.setattr(
        np.random, "standard_normal", lambda shape: draws.append(standard_normal(shape)) or draws[-1]
    )
    rng = np.random.default_rng(0)
    differences = []
    for _ in range(TICKS):
        frame = {
            IMAGE: rng.integers(0, 256, (32, 32, 3), dtype=np.uint8),
            OBS_STATE: rng.random(2, dtype=np.float32),
        }
        drawn = len(draws)
        actual = engine.get_action(dict(frame))
        if len(draws) == drawn:
            expected = reference.get_action(dict(frame))
        else:
            noise = torch.from_numpy(draws[-1].astype(np.float32))
            with ReplayDraws(noise) as replay:
                expected = reference.get_action(dict(frame))
            assert len(replay.shapes) == len(noise)
        differences.append(float(np.abs(actual - expected.numpy()).max()))
    return differences


@pytest.mark.parametrize("backend", BACKENDS)
def test_bundle_plays_like_the_pytorch_policy_over_many_ticks(checkpoint, tmp_path, backend, monkeypatch):
    export(
        [f"--backend.type={backend}", f"--policy.path={checkpoint}", f"--output_dir={tmp_path / 'b'}"],
        IMPORTS,
    )
    bundle = Bundle.read(tmp_path / "b")
    steps = PreTrainedConfig.from_pretrained(checkpoint).n_obs_steps
    assert bundle.manifest.history == {IMAGE: steps, OBS_STATE: steps}
    # A compiled backend sums in another order than PyTorch, so a few float32 ulps of drift are expected.
    assert max(play(checkpoint, bundle, monkeypatch)) < 1e-5


def test_diffusion_runs_its_denoising_step_once_per_timestep(tmp_path):
    diffusion_checkpoint(tmp_path / "ckpt")
    export(
        ["--backend.type=torch_test", f"--policy.path={tmp_path / 'ckpt'}", f"--output_dir={tmp_path / 'b'}"],
        IMPORTS,
    )
    manifest = Bundle.read(tmp_path / "b").manifest
    assert [p.file for p in manifest.programs] == ["encoder.pt2", *["step.pt2"] * 10, "actions.pt2"]
    assert manifest.programs[1].inputs == [
        "sample_0",
        "conditioning",
        "timestep_0",
        "coefficients_0",
        "noise",
        "noise_index_0",
    ]
    # The starting sample, then the noise DDPM adds at each timestep but the last.
    assert manifest.noise_shape == [10, 1, 8, 2]


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"variance_type": "fixed_large"}, "fixed_small"),
        ({"thresholding": True}, "thresholding True"),
        ({"prediction_type": "v_prediction"}, "'v_prediction'"),
    ],
)
def test_a_schedule_the_step_cannot_compute_is_refused(change, error):
    from diffusers import DDPMScheduler

    from lerobot.policies.common.export_denoising import scheduler_schedule

    scheduler = DDPMScheduler(num_train_timesteps=10, **change)
    scheduler.set_timesteps(10)
    with pytest.raises(ValueError, match=error):
        scheduler_schedule(scheduler)
