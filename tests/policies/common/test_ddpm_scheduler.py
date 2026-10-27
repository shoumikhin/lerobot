#!/usr/bin/env python

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

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.common.ddpm_scheduler import ExportableDDPMScheduler  # noqa: E402
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy  # noqa: E402
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig  # noqa: E402
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (  # noqa: E402
    DiffusionObjective,
    MultiTaskDiTPolicy,
)
from lerobot.policies.utils import populate_queues  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

CAMERA = f"{OBS_IMAGES}.front"


class Sampler(torch.nn.Module):
    def __init__(self, model: DiffusionModel):
        super().__init__()
        self.model = model

    def forward(self, global_cond, noise):
        return self.model.conditional_sample(noise.shape[0], global_cond=global_cond, noise=noise)


class Denoiser(torch.nn.Module):  # a stand-in for the DiT
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)

    def forward(self, sample, timestep, conditioning_vec):
        return self.proj(sample) * timestep[:, None, None] / 100 + conditioning_vec[:, None]


class ObjectiveSampler(torch.nn.Module):
    def __init__(self, objective: DiffusionObjective, denoiser: Denoiser):
        super().__init__()
        self.objective, self.denoiser = objective, denoiser

    def forward(self, conditioning_vec, noise):
        return self.objective.conditional_sample(self.denoiser, noise.shape[0], conditioning_vec, noise=noise)


def make_sampler(kind, scheduler):
    """A 10-step diffusion model or Multi-Task DiT objective, a module sampling with it, and its inputs."""
    features = {
        "input_features": {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,))},
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        "noise_scheduler_type": scheduler,
        "num_train_timesteps": 100,
        "num_inference_steps": 10,
    }
    if kind == "diffusion":
        config = DiffusionConfig(
            horizon=8,
            n_action_steps=4,
            down_dims=(16, 32),
            diffusion_step_embed_dim=16,
            n_groups=4,
            **features,
        )
        model = DiffusionModel(config).eval()
        return model, Sampler(model), (torch.randn(2, 4 * config.n_obs_steps), torch.randn(2, 8, 2))
    config = MultiTaskDiTConfig(objective="diffusion", **features)
    objective = DiffusionObjective(config, action_dim=2, horizon=8)
    return objective, ObjectiveSampler(objective, Denoiser()), (torch.randn(2, 2), torch.randn(2, 8, 2))


def sample(sampler, args):
    torch.manual_seed(0)
    return sampler(*args)


@pytest.mark.parametrize(
    "timesteps", [None, 10, [90, 50, 20, 7, 0]], ids=["training_schedule", "10_steps", "custom"]
)
def test_exportable_ddpm_scheduler_steps_like_diffusers(timesteps):
    kwargs = {"num_train_timesteps": 100, "beta_schedule": "squaredcos_cap_v2"}
    ours, theirs = ExportableDDPMScheduler(**kwargs), diffusers.DDPMScheduler(**kwargs)
    for scheduler in (ours, theirs):
        if isinstance(timesteps, int):
            scheduler.set_timesteps(timesteps)
        elif timesteps is not None:
            scheduler.set_timesteps(timesteps=timesteps)
    model_output, noisy = torch.randn(2, 8, 2), torch.randn(2, 8, 2)
    for t in theirs.timesteps.tolist():
        torch.manual_seed(0)
        expected = theirs.step(model_output, t, noisy).prev_sample
        torch.manual_seed(0)
        assert torch.equal(ours.step(model_output, t, noisy).prev_sample, expected)
    if timesteps is not None:
        # diffusers raises IndexError for a timestep that is not in the schedule.
        with pytest.raises(ValueError):
            ours.step(model_output, 55, noisy)


@pytest.mark.parametrize("scheduler", ["DDPM", "DDIM"])
@pytest.mark.parametrize("kind", ["diffusion", "multi_task_dit"])
def test_exported_sampling_matches_eager(kind, scheduler):
    model, sampler, args = make_sampler(kind, scheduler)
    model.num_inference_steps = 5
    assert len(model.noise_scheduler.timesteps) == 5
    with pytest.raises(ValueError, match="at least 1"):
        model.num_inference_steps = 0
    with pytest.raises(ValueError, match="cannot be larger"):
        model.num_inference_steps = 1000
    assert model.num_inference_steps == 5
    # Exported before any eager sample, so it relies on the schedule the assignment set. Strict tracing
    # cannot read the timesteps as ints.
    exported = torch.export.export(sampler, args, strict=False).module()
    assert torch.equal(sample(exported, args), sample(sampler, args))


@pytest.mark.parametrize("replacement", [ExportableDDPMScheduler, diffusers.DDIMScheduler])
@pytest.mark.parametrize("kind", ["diffusion", "multi_task_dit"])
def test_a_replaced_scheduler_samples_the_configured_steps(kind, replacement):
    model, sampler, args = make_sampler(kind, "DDPM")
    model.noise_scheduler = replacement.from_config(model.noise_scheduler.config)
    calls, denoiser = [], model.unet if kind == "diffusion" else sampler.denoiser
    denoiser.register_forward_hook(lambda *args: calls.append(1))
    with pytest.raises(RuntimeError, match="no longer holds the schedule"):
        torch.export.export(sampler, args, strict=False)
    sampler(*args)
    assert len(calls) == 10
    exported = torch.export.export(sampler, args, strict=False).module()
    assert torch.equal(sample(exported, args), sample(sampler, args))


@pytest.mark.parametrize("kind", ["diffusion", "multi_task_dit"])
def test_tracing_refuses_a_schedule_that_eager_sampling_would_reset(kind):
    def export(sampler, args):
        return torch.export.export(sampler, args, strict=False).module()

    # A changed DDIM spacing keeps 10 steps. torch.compile resets the schedule like eager sampling.
    model, sampler, args = make_sampler(kind, "DDIM")
    model.noise_scheduler.register_to_config(timestep_spacing="trailing")
    stale = f"Assign num_inference_steps on the {type(model).__name__}"
    with pytest.raises(RuntimeError, match=stale):
        export(sampler, args)
    assert torch.equal(sample(torch.compile(sampler, backend="eager"), args), sample(sampler, args))
    assert torch.equal(sample(export(sampler, args), args), sample(sampler, args))

    # A DDIM step reads the scheduler's own step count, so setting only that count makes the schedule stale.
    model, sampler, args = make_sampler(kind, "DDIM")
    model.noise_scheduler.num_inference_steps = 20
    with pytest.raises(RuntimeError, match=stale):
        export(sampler, args)

    # Custom DDPM timesteps keep the step count at 10. Assigning it on the model sets the schedule again.
    model, sampler, args = make_sampler(kind, "DDPM")
    model.noise_scheduler.set_timesteps(timesteps=[99, 49, 0])
    with pytest.raises(RuntimeError, match=stale):
        export(sampler, args)
    model.num_inference_steps = 10
    assert torch.equal(sample(export(sampler, args), args), sample(sampler, args))

    # A swapped scheduler whose own step count is set to 10 still holds its default timesteps.
    model.noise_scheduler = diffusers.DDIMScheduler.from_config(model.noise_scheduler.config)
    model.noise_scheduler.num_inference_steps = 10
    with pytest.raises(RuntimeError, match=stale):
        export(sampler, args)
    model.noise_scheduler = diffusers.DDPMScheduler.from_config(model.noise_scheduler.config)
    model.num_inference_steps = 10
    with pytest.raises(RuntimeError, match="only with DDIMScheduler or lerobot's ExportableDDPMScheduler"):
        export(sampler, args)


def assert_export_matches_eager(model, args, other_args):
    with torch.no_grad():
        exported = torch.export.export(model, args, strict=False).module()
        for inputs in (args, other_args):
            assert torch.equal(sample(exported, inputs), sample(model, inputs))


def test_exported_diffusion_policy_matches_eager():
    config = DiffusionConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 40, 40)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        crop_shape=(36, 36),
        pretrained_backbone_weights=None,  # the default ResNet-18, without downloading its weights
        down_dims=(16, 32),
        diffusion_step_embed_dim=16,
        n_groups=4,
        num_inference_steps=3,
    )
    policy = DiffusionPolicy(config).eval()

    class Predict(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = policy

        def forward(self, state, image, noise):
            return self.policy.predict_action_chunk({OBS_STATE: state, CAMERA: image}, noise=noise)

    def inputs():
        return torch.rand(2, 2, 4), torch.rand(2, 2, 3, 40, 40), torch.randn(2, config.horizon, 2)

    assert_export_matches_eager(Predict(), inputs(), inputs())


def test_exported_multi_task_dit_policy_matches_eager(monkeypatch):
    pytest.importorskip("transformers")
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPVisionConfig, CLIPVisionModel

    tiny = {"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2}
    vision = CLIPVisionConfig(**tiny, image_size=32, patch_size=8)
    text = CLIPTextConfig(**tiny, max_position_embeddings=8)
    # Tiny random CLIP encoders, so the test needs no download.
    monkeypatch.setattr(CLIPVisionModel, "from_pretrained", lambda name: CLIPVisionModel(vision))
    monkeypatch.setattr(CLIPTextModel, "from_pretrained", lambda name: CLIPTextModel(text))
    config = MultiTaskDiTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 36, 36)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        image_crop_shape=(32, 32),
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        num_inference_steps=3,
    )
    policy = MultiTaskDiTPolicy(config).eval()

    class Predict(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = policy

        def forward(self, state, image, tokens, mask, noise):
            batch = {OBS_STATE: state, CAMERA: image}
            batch |= {OBS_LANGUAGE_TOKENS: tokens, OBS_LANGUAGE_ATTENTION_MASK: mask}
            # predict_action_chunk reads the observations from the queues that select_action fills.
            self.policy.reset()
            batch = self.policy._prepare_batch(batch)
            populate_queues(self.policy._queues, batch)
            return self.policy.predict_action_chunk(batch, noise=noise)

    def inputs():
        tokens = torch.randint(3, text.vocab_size, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.long)
        return torch.rand(2, 4), torch.rand(2, 3, 36, 36), tokens, mask, torch.randn(2, config.horizon, 2)

    assert_export_matches_eager(Predict(), inputs(), inputs())
