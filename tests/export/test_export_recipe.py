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


"""The default export recipe, on a tiny ACT built in memory."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import Tensor

import lerobot.policies.factory as factory
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.export.recipe import ActionChunkRecipe, ReplayDraws, frame_shapes, robot_features
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor import ProcessorStep, TransitionKey
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

IMAGE = f"{OBS_IMAGES}.front"
STATS = {
    IMAGE: {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.2)},
    OBS_STATE: {"mean": torch.zeros(3), "std": torch.ones(3)},
    ACTION: {"mean": torch.full((3,), 10.0), "std": torch.tensor([30.0, 3.0, 0.3])},
}


def act_config() -> ACTConfig:
    return ACTConfig(
        input_features={
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
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


def export_config(task: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task=task, tolerance=None
    )


def make(
    policy_class=ACTPolicy, task: str = "", extra_step: ProcessorStep | None = None
) -> ActionChunkRecipe:
    torch.manual_seed(0)
    config = act_config()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=STATS)
    if extra_step is not None:
        preprocessor.steps.insert(0, extra_step)
    return ActionChunkRecipe(policy_class(config).eval(), preprocessor, postprocessor, export_config(task))


class TaskCodes(ProcessorStep):
    """Reads the task like a tokenizer that keeps only its first `limit` characters."""

    limit = 4

    def __call__(self, transition):
        task = transition[TransitionKey.COMPLEMENTARY_DATA]["task"]
        codes = [ord(c) for c in task[: self.limit].ljust(self.limit)]
        observation = {**transition[TransitionKey.OBSERVATION], "observation.task_codes": torch.tensor(codes)}
        return {**transition, TransitionKey.OBSERVATION: observation}

    def transform_features(self, features):
        return features


class NoisyACT(ACTPolicy):
    """ACT that adds a starting noise of its own shape, (chunk_size, 5), to its actions."""

    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = torch.randn(1, self.config.chunk_size, 5)
        return super().predict_action_chunk(batch) + noise[..., :1]


class TwoNoisesACT(ACTPolicy):
    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        return super().predict_action_chunk(batch) + torch.randn(1, 4, 1) + torch.randn(1, 4, 1)


class SamplingACT(ACTPolicy):
    """ACT that adds a category it samples from its actions, as a policy that samples codes would."""

    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        actions = super().predict_action_chunk(batch)
        return actions + torch.multinomial(actions[0].softmax(-1), 1).float()


def test_recipe_measures_the_action_scale_from_the_postprocessor():
    np.testing.assert_allclose(make().test_case().action_scale, [30.0, 3.0, 0.3], rtol=1e-6)


def test_recipe_keeps_a_task_given_even_when_a_tokenizer_cuts_it_short():
    recipe = make(task="pick cube pick cube", extra_step=TaskCodes())
    assert recipe.task == "pick cube pick cube"
    with pytest.raises(ValueError, match="reads the task"):
        make(extra_step=TaskCodes())
    assert make().task is None


def test_recipe_asks_the_policy_for_its_noise_shape():
    recipe = make(NoisyACT)
    assert recipe.test_case().noise.shape == (4, 5)
    [part] = recipe.parts()
    assert part.inputs[-1] == "noise"


def test_recipe_refuses_a_policy_that_draws_two_noises():
    with pytest.raises(ValueError, match="draws 2 random tensors"):
        make(TwoNoisesACT)


def test_recipe_refuses_a_policy_that_samples_categories_itself():
    with pytest.raises(ValueError, match=r"draws 1 random tensors shaped \[\(2, 4, 3\)\]"):
        make(SamplingACT)


def test_replayed_draws_read_the_noise_in_order():
    noise = torch.arange(6.0).reshape(3, 1, 2)
    with ReplayDraws(noise) as draws:
        drawn = [torch.randn(1, 2), torch.randn_like(torch.ones(1, 2)), torch.ones(1, 2).normal_()]
    assert torch.equal(torch.stack(drawn), noise) and draws.shapes == [(1, 2)] * 3
    with pytest.raises(ValueError, match=r"Draw 0 reads noise shaped \[1, 3\]"), ReplayDraws(noise):
        torch.randn(1, 3)
    with pytest.raises(ValueError, match="Draw 3"), ReplayDraws(noise):
        for _ in range(4):
            torch.randn(1, 2)


def test_pytorch_samples_one_category_by_an_exponential_race():
    """What `ReplayDraws` relies on to sample categories as PyTorch does."""
    probs = torch.softmax(torch.randn(64, 16) / 0.3, -1)
    for seed in range(20):
        sampled = torch.multinomial(probs, 1, generator=torch.Generator().manual_seed(seed))
        race = torch.empty_like(probs).exponential_(generator=torch.Generator().manual_seed(seed))
        assert torch.equal(sampled, torch.argmax(probs / race, dim=-1, keepdim=True))


def test_replayed_categories_follow_the_probabilities():
    probs = torch.tensor([[0.1, 0.2, 0.7]]).expand(20000, 3)
    noise = torch.randn(1, 2, 20000, 3, generator=torch.Generator().manual_seed(0))
    with ReplayDraws(noise):
        sampled = torch.multinomial(probs, num_samples=1)
    counts = torch.bincount(sampled[:, 0], minlength=3) / 20000
    torch.testing.assert_close(counts, probs[0], atol=0.01, rtol=0)


def test_a_program_samples_from_its_noise_input():
    class Sample(torch.nn.Module):
        def forward(self, logits: Tensor, noise: Tensor) -> Tensor:
            with ReplayDraws(noise):
                return torch.multinomial(logits.softmax(-1), 1)

    logits, noise = torch.randn(4, 8), torch.randn(1, 2, 4, 8)
    program = torch.export.export(Sample(), (logits, noise))
    assert "multinomial" not in str(program.graph)
    assert torch.equal(program.module()(logits, noise), Sample()(logits, noise))


def test_recipe_does_not_ask_the_dataset_for_empty_camera_slots():
    empty = f"{OBS_IMAGES}.empty_camera_0"
    config = SmolVLAConfig(
        empty_cameras=1,
        input_features={
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            empty: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    metadata = SimpleNamespace(
        features={
            IMAGE: {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]},
            OBS_STATE: {"dtype": "float32", "shape": (3,), "names": None},
        },
    )
    features = robot_features(config, SimpleNamespace(steps=[]), metadata)
    assert frame_shapes(features) == {IMAGE: (32, 32, 3), OBS_STATE: (3,)}


def test_an_image_recorded_channels_last_keeps_its_shape():
    features = {
        IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 24, 32)),
        f"{IMAGE}_hwc": PolicyFeature(type=FeatureType.VISUAL, shape=(24, 32, 3)),
    }
    assert frame_shapes(features) == {IMAGE: (24, 32, 3), f"{IMAGE}_hwc": (24, 32, 3)}


def test_a_policy_without_an_export_module_gets_the_one_program_recipe():
    assert factory.get_export_recipe_class(act_config()) is ActionChunkRecipe


def test_a_policy_export_module_must_name_its_recipe(monkeypatch):
    monkeypatch.setattr(
        factory, "_import_sibling_policy_module", lambda *args: SimpleNamespace(__name__="export_act")
    )
    with pytest.raises(ValueError, match="export_act must define the export recipe ACTExportRecipe"):
        factory.get_export_recipe_class(act_config())
