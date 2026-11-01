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

"""The FastWAM export recipe, on a FastWAM whose core is a stand-in."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers", reason="fastwam requires the `fastwam` extra (transformers)")
pytest.importorskip("diffusers", reason="fastwam requires the `fastwam` extra (diffusers)")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_export_recipe_class
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
from lerobot.policies.fastwam.processor_fastwam import make_fastwam_pre_post_processors
from lerobot.policies.fastwam.wan import FastWAM
from lerobot.utils.constants import ACTION, OBS_STATE

IMAGE = "observation.images.image"


class SeededCore(nn.Module):
    """Stands in for FastWAM's core: draws its starting noise with FastWAM's own code and reads the prompt."""

    _make_action_latents = FastWAM._make_action_latents

    def __init__(self):
        super().__init__()
        self.dit = nn.Linear(3, 3)
        self.action_expert = SimpleNamespace(action_dim=3)
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32

    def encode_prompt(self, prompt):
        # The policy encodes the task once and passes the embeddings as the context.
        return torch.full((1, 1, 3), float(len(str(prompt)))), torch.ones(1, 1, dtype=torch.bool)

    def infer_action(
        self, input_image, action_horizon, seed, rand_device, noise=None, context=None, **kwargs
    ):
        noise = self._make_action_latents(action_horizon, seed, rand_device, noise=noise)
        return {"action": self.dit(noise) + input_image.mean() + context.mean()}


def make_recipe(monkeypatch, task: str = "pick up the cube", inference_seed: int | None = 42):
    monkeypatch.setattr(FastWAMPolicy, "_build_core_model", lambda self, config: SeededCore())
    torch.manual_seed(0)
    config = FastWAMConfig(
        action_dim=3,
        proprio_dim=2,
        action_horizon=4,
        n_action_steps=2,
        num_video_frames=5,
        action_video_freq_ratio=1,
        image_size=(16, 16),
        device="cpu",
        base_model_id=None,
        inference_seed=inference_seed,
        input_features={
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    preprocessor, postprocessor = make_fastwam_pre_post_processors(config)
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task=task, tolerance=None
    )
    return get_export_recipe_class(config)(FastWAMPolicy(config).eval(), preprocessor, postprocessor, cfg)


def test_without_a_seed_the_bundle_draws_fastwams_noise(monkeypatch):
    [part] = make_recipe(monkeypatch, inference_seed=None).parts()
    assert part.inputs == [IMAGE, OBS_STATE, "noise"]
