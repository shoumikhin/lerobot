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
"""The FLUX3 export recipe, on a tiny frame checkpoint built in memory."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.policies import prepare_observation_for_inference
from lerobot.policies.factory import get_export_recipe_class
from lerobot.policies.flux3 import Flux3Policy, make_flux3_pre_post_processors
from tests.policies.flux3.helpers import FakeVideoVAE, single_config


@pytest.fixture
def recipe(monkeypatch, fake_text_encoder):
    monkeypatch.setattr(Flux3Policy, "_build_video_vae", lambda self, config: FakeVideoVAE())
    torch.manual_seed(0)
    config = single_config(n_action_steps=8)
    preprocessor, postprocessor = make_flux3_pre_post_processors(config)
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task="pick the cube", tolerance=None
    )
    return get_export_recipe_class(config)(Flux3Policy(config).eval(), preprocessor, postprocessor, cfg)


def test_one_noise_input_takes_the_place_of_both_draws(recipe):
    [part] = recipe.parts()  # checks the program against the policy, bit for bit, on two frames
    assert part.inputs == ["observation.images.top", "observation.state", "noise"]
    policy, config = recipe.policy, recipe.policy.config
    generator = torch.Generator().manual_seed(config.inference_seed)
    n_pred = policy.packer.predicted_latent_frames(config)
    video = torch.randn(1, 96, n_pred, *config.latent_hw, generator=generator)
    action = torch.randn(1, config.action_dim, config.chunk_size, generator=generator)
    noise = torch.cat([video.flatten(), action.flatten()])
    assert noise.shape == recipe.test_case().noise.shape
    observation = prepare_observation_for_inference(
        dict(recipe.test_case().frame), torch.device("cpu"), recipe.task
    )
    batch = recipe.preprocessor(observation)
    # The policy's own draws, joined in the order it draws them, give what it predicts when it draws them itself.
    torch.testing.assert_close(
        recipe.predict_chunk(policy, batch, noise[None]), policy.predict_action_chunk(batch), rtol=0, atol=0
    )
