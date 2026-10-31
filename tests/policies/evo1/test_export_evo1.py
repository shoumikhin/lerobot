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

"""The EVO1 export recipe, on a tiny EVO1 whose vision-language model is a stand-in."""

import math
from types import SimpleNamespace

import torch
from torch import nn

import lerobot.policies.evo1.evo1_model as evo1_model
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.export.recipe import ActionChunkRecipe
from lerobot.policies import prepare_observation_for_inference
from lerobot.policies.evo1.configuration_evo1 import Evo1Config
from lerobot.policies.evo1.modeling_evo1 import Evo1Policy
from lerobot.policies.evo1.processor_evo1 import make_evo1_pre_post_processors
from lerobot.policies.factory import get_export_recipe_class
from lerobot.utils.constants import ACTION, OBS_STATE

CAMERA = "observation.images.front"
CHUNK = 4
MAX_ACTION_DIM = 5
WIDTH = 8


class InternVL(nn.Module):
    """The parts of InternVL3 that EVO1 freezes or trains."""

    def __init__(self):
        super().__init__()
        self.language_model = nn.Identity()
        self.vision_tower = nn.Linear(3, WIDTH)
        self.multi_modal_projector = nn.Identity()


class Embedder(nn.Module):
    """Stands in for EVO1's InternVL3 embedder: one context token per camera, from its colors and the prompt."""

    def __init__(self, **kwargs):
        super().__init__()
        self.model = InternVL()

    def get_fused_image_text_embedding_batched(
        self, camera_images, image_masks, text_prompts, return_cls_only
    ):
        colors = torch.stack([image.mean(dim=(2, 3)) for image in camera_images], dim=1)
        prompts = torch.tensor([[[len(text)]] for text in text_prompts], device=colors.device)
        return self.model.vision_tower(colors) + prompts, image_masks[:, : len(camera_images)]


def make_recipe(monkeypatch, task: str = "pick up the cube") -> ActionChunkRecipe:
    monkeypatch.setattr(evo1_model, "InternVL3Embedder", Embedder)
    torch.manual_seed(0)
    config = Evo1Config(
        device="cpu",
        vlm_model_name="stand-in",
        chunk_size=CHUNK,
        n_action_steps=3,
        max_state_dim=6,
        max_action_dim=MAX_ACTION_DIM,
        max_views=2,
        embed_dim=WIDTH,
        hidden_dim=16,
        state_hidden_dim=16,
        num_heads=2,
        num_layers=1,
        num_inference_timesteps=2,
        input_features={
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    stats = {
        OBS_STATE: {"min": torch.full((4,), -1.0), "max": torch.ones(4)},
        ACTION: {"min": torch.zeros(3), "max": torch.tensor([2.0, 4.0, 6.0])},
    }
    preprocessor, postprocessor = make_evo1_pre_post_processors(config, stats)
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task=task, tolerance=None
    )
    return get_export_recipe_class(config)(Evo1Policy(config).eval(), preprocessor, postprocessor, cfg)


def test_the_noise_input_takes_the_place_of_evo1s_uniform_draw(monkeypatch):
    recipe = make_recipe(monkeypatch)
    assert recipe.test_case().noise.shape == (CHUNK, MAX_ACTION_DIM)
    [part] = recipe.parts()  # checks the program against the policy, bit for bit, on two frames
    assert part.inputs == [CAMERA, OBS_STATE, "noise"]
    observation = prepare_observation_for_inference(
        dict(recipe.test_case().frame), torch.device("cpu"), recipe.task
    )
    batch = recipe.preprocessor(observation)
    torch.manual_seed(1)
    drawn = recipe.policy.predict_action_chunk(batch)
    torch.manual_seed(1)
    uniform = torch.rand(1, CHUNK * MAX_ACTION_DIM) * 2 - 1
    normal = (torch.special.erfinv(uniform) * math.sqrt(2)).reshape(1, CHUNK, MAX_ACTION_DIM)
    # The normal noise that maps to EVO1's own uniform draw gives the chunk EVO1 predicts from that draw.
    torch.testing.assert_close(recipe.predict_chunk(recipe.policy, batch, normal), drawn)
