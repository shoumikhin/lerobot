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

"""The Gaussian actor export recipe, on a tiny actor built in memory."""

from types import SimpleNamespace

import numpy as np
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.export.recipe import recipe_class
from lerobot.policies import make_pre_post_processors
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import GaussianActorConfig
from lerobot.policies.gaussian_actor.export_gaussian_actor import GaussianActorExportRecipe
from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy
from lerobot.utils.constants import ACTION, OBS_STATE


def make() -> GaussianActorExportRecipe:
    torch.manual_seed(0)
    config = GaussianActorConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(5,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        normalization_mapping={"STATE": NormalizationMode.IDENTITY, "ACTION": NormalizationMode.IDENTITY},
        device="cpu",
    )
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=None)
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task="", tolerance=None
    )
    return recipe_class(config)(GaussianActorPolicy(config).eval(), preprocessor, postprocessor, cfg)


def test_the_program_samples_one_action_from_its_noise_input():
    recipe = make()
    case = recipe.test_case()
    assert case.noise.shape == (3,)
    assert case.actions.shape == (1, 3)
    [part] = recipe.parts()
    assert part.input_names == [OBS_STATE, "noise"]
    program = torch.export.export(part.module, part.example_inputs).module()
    with torch.no_grad():
        np.testing.assert_array_equal(program(*part.example_inputs).numpy(), case.actions)
        other = program(part.example_inputs[0], -part.example_inputs[1]).numpy()
    assert not np.array_equal(other, case.actions)


def test_the_noise_input_takes_the_place_of_the_actors_own_draw():
    recipe = make()
    [part] = recipe.parts()
    state = part.example_inputs[0]
    batch = recipe.preprocessor({OBS_STATE: state[None], "task": ""})
    torch.manual_seed(1)
    sampled = recipe.postprocessor(recipe.policy.select_action(batch))[0]
    torch.manual_seed(1)
    noise = torch.empty(1, 3).normal_()[0]
    with torch.no_grad():
        np.testing.assert_array_equal(part.module(state, noise).numpy()[0], sampled.numpy())
