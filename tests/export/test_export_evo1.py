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

"""The EVO1 export recipe gives EVO1 uniform starting noise, made from the bundle's normal noise."""

from types import SimpleNamespace

import torch

from lerobot.export.recipe import recipe_class
from lerobot.policies.evo1.configuration_evo1 import Evo1Config


class StartingNoise:
    """Returns the starting noise it is given, in place of EVO1's chunk."""

    def predict_action_chunk(self, batch, *, noise=None):
        return noise


def test_the_policy_starts_from_uniform_noise():
    recipe = recipe_class(Evo1Config())
    recipe_instance = object.__new__(recipe)
    recipe_instance.policy = SimpleNamespace(config=SimpleNamespace(chunk_size=50, max_action_dim=24))
    assert recipe_instance._sampled_noise_shape({}) == (50, 24)

    normal = torch.randn(1, 50, 24, generator=torch.Generator().manual_seed(0))
    uniform = recipe.predict_chunk(StartingNoise(), {}, normal)
    assert uniform.shape == normal.shape
    assert uniform.abs().max() < 1
    # A uniform distribution on (-1, 1) has mean 0 and variance 1/3.
    assert abs(uniform.mean()) < 0.05
    assert abs(uniform.var() - 1 / 3) < 0.03
    assert torch.equal(uniform.argsort(dim=-1), normal.argsort(dim=-1))
