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

"""Each policy's export recipe on a stand-in policy, in memory: its program computes what the policy does."""

from dataclasses import dataclass
from importlib import import_module

import numpy as np
import pytest
import torch
from packaging.version import Version

from lerobot.bundle.manifest import NOISE
from lerobot.export.recipe import ActionChunkRecipe, random_frame

# torch.export in PyTorch 2.10 cannot trace these policies' draws with the noise input in their place, and an
# export needs PyTorch 2.15 anyway.
OLD_TORCH = pytest.mark.skipif(Version(torch.__version__).release < (2, 15), reason="needs PyTorch 2.15")


@dataclass
class Recipe:
    """A policy's recipe, which `make_recipe` in `module` builds."""

    module: str
    marks: tuple[pytest.MarkDecorator, ...] = ()


RECIPES = {
    "gaussian_actor": Recipe("tests.export.test_export_gaussian_actor", marks=(OLD_TORCH,)),
}


def make_recipe(name: str, **kwargs) -> ActionChunkRecipe:
    return import_module(RECIPES[name].module).make_recipe(**kwargs)


@pytest.mark.parametrize("name", [pytest.param(name, marks=recipe.marks) for name, recipe in RECIPES.items()])
def test_the_exported_program_computes_what_the_policy_does(name):
    built = make_recipe(name)

    # Checks the eager module against the policy as lerobot-rollout runs it, on two frames.
    [part] = built.parts()
    program = torch.export.export(part.module, part.example_inputs).module()

    assert NOISE in part.inputs
    # A third frame and another noise, which the program must read rather than hold.
    frame = random_frame(built.config.input_features, torch.Generator().manual_seed(1))
    noise = np.random.default_rng(2).standard_normal(built.test_case().noise.shape, np.float32)
    inputs = tuple(torch.from_numpy(x) for x in (*frame.values(), noise))
    with torch.no_grad():
        torch.testing.assert_close(program(*inputs), part.module(*inputs), rtol=0, atol=0)
