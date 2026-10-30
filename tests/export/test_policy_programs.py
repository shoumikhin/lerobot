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

"""Each policy's export recipe on a stand-in policy, in memory: its program computes what the eager module the
recipe checked does, and the recipe refuses what its program cannot compute."""

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
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@dataclass
class Recipe:
    """A policy's recipe, which `make_recipe` in `module` builds from the fixtures that `uses` names.

    `cuda` also runs the recipe on a GPU, where its pixels must still be the processor's bit for bit.
    """

    module: str
    uses: tuple[str, ...] = ()
    cuda: bool = False
    marks: tuple[pytest.MarkDecorator, ...] = ()


RECIPES = {
    "wall_x": Recipe("tests.policies.wall_x.export_stand_in"),
    "gaussian_actor": Recipe("tests.export.test_export_gaussian_actor", marks=(OLD_TORCH,)),
    # Its language recipe comes with the dataset extra.
    "eo1": Recipe(
        "tests.policies.eo1.export_stand_in", ("monkeypatch",), cuda=True, marks=(pytest.mark.dataset,)
    ),
}


def make_recipe(request: pytest.FixtureRequest, name: str, **kwargs) -> ActionChunkRecipe:
    recipe = RECIPES[name]
    fixtures = [request.getfixturevalue(fixture) for fixture in recipe.uses]
    return import_module(recipe.module).make_recipe(*fixtures, **kwargs)


@pytest.mark.parametrize(
    ("name", "device"),
    [
        pytest.param(name, device, marks=[*recipe.marks, *([CUDA] if device == "cuda" else [])])
        for name, recipe in RECIPES.items()
        for device in ["cpu", "cuda"][: 1 + recipe.cuda]
    ],
)
def test_the_exported_program_computes_what_the_policy_does(request, name, device):
    built = make_recipe(request, name, **({} if device == "cpu" else {"device": device}))

    # Checks the eager module against the recipe's reference on two frames: the policy as lerobot-rollout runs
    # it, or the recipe's own chunk for a policy whose recipe computes it otherwise.
    [part] = built.parts()
    program = torch.export.export(part.module, part.example_inputs).module()

    assert NOISE in part.inputs
    # A third frame and another noise, which the program must read rather than hold.
    frame = random_frame(built.config.input_features, torch.Generator().manual_seed(1))
    noise = np.random.default_rng(2).standard_normal(built.test_case().noise.shape, np.float32)
    inputs = tuple(torch.from_numpy(x).to(device) for x in (*frame.values(), noise))
    with torch.no_grad():
        torch.testing.assert_close(program(*inputs), part.module(*inputs), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("name", "change", "error"),
    [
        pytest.param("eo1", {"side": 64}, "64x64 becomes 56x56", id="eo1-camera", marks=pytest.mark.dataset),
        pytest.param(
            "eo1",
            {"use_fast_processor": True},
            "use_fast_processor=True",
            id="eo1-fast-processor",
            marks=pytest.mark.dataset,
        ),
    ],
)
def test_the_recipe_refuses_what_its_program_cannot_compute(request, name, change, error):
    with pytest.raises(ValueError, match=error):
        list(make_recipe(request, name, **change).parts())
