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

"""Each policy's export recipe on a stand-in policy, in memory: its program computes what the policy does,
and the recipe refuses what its program cannot compute."""

from dataclasses import dataclass
from importlib import import_module

import numpy as np
import pytest
import torch
from packaging.version import Version

from lerobot.bundle.manifest import NOISE
from lerobot.export.recipe import (
    BFLOAT16_STEPS_TOLERANCE,
    BFLOAT16_TOLERANCE,
    ActionChunkRecipe,
    random_frame,
)
from lerobot.utils.constants import OBS_STATE

# torch.export in PyTorch 2.10 cannot trace these policies' draws with the noise input in their place, and an
# export needs PyTorch 2.15 anyway.
OLD_TORCH = pytest.mark.skipif(Version(torch.__version__).release < (2, 15), reason="needs PyTorch 2.15")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@dataclass
class Recipe:
    """A policy's recipe, which `make_recipe` in `module` builds from the fixtures that `uses` names.

    `state` is the state of the frame the program is compared on, for a policy that writes the state into its
    prompt: its numbers take other widths there than in the frames the recipe checked. `cuda` also runs the
    recipe on a GPU, where its pixels must still be the processor's bit for bit. `untraceable` is the error
    that tracing the saved preprocessor gives, for a recipe made to avoid it.
    """

    module: str
    uses: tuple[str, ...] = ()
    state: list[float] | None = None
    cuda: bool = False
    untraceable: tuple[type[Exception], str] | None = None
    bfloat16_tolerance: float = BFLOAT16_TOLERANCE
    marks: tuple[pytest.MarkDecorator, ...] = ()


NUMPY_PIXELS = (RuntimeError, r"\.numpy\(\) is not supported for tensor subclasses")
RECIPES = {
    "gaussian_actor": Recipe("tests.export.test_export_gaussian_actor", marks=(OLD_TORCH,)),
    "lawam": Recipe("tests.policies.lawam.export_stand_in", untraceable=(Exception, "data-dependent")),
    "wall_x": Recipe(
        "tests.policies.wall_x.export_stand_in",
        untraceable=(Exception, "data-dependent|GuardOnDataDependentSymNode"),
    ),
    "eo1": Recipe(
        "tests.policies.eo1.export_stand_in",
        ("monkeypatch",),
        cuda=True,
        untraceable=NUMPY_PIXELS,
        bfloat16_tolerance=BFLOAT16_STEPS_TOLERANCE,
    ),
    # Bins 0, 5 and 191: numbers of one, one and three digits, where the checked frames' take three each.
    "dm05": Recipe(
        "tests.policies.dm05.export_stand_in",
        ("tmp_path",),
        state=[-1.0, -0.96, 0.5],
        cuda=True,
        bfloat16_tolerance=BFLOAT16_STEPS_TOLERANCE,
    ),
    # Bins -1, 6 and 255: numbers of two, one and three tokens.
    "fineart_vla": Recipe(
        "tests.policies.fineart_vla.export_stand_in",
        ("tmp_path",),
        state=[-2.0, -0.95, 1.0],
        bfloat16_tolerance=BFLOAT16_STEPS_TOLERANCE,
    ),
    # Bins -1, 0 and 511: numbers of two, one and three tokens, so the padding before the prompt changes.
    "wall_oss_05": Recipe(
        "tests.policies.wall_oss_05.export_stand_in",
        ("tmp_path",),
        state=[-2.0, -0.999, 1.0],
        cuda=True,
        bfloat16_tolerance=BFLOAT16_STEPS_TOLERANCE,
    ),
    "molmoact2": Recipe(
        "tests.policies.molmoact2.export_stand_in",
        ("tmp_path",),
        cuda=True,
        untraceable=NUMPY_PIXELS,
        bfloat16_tolerance=BFLOAT16_STEPS_TOLERANCE,
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
    recipe = RECIPES[name]
    built = make_recipe(request, name, **({} if device == "cpu" else {"device": device}))

    # Checks the eager module against the policy as lerobot-rollout runs it, on two frames.
    [part] = built.parts()
    program = torch.export.export(part.module, part.example_inputs).module()

    assert NOISE in part.inputs
    assert built.bfloat16_tolerance == recipe.bfloat16_tolerance
    # A third frame and another noise, which the program must read rather than hold.
    frame = random_frame(built.config.input_features, torch.Generator().manual_seed(1))
    if recipe.state is not None:
        frame[OBS_STATE] = np.array(recipe.state, dtype=np.float32)
    noise = np.random.default_rng(2).standard_normal(built.test_case().noise.shape, np.float32)
    inputs = tuple(torch.from_numpy(x).to(device) for x in (*frame.values(), noise))
    with torch.no_grad():
        torch.testing.assert_close(program(*inputs), part.module(*inputs), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("name", "change", "error"),
    [
        pytest.param("eo1", {"side": 64}, "64x64 becomes 56x56", id="eo1-camera"),
        # Tokenizers that read all the text between special tokens as one word, so a number alone is not its
        # own token there.
        pytest.param(
            "dm05", {"split_words": False}, "does not hold the state's numbers", id="dm05-tokenizer"
        ),
        pytest.param(
            "fineart_vla",
            {"split_words": False},
            "does not hold the state's numbers once",
            id="fineart_vla-tokenizer",
        ),
        pytest.param(
            "wall_oss_05",
            {"split_words": False},
            "does not hold the state's numbers once",
            id="wall_oss_05-tokenizer",
        ),
        pytest.param("wall_oss_05", {"size": 224}, "long side is 448 pixels", id="wall_oss_05-camera"),
        pytest.param("molmoact2", {"side": 32}, "32x32 becomes 28x28", id="molmoact2-camera"),
    ],
)
def test_the_recipe_refuses_what_its_program_cannot_compute(request, name, change, error):
    with pytest.raises(ValueError, match=error):
        list(make_recipe(request, name, **change).parts())


@pytest.mark.parametrize("name", [name for name, recipe in RECIPES.items() if recipe.untraceable])
def test_the_saved_preprocessor_cannot_be_traced(request, name):
    error, message = RECIPES[name].untraceable
    [part] = make_recipe(request, name, recipe_class=ActionChunkRecipe).parts()

    with pytest.raises(error, match=message):
        torch.export.export(part.module, part.example_inputs)
