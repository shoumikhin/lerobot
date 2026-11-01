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

import importlib
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from packaging.version import Version

from lerobot.bundle.manifest import NOISE
from lerobot.export.recipe import ActionChunkRecipe, random_frame
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
    recipe on a GPU, where its pixels must still be the processor's bit for bit. `seeded` is for a policy that
    seeds its own noise, which the program holds, and `reads_task` for one whose bundle holds the task, as the
    recipe's own probe finds.
    `options` are more arguments of `make_recipe`.
    """

    module: str
    uses: tuple[str, ...] = ()
    state: list[float] | None = None
    cuda: bool = False
    seeded: bool = False
    reads_task: bool = False
    marks: tuple[pytest.MarkDecorator, ...] = ()
    options: dict = field(default_factory=dict)


RECIPES = {
    "wall_x": Recipe("tests.policies.wall_x.export_stand_in", reads_task=True),
    "gaussian_actor": Recipe("tests.export.test_export_gaussian_actor", marks=(OLD_TORCH,)),
    # Bins 0, 5 and 192: numbers of one, one and three digits, where the checked frames' take three each.
    "fineart_vla": Recipe(
        "tests.policies.fineart_vla.export_stand_in",
        ("monkeypatch",),
        state=[-1.0, -0.96, 0.5],
        reads_task=True,
        marks=(pytest.mark.dataset,),
    ),
    # Bins 0, 10 and 384: numbers of one, two and three digits, where the checked frames' take three each.
    "wall_oss_05": Recipe(
        "tests.policies.wall_oss_05.export_stand_in", state=[-1.0, -0.96, 0.5], reads_task=True
    ),
    # A camera the image processor resizes, which it does otherwise on float pixels than on the step's uint8.
    "wall_oss_05_250x448": Recipe(
        "tests.policies.wall_oss_05.export_stand_in", options={"short_side": 250}, reads_task=True
    ),
    "molmoact2": Recipe("tests.policies.molmoact2.export_stand_in", ("monkeypatch",), reads_task=True),
    "molmoact2_state_mask": Recipe(
        "tests.policies.molmoact2.export_stand_in",
        ("monkeypatch",),
        options={"state_mask": True},
        reads_task=True,
    ),
    # Its language recipe comes with the dataset extra.
    "eo1": Recipe(
        "tests.policies.eo1.export_stand_in", ("monkeypatch",), cuda=True, marks=(pytest.mark.dataset,)
    ),
    # Bins 0, 5 and 191: numbers of one, one and three digits, where the checked frames' take three each.
    "dm05": Recipe(
        "tests.policies.dm05.export_stand_in",
        ("tmp_path",),
        state=[-1.0, -0.96, 0.5],
        cuda=True,
        reads_task=True,
    ),
    "flux3": Recipe("tests.policies.flux3.export_stand_in", ("monkeypatch",), seeded=True, reads_task=True),
    "evo1": Recipe(
        "tests.policies.evo1.test_export_evo1", ("monkeypatch",), reads_task=True, marks=(OLD_TORCH,)
    ),
    "fastwam": Recipe(
        "tests.policies.fastwam.test_export_fastwam", ("monkeypatch",), seeded=True, reads_task=True
    ),
}


def make_recipe(request: pytest.FixtureRequest, name: str, **kwargs) -> ActionChunkRecipe:
    recipe = RECIPES[name]
    fixtures = [request.getfixturevalue(fixture) for fixture in recipe.uses]
    return importlib.import_module(recipe.module).make_recipe(*fixtures, **recipe.options, **kwargs)


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

    # Checks the eager module against the recipe's reference on two frames: the policy as lerobot-rollout runs
    # it, or the recipe's own chunk for a policy whose recipe computes it otherwise.
    [part] = built.parts()
    program = torch.export.export(part.module, part.example_inputs).module()

    assert (NOISE not in part.inputs) == (built.test_case().noise is None) == recipe.seeded
    # A third frame, and for a policy that does not seed its own noise another noise, which the program must
    # read rather than hold.
    frame = random_frame(built.config.input_features, torch.Generator().manual_seed(1))
    assert built._reads_task(frame) == recipe.reads_task
    if recipe.state is not None:
        frame[OBS_STATE] = np.array(recipe.state, dtype=np.float32)
    shape = None if recipe.seeded else built.test_case().noise.shape
    noise = [] if shape is None else [np.random.default_rng(2).standard_normal(shape, np.float32)]
    inputs = tuple(torch.from_numpy(x).to(device) for x in (*frame.values(), *noise))
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
        pytest.param("dm05", {"add_state": False}, "add_state=false", id="dm05-no-state"),
        pytest.param("fineart_vla", {"text_recipe": False}, "has no text recipe", id="fineart-no-recipe"),
        pytest.param("wall_oss_05", {"long_side": 224}, "long side is 448 pixels", id="wall_oss_05-camera"),
        pytest.param(
            "molmoact2", {"crop_mode": "overlap-and-resize"}, "uses 'overlap-and-resize'", id="molmoact2-crop"
        ),
        pytest.param("molmoact2", {"width": 42}, "28x42 becomes 28x28", id="molmoact2-camera"),
    ],
)
def test_the_recipe_refuses_what_its_program_cannot_compute(request, name, change, error):
    with pytest.raises(ValueError, match=error):
        list(make_recipe(request, name, **change).parts())


@pytest.mark.parametrize(
    ("name", "broken", "error"),
    [
        pytest.param("dm05", "prompt", "writes another prompt", id="dm05-prompt"),
        pytest.param("dm05", "pixels", "pixel values differ", id="dm05-pixels"),
        pytest.param(
            "fineart_vla",
            "prompt",
            "writes another prompt",
            id="fineart_vla-prompt",
            marks=pytest.mark.dataset,
        ),
        pytest.param("wall_oss_05", "prompt", "writes another prompt", id="wall_oss_05-prompt"),
        pytest.param("wall_oss_05", "pixels", "pixel_values differ", id="wall_oss_05-pixels"),
    ],
)
def test_the_checks_refuse_another_prompt_or_other_pixels(request, monkeypatch, name, broken, error):
    """A program whose prompt or pixels are not the saved steps' is refused. Here the first bin's number gets
    another number's tokens, as from a tokenizer that splits it otherwise alone, or the pixels move."""
    step_type = importlib.import_module(f"lerobot.policies.{name}.export_{name}")._StatePromptStep
    init = step_type.__init__

    def broken_init(self, *args):
        init(self, *args)
        if broken == "prompt":
            self.pieces[0] = self.pieces[1]
        else:
            process = self.image_processor
            self.image_processor = lambda *a, **k: {"pixel_values": process(*a, **k)["pixel_values"] + 1}

    monkeypatch.setattr(step_type, "__init__", broken_init)
    with pytest.raises(ValueError, match=error):
        list(make_recipe(request, name).parts())


@pytest.mark.parametrize("name", [name for name, recipe in RECIPES.items() if recipe.reads_task])
def test_the_bundle_holds_the_task_the_policy_reads(request, name):
    with pytest.raises(ValueError, match="reads the task"):
        make_recipe(request, name, task="")
    cube, stop = (make_recipe(request, name, task=task) for task in ("pick up the cube", "stop"))
    assert cube.task == "pick up the cube"
    assert not np.array_equal(cube.test_case().actions, stop.test_case().actions)
