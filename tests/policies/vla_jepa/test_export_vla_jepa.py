#!/usr/bin/env python

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

"""The VLA-JEPA export recipe, on VLA-JEPA's preprocessor and a tiny random Qwen3-VL saved with its processor."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from lerobot.export.recipe import ExportRecipe, random_frame  # noqa: E402
from lerobot.policies.factory import get_export_recipe_class  # noqa: E402
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig  # noqa: E402
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy  # noqa: E402
from lerobot.policies.vla_jepa.processor_vla_jepa import make_vla_jepa_pre_post_processors  # noqa: E402


def make_recipe(config: VLAJEPAConfig, task: str | None = "pick up the cube") -> ExportRecipe:
    """The recipe `lerobot-export` picks for `config`."""
    torch.manual_seed(0)
    preprocessor, postprocessor = make_vla_jepa_pre_post_processors(config)
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task=task, tolerance=None
    )
    return get_export_recipe_class(config)(VLAJEPAPolicy(config).eval(), preprocessor, postprocessor, cfg)


def program_inputs(recipe: ExportRecipe) -> tuple[torch.Tensor, ...]:
    frame = random_frame(recipe.config.input_features, torch.Generator().manual_seed(1))
    noise = torch.randn(recipe.test_case().noise.shape, generator=torch.Generator().manual_seed(2))
    return tuple(torch.from_numpy(x) for x in (*frame.values(), noise.numpy()))


def test_the_program_keeps_the_prompt_and_computes_the_pixels(tiny_qwen_config: VLAJEPAConfig) -> None:
    recipe = make_recipe(tiny_qwen_config)

    [part] = recipe.parts()
    program = torch.export.export(part.module, part.example_inputs, strict=False)

    # Another frame than the one traced: the program reads its pixels.
    inputs = program_inputs(recipe)
    with torch.no_grad():
        torch.testing.assert_close(program.module()(*inputs), part.module(*inputs), rtol=0, atol=0)


def test_the_task_must_be_given(tiny_qwen_config: VLAJEPAConfig) -> None:
    # The prompt holds the task, so a program without one would act on an empty task.
    with pytest.raises(ValueError, match="reads the task"):
        make_recipe(tiny_qwen_config, task=None)
