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

from lerobot.bundle.manifest import NOISE  # noqa: E402
from lerobot.export.recipe import ExportRecipe, random_frame  # noqa: E402
from lerobot.policies.factory import get_export_recipe_class  # noqa: E402
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig  # noqa: E402
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy  # noqa: E402
from lerobot.policies.vla_jepa.processor_vla_jepa import make_vla_jepa_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402


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


def chained(recipe: ExportRecipe, modules: dict[str, torch.nn.Module], inputs: tuple[torch.Tensor, ...]):
    """The actions of `modules`, one per part, run in the order of the recipe's plan on the frame and noise."""
    values = dict(zip([*recipe._features, NOISE], inputs, strict=True))
    with torch.no_grad():
        for run in recipe.plan().runs:
            outputs = modules[run.program](*(values[name] for name in run.inputs))
            values.update(
                zip(run.outputs, outputs if isinstance(outputs, tuple) else (outputs,), strict=True)
            )
    return values[ACTION]


def test_the_programs_keep_the_prompt_and_compute_the_pixels(tiny_qwen_config: VLAJEPAConfig) -> None:
    recipe = make_recipe(tiny_qwen_config)

    parts = {part.name: part for part in recipe.parts()}
    programs = {
        name: torch.export.export(part.module, part.example_inputs, strict=False).module()
        for name, part in parts.items()
    }

    assert list(parts) == ["embed", "language_model_0", "actions"]
    # Another frame than the one traced: the programs read its pixels.
    inputs = program_inputs(recipe)
    eager = {name: part.module for name, part in parts.items()}
    torch.testing.assert_close(
        chained(recipe, programs, inputs), chained(recipe, eager, inputs), rtol=0, atol=0
    )


def test_each_program_holds_only_the_weights_it_runs(tiny_qwen_config: VLAJEPAConfig) -> None:
    # Compiled as one program, the 5 GiB of a real checkpoint need more memory to build than an 8 GB board has.
    recipe = make_recipe(tiny_qwen_config)
    model = recipe.policy.model
    weights = {p.data_ptr() for p in recipe.policy.parameters()}
    vision = {p.data_ptr() for p in model.qwen.model.model.visual.parameters()}
    action_head = {p.data_ptr() for p in model.action_model.parameters()}
    for part in recipe.parts():
        program = torch.export.export(part.module, part.example_inputs, strict=False)
        tensors = [*program.state_dict.values(), *program.constants.values()]
        held = {t.data_ptr() for t in tensors if isinstance(t, torch.Tensor)} & weights
        assert held == {p.data_ptr() for p in part.module.parameters()}, part.name
        assert not (held & vision and held & action_head), part.name


def test_the_task_must_be_given(tiny_qwen_config: VLAJEPAConfig) -> None:
    # The prompt holds the task, so a program without one would act on an empty task.
    with pytest.raises(ValueError, match="reads the task"):
        make_recipe(tiny_qwen_config, task=None)


def test_the_program_is_checked_against_the_inputs_the_policy_makes(
    tiny_qwen_config: VLAJEPAConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    # In rollout the policy makes its own Qwen inputs, which the program's step must match.
    qwen_inputs = VLAJEPAPolicy.qwen_inputs

    def another_task(self, batch):
        return qwen_inputs(self, {**batch, "task": [f"{task} slowly" for task in batch["task"]]})

    monkeypatch.setattr(VLAJEPAPolicy, "qwen_inputs", another_task)
    with pytest.raises(ValueError, match="lerobot-rollout"):
        list(make_recipe(tiny_qwen_config).parts())
