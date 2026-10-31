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

import numpy as np
import pytest
import torch

from lerobot.policies.factory import get_export_recipe_class
from lerobot.policies.flux3 import Flux3Policy, make_flux3_pre_post_processors
from tests.policies.flux3.helpers import FakeVideoVAE, single_config


def make_recipe(monkeypatch, task: str = "pick the cube"):
    monkeypatch.setattr(Flux3Policy, "_build_video_vae", lambda self, config: FakeVideoVAE())
    torch.manual_seed(0)
    config = single_config(n_action_steps=8)
    preprocessor, postprocessor = make_flux3_pre_post_processors(config)
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task=task, tolerance=None
    )
    return get_export_recipe_class(config)(Flux3Policy(config).eval(), preprocessor, postprocessor, cfg)


@pytest.fixture
def recipe(monkeypatch, fake_text_encoder):
    return make_recipe(monkeypatch)


def test_the_bundle_holds_the_task_flux3_reads(monkeypatch, fake_text_encoder):
    with pytest.raises(ValueError, match="reads the task"):
        make_recipe(monkeypatch, task="")
    cube, stop = (make_recipe(monkeypatch, task) for task in ("pick the cube", "stop"))
    assert cube.task == "pick the cube"
    assert (cube.test_case().actions != stop.test_case().actions).any()


def test_the_program_holds_the_noise_flux3_seeds_on_every_call(recipe):
    # Building the part checks the program, which holds the noise, against the policy drawing its own noise,
    # bit for bit, on two frames.
    [part] = recipe.parts()
    assert part.inputs == ["observation.images.top", "observation.state"]
    assert recipe.test_case().noise is None


def test_the_program_exports_with_the_default_sampler(recipe):
    assert recipe.policy.config.sampler == "cosmos_unipc"
    [part] = recipe.parts()
    program = torch.export.export(part.module, part.example_inputs).module()
    with torch.no_grad():
        np.testing.assert_array_equal(program(*part.example_inputs).numpy(), recipe.test_case().actions)
