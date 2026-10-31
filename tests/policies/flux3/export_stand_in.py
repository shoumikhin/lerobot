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
"""A stand-in FLUX3 frame policy for its export recipe, built in memory."""

from types import SimpleNamespace

import torch

from lerobot.policies.factory import get_export_recipe_class
from lerobot.policies.flux3 import Flux3Policy, make_flux3_pre_post_processors
from tests.policies.flux3.helpers import FakeVideoVAE, MockTextEncoder, single_config


def make_recipe(monkeypatch, task: str = "pick the cube"):
    monkeypatch.setattr(Flux3Policy, "_build_video_vae", lambda self, config: FakeVideoVAE())
    monkeypatch.setattr(
        Flux3Policy,
        "_build_text_encoder",
        lambda self, config: MockTextEncoder(self.dit_params.context_in_dim),
    )
    torch.manual_seed(0)
    config = single_config(n_action_steps=8)
    preprocessor, postprocessor = make_flux3_pre_post_processors(config)
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task=task, tolerance=None
    )
    return get_export_recipe_class(config)(Flux3Policy(config).eval(), preprocessor, postprocessor, cfg)
