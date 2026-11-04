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

"""pi0's own chain parts, on a tiny pi0 built in this process."""

import weakref

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies.pi0 import export_pi0  # noqa: E402
from lerobot.policies.pi05 import export_pi05  # noqa: E402
from tests.export.recipe_config import export_config  # noqa: E402
from tests.export.tiny_pi0 import save_checkpoint, use_tiny_pi0  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pi0")
    with pytest.MonkeyPatch.context() as patch:
        save_checkpoint(folder / "checkpoint", patch)
    return folder / "checkpoint"


@pytest.fixture
def recipe_config(checkpoint, monkeypatch):
    """What `lerobot-export` hands the recipe."""
    use_tiny_pi0(monkeypatch)
    policy = PreTrainedConfig.from_pretrained(checkpoint)
    policy.pretrained_path = checkpoint
    return export_config("pick up the cube", policy=policy, max_task_tokens=None)


def test_a_bfloat16_chain_computes_as_the_policy(recipe_config, monkeypatch):
    """The parts cast between the dtypes of their weights as the policy does, which float32 weights never test."""
    recipe_config.policy.dtype = torch.bfloat16
    # Where the policy does not fit in float32, the chain is compared with it in bfloat16.
    monkeypatch.setattr(export_pi05, "whole_policy_fits", lambda *args, float32=False: not float32)
    recipe = export_pi0.PI0ExportRecipe(recipe_config)
    list(recipe.parts())
    recipe.test_case()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_a_bfloat16_recipe_on_a_gpu_computes_as_the_policy(recipe_config, monkeypatch):
    """On a GPU, a bfloat16 pi0 runs its float32 vision tower under bfloat16 autocast. The recipe stores those
    linear weights in bfloat16, the same arithmetic, and its chain passes the bfloat16 check against the policy."""
    recipe_config.policy.dtype = torch.bfloat16
    recipe_config.policy.device = recipe_config.backend.device = "cuda"
    monkeypatch.setattr(export_pi05, "whole_policy_fits", lambda *args, float32=False: not float32)
    recipe = export_pi0.PI0ExportRecipe(recipe_config)
    list(recipe.parts())
    recipe.test_case()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_a_bfloat16_recipe_on_a_gpu_frees_the_vision_tower_once_the_embed_part_is_built(recipe_config):
    recipe_config.policy.dtype = torch.bfloat16
    recipe_config.policy.device = recipe_config.backend.device = "cuda"
    parts = export_pi0.PI0ExportRecipe(recipe_config).parts()
    embed = next(parts)
    tower = weakref.ref(embed.module.policy.model.paligemma_with_expert.paligemma.model.vision_tower)
    # The export deletes each part before it builds the next one.
    del embed
    next(parts)
    assert tower() is None
