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

"""LingBot-VA's export recipe: what it refuses to export, and the inputs its parts are checked on."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from lerobot.policies.lingbot_va import export_lingbot_va  # noqa: E402
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig  # noqa: E402
from lerobot.policies.lingbot_va.export_lingbot_va import VALID, LingBotVAExportRecipe  # noqa: E402
from tests.export import tiny_lingbot_va  # noqa: E402


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"camera_layout": "robotwin_tshape"}, "two streaming VAEs"),
        ({"frame_chunk_size": 1}, "frame_chunk_size is 1"),
        ({"action_per_frame": 6}, "multiple of 4"),
        ({"video_exec_step": 0}, "video_exec_step=0"),
    ],
)
def test_export_refuses_what_it_does_not_run(change, message):
    config = LingBotVAConfig(**change)
    with pytest.raises(ValueError, match=message):
        LingBotVAExportRecipe(SimpleNamespace(config=config), None, None, None)


def make_recipe(path, **changes) -> LingBotVAExportRecipe:
    """The recipe for the tiny checkpoint saved into `path`, with its config changed as given."""
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy

    tiny_lingbot_va.save_checkpoint(path)
    config = PreTrainedConfig.from_pretrained(path)
    for name, value in changes.items():
        setattr(config, name, value)
    policy = LingBotVAPolicy.from_pretrained(path, config=config).eval()
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(path))
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task=tiny_lingbot_va.TASK, tolerance=None
    )
    return LingBotVAExportRecipe(policy, preprocessor, postprocessor, cfg)


def test_parts_play_like_the_policy_with_a_single_video_pass(tmp_path):
    # In float32 the parts are held to 2e-05 of the policy's actions; with one pass no run steps the video.
    parts = [part.name for part in make_recipe(tmp_path / "checkpoint", video_exec_step=1).parts()]
    assert "video_out" not in parts


def test_export_refuses_parts_that_play_other_actions_than_the_policy(tmp_path, monkeypatch):
    recipe = make_recipe(tmp_path / "checkpoint")
    forward = export_lingbot_va._Actions.forward

    def drifted(self, executed):
        actions, executed = forward(self, executed)
        return actions + 1e-3, executed

    monkeypatch.setattr(export_lingbot_va._Actions, "forward", drifted)
    with pytest.raises(ValueError, match="play other actions than the policy itself"):
        list(recipe.parts())


def test_the_bundle_config_leaves_out_a_local_frozen_modules_folder(tmp_path):
    recipe = make_recipe(tmp_path / "checkpoint")
    assert recipe.config.wan_pretrained_path == ""
    assert recipe.policy.config.wan_pretrained_path == str(tmp_path / "frozen")


def test_blocks_hold_the_text_the_policy_attends_to(tmp_path):
    recipe = make_recipe(tmp_path / "checkpoint", dtype=torch.bfloat16)
    blocks = next(part for part in recipe.parts() if part.name == "blocks")
    policy = recipe.policy
    with torch.no_grad():
        prompt, negative = policy._encode_prompt(recipe.task)
        text = torch.cat([prompt.to(torch.bfloat16), negative.to(torch.bfloat16)])
        expected = policy.transformer.condition_embedder.text_embedder(text)
    assert torch.equal(blocks.module.text, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_each_part_is_checked_on_a_kv_cache_past_its_first_eviction(tmp_path, dtype):
    """The tiny KV cache holds two chunks, so the test case's third chunk frees some of it. In half precision every
    part is checked on its own, on the inputs of its last call: the cache then holds keys, and it was full when the
    third chunk took its slots."""
    parts = {part.name: part for part in make_recipe(tmp_path / "checkpoint", dtype=dtype).parts()}

    assert all(part.check is not None for part in parts.values())
    blocks = parts["blocks"]
    assert blocks.example_inputs[blocks.inputs.index(VALID)].any()
    for name in ("alloc_video", "alloc_action"):
        allocate = parts[name]
        free = (~allocate.example_inputs[allocate.inputs.index(VALID)]).sum()
        assert free < allocate.module.count, name
