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

"""Praxis-1's export recipe: what it refuses to export, its bundle, and the check of its parts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.bundle.manifest import NOISE, Bundle
from lerobot.export import recipe as export_recipe
from lerobot.policies.praxis import export_praxis
from lerobot.policies.praxis.export_praxis import PraxisExportRecipe
from lerobot.policies.praxis.praxis_core import layers as praxis_layers
from lerobot.policies.praxis.praxis_core.vae import Autoencoder
from lerobot.scripts.lerobot_export import export
from lerobot.utils.constants import ACTION
from tests.export import (
    tiny_praxis,
    torch_backend,  # noqa: F401  # registers torch_test for the in-process trace step
)
from tests.export.plugins import use_plugins
from tests.policies.praxis.tiny_model import tiny_lerobot_config


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (tiny_lerobot_config(max_context_chunks=2), "max_context_chunks=1"),
        (tiny_lerobot_config(n_action_steps=12), "n_action_steps=24"),
    ],
)
def test_export_refuses_what_it_does_not_run(monkeypatch, config, message):
    with pytest.raises(ValueError, match=message):
        PraxisExportRecipe(SimpleNamespace(config=config), None, None, None)

    def load_policy(cfg):
        raise AssertionError("loaded the policy")

    monkeypatch.setattr(export_recipe, "load_policy", load_policy)
    with pytest.raises(ValueError, match=message):
        PraxisExportRecipe.load(SimpleNamespace(policy=config))


@pytest.mark.parametrize("bimanual", [False, True], ids=["single_arm", "bimanual"])
def test_bundle_plays_the_policys_first_chunks(tmp_path, monkeypatch, bimanual):
    """The export replays the bundle's test case, three chunks from a reset."""
    tiny_praxis.save_checkpoint(tmp_path / "checkpoint", bimanual=bimanual)
    use_plugins(monkeypatch, "torch_backend", "tiny_praxis")
    export(
        [
            "--backend.type=torch_test",
            f"--policy.path={tmp_path / 'checkpoint'}",
            f"--output_dir={tmp_path / 'bundle'}",
            f"--task={tiny_praxis.TASK}",
        ]
    )
    manifest = Bundle.read(tmp_path / "bundle").manifest
    assert sorted(manifest.programs) == [
        "actions",
        "actions_first",
        "decoder",
        "encode",
        "start",
        "step",
        "velocity",
    ]
    assert [run.program for run in manifest.first_runs[0]][-1] == "actions_first"
    assert [run.program for run in manifest.runs].count("velocity") == 2
    assert set(manifest.history.values()) == {22}
    assert manifest.task == tiny_praxis.TASK


def make_recipe(path, monkeypatch, **overrides) -> PraxisExportRecipe:
    """The recipe for the tiny checkpoint saved into `path`, with `overrides` of its config."""
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.praxis.modeling_praxis import PraxisPolicy

    tiny_praxis.save_checkpoint(path, **overrides)
    tiny_praxis.use_tiny_praxis(monkeypatch)
    config = PreTrainedConfig.from_pretrained(path)
    policy = PraxisPolicy.from_pretrained(path, config=config).eval()
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(path))
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task=tiny_praxis.TASK, tolerance=None
    )
    return PraxisExportRecipe(policy, preprocessor, postprocessor, cfg)


def test_the_test_case_holds_three_chunks_of_actions_and_noise(tmp_path, monkeypatch):
    """The noise's values are checked where the bundle plays the policy's actions with it."""
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    case = recipe.test_case()
    # 21 actions from the first chunk, 24 from the second, and the first of the third.
    assert case.actions.shape == (46, 7)
    assert case.noise is not None and len(case.noise) == export_praxis.CHUNKS
    assert np.isfinite(case.actions).all()


@pytest.mark.parametrize(
    "settings",
    [
        {"cfg_scale": 2.0},
        {"cfg_scale": 2.0, "cfg_video_scale": 1.0},
        {"cfg_video_scale": 1.0},
        {"num_steps": 1},
        {"num_steps": 3},
    ],
    ids=["actions_guided", "only_actions_guided", "no_guidance", "one_step", "three_steps"],
)
def test_parts_play_the_policys_actions_with_other_sampler_settings(tmp_path, monkeypatch, settings):
    """The parts must play the policy's actions over the test case, which `parts` checks before it yields them."""
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch, **settings)
    names = [part.name for part in recipe.parts()]
    assert ("decoder" in names) == (settings.get("num_steps", 2) > 1)


def test_the_rehearsal_draws_other_noise_than_the_test_case(tmp_path, monkeypatch):
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    drawn = []
    noise = recipe._noise
    monkeypatch.setattr(recipe, "_noise", lambda seed: drawn.append(noise(seed)) or drawn[-1])
    list(recipe.parts())
    case = {chunk.tobytes() for chunk in recipe.test_case().noise}
    assert not any(chunk.tobytes() in case for chunk in drawn[0])


@pytest.mark.parametrize("enabled", [True, False])
def test_the_export_leaves_the_custom_kernels_as_it_found_them(tmp_path, monkeypatch, enabled):
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    monkeypatch.setattr(praxis_layers, "_CUSTOM_KERNELS", enabled)
    list(recipe.parts())
    assert praxis_layers.custom_kernels_enabled() == enabled


def test_encode_holds_only_the_autoencoders_encoder_half(tmp_path, monkeypatch):
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    with torch.device("meta"):
        vae = Autoencoder()
    # As the policy keeps its frozen modules: out of its submodules, so not with `policy.vae = vae`.
    recipe.policy._frozen.vae = vae
    held = {id(parameter) for parameter in export_praxis._Encode(recipe.policy, 21).parameters()}
    assert held == {id(parameter) for module in (vae.encoder, vae.conv1) for parameter in module.parameters()}


def test_encode_tiles_the_frames_as_the_policy_does(tmp_path, monkeypatch):
    """A config without the video's tiling keys tiles at the default size, as the policy's sessions do."""
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    policy = recipe.policy
    policy.config.model_config["video"] = {}
    encode = export_praxis._Encode(policy, 21)
    assert (encode.tile_height, encode.tile_width) == tuple(policy.variant.tile_size)
    assert (encode.aspect_tol, encode.antialias) == (policy.variant.aspect_tol, policy.variant.antialias)


def test_the_bundle_config_leaves_out_a_local_frozen_modules_folder(tmp_path, monkeypatch):
    # The tiny checkpoint has no text encoder to name: a folder there would load a tokenizer step.
    frozen = str(tmp_path / "frozen")
    (tmp_path / "frozen").mkdir()
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch, vae_id=frozen)
    assert recipe.config.vae_id == ""
    assert recipe.policy.config.vae_id == frozen


def test_export_refuses_parts_that_play_other_actions_than_the_policy(tmp_path, monkeypatch):
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    forward = export_praxis._Actions.forward

    def drifted(self, *inputs):
        return forward(self, *inputs) + 1e-3

    monkeypatch.setattr(export_praxis._Actions, "forward", drifted)
    with pytest.raises(ValueError, match="play other actions than the policy itself"):
        list(recipe.parts())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_velocity_and_decoder_are_checked_on_their_own_in_bfloat16(tmp_path, monkeypatch, dtype):
    """In bfloat16 the test case's chunks amplify rounding so much that it cannot tell a correct program from a
    broken one, so `velocity` and `decoder` are checked on their own. No rerun measures the other parts' rounding."""
    parts = list(make_recipe(tmp_path / "checkpoint", monkeypatch, dtype=dtype).parts())
    checked = {part.name for part in parts if part.check is not None}
    assert checked == ({"velocity", "decoder"} if dtype == torch.bfloat16 else set())
    for part in parts:
        if part.check is not None:
            assert part.check.failures(part.name, part.check.outputs) == []
            assert part.check.failures(part.name, [torch.zeros_like(x) for x in part.check.outputs])


def test_every_chunk_reads_the_noise_and_the_kept_frames(tmp_path, monkeypatch):
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    first, later = recipe.plan().first_runs[0], recipe.plan().runs
    for runs in (first, later):
        assert runs[1].inputs == [NOISE]
        assert runs[-1].outputs == [ACTION]
        assert runs[0].inputs == [*recipe._cameras, recipe._state]
