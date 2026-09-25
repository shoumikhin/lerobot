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

"""PI0Policy.from_pretrained builds parameters on the meta device and matches the regular load."""

import logging
import sys

import pytest
import torch

pytest.importorskip("transformers")

import transformers.utils  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.common.openpi_checkpoint import _parameters_on_meta  # noqa: E402
from lerobot.policies.pi0 import PI0Config, PI0Policy, modeling_pi0  # noqa: E402
from lerobot.utils import import_utils  # noqa: E402
from tests.policies.common.meta_load import (  # noqa: E402
    DOWNLOAD_ARGUMENTS,
    EMBED_TOKENS,
    LM_HEAD,
    LOADER_LOGGER,
    assert_full_size_qualifies,
    assert_lm_head_not_copied,
    assert_same,
    assert_same_actions,
    drop_projector_bias,
    embedding_stored_first,
    load,
    save_checkpoint,
    store_embedding,
    use_tiny_backbone,
)
from tests.utils import require_cuda  # noqa: E402

RENAMES = {f"action_time_mlp_in.{name}": f"time_mlp_in.{name}" for name in ("weight", "bias")}


@pytest.fixture
def config(monkeypatch):
    use_tiny_backbone(monkeypatch, modeling_pi0)
    config = PI0Config(image_resolution=(28, 28), device="cpu", dtype=torch.bfloat16)
    config.input_features = {
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    return config


def save(config, path, edit=None):
    return save_checkpoint(PI0Policy, config, path, RENAMES, edit)


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_matches_regular_load(config, tmp_path, monkeypatch, gradient_checkpointing):
    config.gradient_checkpointing = gradient_checkpointing
    path = save(config, tmp_path / "ckpt")
    model, used_meta = load(PI0Policy, path, config, monkeypatch)
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert model.model.gradient_checkpointing_enabled is expected.model.gradient_checkpointing_enabled
    assert model.model.gradient_checkpointing_enabled is gradient_checkpointing
    assert_same(model, expected)
    assert_same_actions(model, expected)
    assert_lm_head_not_copied(model)
    lm_head = model.model.paligemma_with_expert.paligemma.lm_head.weight
    embed_tokens = model.get_parameter(EMBED_TOKENS)
    assert torch.equal(lm_head, embed_tokens) and lm_head.data_ptr() != embed_tokens.data_ptr()


@pytest.mark.parametrize("first", [True, False])
def test_stored_embedding_next_to_lm_head_matches_regular_load(config, tmp_path, monkeypatch, first):
    path = save(config, tmp_path / "ckpt", store_embedding(first))
    assert embedding_stored_first(path) == first
    model, used_meta = load(PI0Policy, path, config, monkeypatch)
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)


@require_cuda
def test_matches_regular_load_on_cuda(config, tmp_path, monkeypatch):
    path = save(config, tmp_path / "ckpt")
    config.device = "cuda"
    model, used_meta = load(PI0Policy, path, config, monkeypatch)
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)
    assert_same_actions(model, expected)


def test_frozen_parameters_keep_requires_grad(config, tmp_path, monkeypatch):
    config.freeze_vision_encoder = True
    path = save(config, tmp_path / "ckpt")
    model, used_meta = load(PI0Policy, path, config, monkeypatch)
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)


def test_subclass_with_its_own_constructor(config, tmp_path, monkeypatch):
    class Subclass(PI0Policy):
        _supports_meta_load = True

        def __init__(self, config, extra=None):
            super().__init__(config)
            self.extra = extra

    path = save(config, tmp_path / "ckpt")
    model, used_meta = load(Subclass, path, config, monkeypatch, extra=1)
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True)
    assert used_meta and type(model) is Subclass and model.extra == 1
    assert_same(model, expected)


def test_subclass_loads_the_regular_way(config, tmp_path, monkeypatch, caplog):
    # A subclass may, for example, read its parameters in its constructor, which only the regular path allows.
    class Subclass(PI0Policy):
        pass

    path = save(config, tmp_path / "ckpt")
    with caplog.at_level(logging.INFO, logger=LOADER_LOGGER):
        model, used_meta = load(Subclass, path, config, monkeypatch)
    assert "regular way: Subclass does not set _supports_meta_load in its own body" in caplog.text
    expected, _ = load(Subclass, path, config, monkeypatch, regular=True)
    assert not used_meta
    assert_same(model, expected)


def test_key_fixes_that_change_values_match_regular_load(config, tmp_path, monkeypatch):
    class Subclass(PI0Policy):
        _supports_meta_load = True

        def _fix_pytorch_state_dict_keys(self, state_dict, model_config):
            fixed = super()._fix_pytorch_state_dict_keys(state_dict, model_config)
            return {key: value * 2 for key, value in fixed.items()}

    path = save(config, tmp_path / "ckpt")
    model, used_meta = load(Subclass, path, config, monkeypatch)
    expected, _ = load(Subclass, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)


def test_copies_of_one_weight_keep_the_regular_winner(config, tmp_path, monkeypatch):
    # The regular load fixes every key, then adds "model.", so the unprefixed embedding wins although the file
    # holds the prefixed one, in bfloat16, last.
    def store_copies(state_dict):
        lm_head = state_dict["model." + LM_HEAD] = state_dict.pop(LM_HEAD)
        state_dict[EMBED_TOKENS.removeprefix("model.")] = torch.full_like(lm_head, 2)
        state_dict[EMBED_TOKENS] = torch.full_like(lm_head, 3, dtype=torch.bfloat16)

    path = save(config, tmp_path / "ckpt", store_copies)
    model, used_meta = load(PI0Policy, path, config, monkeypatch)
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)


def tie_lm_head(policy):
    paligemma = policy.model.paligemma_with_expert.paligemma
    paligemma.lm_head.weight = paligemma.model.language_model.embed_tokens.weight


def derive_buffer(policy):
    policy.model.register_buffer("doubled", policy.model.state_proj.bias.detach() * 2, persistent=False)


@pytest.mark.parametrize("change", [tie_lm_head, derive_buffer])
def test_model_the_loader_cannot_fill_loads_the_regular_way(config, tmp_path, monkeypatch, caplog, change):
    class Subclass(PI0Policy):
        _supports_meta_load = True

        def __init__(self, config):
            super().__init__(config)
            change(self)

    path = save(config, tmp_path / "ckpt")
    with caplog.at_level(logging.INFO, logger=LOADER_LOGGER):
        model, used_meta = load(Subclass, path, config, monkeypatch)
    assert "regular way: the policy shares a parameter or has a buffer on the meta device" in caplog.text
    expected, _ = load(Subclass, path, config, monkeypatch, regular=True)
    assert not used_meta
    assert_same(model, expected)


def test_full_size_policy_qualifies():
    assert_full_size_qualifies(PI0Policy, PI0Config)


def test_meta_build_draws_no_random_numbers(config):
    # Falling back builds the model again, so it must see the same random state as without the fast path.
    state = torch.random.get_rng_state()
    with _parameters_on_meta():
        PI0Policy(config)
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize(
    "edit, found",
    [
        (lambda sd: sd.pop("state_proj.bias"), "nothing"),
        (lambda sd: sd.update({"state_proj.bias": torch.ones(3)}), "shape (3,)"),
    ],
)
@pytest.mark.parametrize("strict", [True, False])
def test_other_checkpoints_load_the_regular_way(config, tmp_path, monkeypatch, caplog, edit, found, strict):
    path = save(config, tmp_path / "ckpt", edit)
    with caplog.at_level(logging.INFO, logger=LOADER_LOGGER):
        model, used_meta = load(PI0Policy, path, config, monkeypatch, strict=strict)
    assert f"the checkpoint has {found} for model.state_proj.bias, which needs shape" in caplog.text
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True, strict=strict)
    assert not used_meta
    assert_same(model, expected)


@pytest.mark.parametrize("strict", [True, False])
def test_weights_the_policy_does_not_have_are_skipped(config, tmp_path, monkeypatch, strict):
    # The regular path ignores them too: load_state_dict copies all the other weights before it reports them.
    path = save(config, tmp_path / "ckpt", lambda sd: sd.update(extra=torch.ones(1)))
    model, used_meta = load(PI0Policy, path, config, monkeypatch, strict=strict)
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True, strict=strict)
    assert used_meta
    assert_same(model, expected)


def test_bigger_action_space_loads_the_regular_way(config, tmp_path, monkeypatch, caplog):
    path = save(config, tmp_path / "ckpt")
    config.max_action_dim = 48
    with caplog.at_level(logging.INFO, logger=LOADER_LOGGER):
        model, used_meta = load(PI0Policy, path, config, monkeypatch)
    assert "for model.action_in_proj.weight, which needs shape (16, 48)" in caplog.text
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True)
    assert not used_meta
    assert_same(model, expected)


def test_unreadable_checkpoint_loads_the_regular_way(config, tmp_path, monkeypatch, caplog):
    path = tmp_path / "ckpt"
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"not a safetensors file")
    with caplog.at_level(logging.INFO, logger=LOADER_LOGGER):
        model, used_meta = load(PI0Policy, path, config, monkeypatch)
    assert "regular way: cannot read" in caplog.text
    expected, _ = load(PI0Policy, path, config, monkeypatch, regular=True)
    assert not used_meta
    assert_same(model, expected)


@pytest.mark.parametrize("complete", [True, False])
def test_download_arguments_reach_the_weights(config, tmp_path, monkeypatch, complete):
    path = save(config, tmp_path / "ckpt", None if complete else drop_projector_bias)
    cached_file = transformers.utils.cached_file
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return cached_file(*args)

    monkeypatch.setattr(transformers.utils, "cached_file", record)
    _, used_meta = load(PI0Policy, path, config, monkeypatch, **DOWNLOAD_ARGUMENTS)
    # One lookup, with every option, serves whichever path loads the weights.
    assert calls == [DOWNLOAD_ARGUMENTS] and used_meta == complete


def test_missing_weights_raise(config, tmp_path, monkeypatch):
    with pytest.raises(OSError, match="model.safetensors"):
        load(PI0Policy, tmp_path, config, monkeypatch)


def test_loading_inside_torch_device_meta_raises(config, tmp_path):
    # Nothing would place the weights, so it must fail instead of returning a policy without them.
    path = save(config, tmp_path / "ckpt")
    with pytest.raises(NotImplementedError, match="meta tensor"), torch.device("meta"):
        PI0Policy.from_pretrained(path, config=config)


def test_missing_transformers_names_the_extra(config, tmp_path, monkeypatch):
    monkeypatch.setitem(import_utils._require_package_cache, "transformers", False)
    monkeypatch.setitem(sys.modules, "transformers.utils", None)
    with pytest.raises(ImportError, match=r"lerobot\[pi\]"):
        PI0Policy.from_pretrained(tmp_path, config=config)
