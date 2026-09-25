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

"""PI05Policy.from_pretrained builds parameters on the meta device and matches the regular load."""

import logging
import sys

import pytest
import torch

pytest.importorskip("transformers")

import transformers.utils  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi05 import PI05Config, PI05Policy, modeling_pi05  # noqa: E402
from lerobot.utils import import_utils  # noqa: E402
from tests.policies.common.meta_load import (  # noqa: E402
    DOWNLOAD_ARGUMENTS,
    LOADER_LOGGER,
    add_unknown_weights,
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

RENAMES = {f"time_mlp_in.{name}": f"action_time_mlp_in.{name}" for name in ("weight", "bias")}


@pytest.fixture
def config(monkeypatch):
    use_tiny_backbone(monkeypatch, modeling_pi05)
    config = PI05Config(image_resolution=(28, 28), device="cpu", dtype=torch.bfloat16)
    config.input_features = {
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    return config


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_matches_regular_load(config, tmp_path, monkeypatch, gradient_checkpointing):
    config.gradient_checkpointing = gradient_checkpointing
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES)
    model, used_meta = load(PI05Policy, path, config, monkeypatch)
    expected, _ = load(PI05Policy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert model.model.gradient_checkpointing_enabled is expected.model.gradient_checkpointing_enabled
    assert model.model.gradient_checkpointing_enabled is gradient_checkpointing
    assert_same(model, expected)
    assert_same_actions(model, expected)
    assert_lm_head_not_copied(model)


def test_full_size_policy_qualifies():
    assert_full_size_qualifies(PI05Policy, PI05Config)


@require_cuda
def test_matches_regular_load_on_cuda(config, tmp_path, monkeypatch):
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES)
    config.device = "cuda"
    model, used_meta = load(PI05Policy, path, config, monkeypatch)
    expected, _ = load(PI05Policy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)
    assert_same_actions(model, expected)


def test_skips_weights_the_policy_does_not_have(config, tmp_path, monkeypatch, caplog):
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES, add_unknown_weights)
    with caplog.at_level(logging.WARNING):
        model, used_meta = load(PI05Policy, path, config, monkeypatch)
    unexpected = ["model.state_mlp_in.bias", "model.state_mlp_in.weight"]
    assert f"Unexpected key(s) when loading model: {unexpected}" in caplog.text
    expected, _ = load(PI05Policy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)


@pytest.mark.parametrize("first", [True, False])
def test_stored_embedding_next_to_lm_head_matches_regular_load(config, tmp_path, monkeypatch, first):
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES, store_embedding(first))
    assert embedding_stored_first(path) == first
    model, used_meta = load(PI05Policy, path, config, monkeypatch)
    expected, _ = load(PI05Policy, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)


class DoubledConversion(PI05Policy):
    _supports_meta_load = True

    def _convert_openpi_state_dict(self, state_dict):
        return {key: value * 2 for key, value in super()._convert_openpi_state_dict(state_dict).items()}


class DoubledPreparation(PI05Policy):
    _supports_meta_load = True

    def _prepare_pretrained_state_dict(self, state_dict):
        return {key: value * 2 for key, value in super()._prepare_pretrained_state_dict(state_dict).items()}


@pytest.mark.parametrize("policy_cls", [DoubledConversion, DoubledPreparation])
def test_conversions_that_change_values_match_regular_load(config, tmp_path, monkeypatch, policy_cls):
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES)
    model, used_meta = load(policy_cls, path, config, monkeypatch)
    expected, _ = load(policy_cls, path, config, monkeypatch, regular=True)
    assert used_meta
    assert_same(model, expected)


@pytest.mark.parametrize("saved_with_memory", [True, False])
def test_proprioceptive_memory(config, tmp_path, monkeypatch, caplog, saved_with_memory):
    # A checkpoint saved without the memory projection keeps the regular path, which initializes it fresh.
    # That is the expected way to start from a base checkpoint, so it is not reported as a fallback.
    config.use_proprioceptive_memory = saved_with_memory
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES)
    config.use_proprioceptive_memory = True
    with caplog.at_level(logging.INFO, logger=LOADER_LOGGER):
        model, used_meta = load(PI05Policy, path, config, monkeypatch)
    assert "regular way" not in caplog.text
    expected, _ = load(PI05Policy, path, config, monkeypatch, regular=True)
    assert used_meta == saved_with_memory
    assert_same(model, expected)


@pytest.mark.parametrize("complete", [True, False])
def test_download_arguments_reach_the_weights(config, tmp_path, monkeypatch, complete):
    path = save_checkpoint(
        PI05Policy, config, tmp_path / "ckpt", RENAMES, None if complete else drop_projector_bias
    )
    cached_file = transformers.utils.cached_file
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return cached_file(*args)

    monkeypatch.setattr(transformers.utils, "cached_file", record)
    _, used_meta = load(PI05Policy, path, config, monkeypatch, **DOWNLOAD_ARGUMENTS)
    # One lookup, with every option, serves whichever path loads the weights.
    assert calls == [DOWNLOAD_ARGUMENTS] and used_meta == complete


def test_missing_weights_raise(config, tmp_path, monkeypatch):
    with pytest.raises(OSError, match="model.safetensors"):
        load(PI05Policy, tmp_path, config, monkeypatch)


def test_loading_inside_torch_device_meta_raises(config, tmp_path):
    # Nothing would place the weights, so it must fail instead of returning a policy without them.
    path = save_checkpoint(PI05Policy, config, tmp_path / "ckpt", RENAMES)
    with pytest.raises(NotImplementedError, match="meta tensor"), torch.device("meta"):
        PI05Policy.from_pretrained(path, config=config)


def test_missing_transformers_names_the_extra(config, tmp_path, monkeypatch):
    monkeypatch.setitem(import_utils._require_package_cache, "transformers", False)
    monkeypatch.setitem(sys.modules, "transformers.utils", None)
    with pytest.raises(ImportError, match=r"lerobot\[pi\]"):
        PI05Policy.from_pretrained(tmp_path, config=config)
