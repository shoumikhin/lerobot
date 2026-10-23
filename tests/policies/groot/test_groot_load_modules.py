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

"""Some submodules of a fine-tuned GR00T checkpoint load on their own, without the rest of its weights."""

import pytest
import torch
from safetensors.torch import load_file, save_file

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from lerobot.configs import PreTrainedConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from tests.policies.groot.test_groot_finetune_load import _save_finetune

EMBEDDING = "_groot_model.backbone.model.model.language_model.embed_tokens"
ACTION_HEAD = "_groot_model.action_head"


def test_listed_modules_load_in_the_compute_dtype_and_the_rest_are_dropped(tmp_path):
    base, finetune = _save_finetune(tmp_path)
    expected = GrootPolicy.from_pretrained(finetune)
    (base / "model.safetensors").unlink()
    config = PreTrainedConfig.from_pretrained(finetune)

    policy = GrootPolicy.load_checkpoint_modules(
        str(finetune / "model.safetensors"), config, [EMBEDDING, ACTION_HEAD]
    )

    assert config.use_bf16
    for name in (EMBEDDING, ACTION_HEAD):
        loaded = policy.get_submodule(name).state_dict()
        for key, tensor in expected.get_submodule(name).state_dict().items():
            assert torch.equal(loaded[key], tensor.to(torch.bfloat16)), f"{name}.{key}"
    backbone = policy._groot_model.backbone.model
    assert backbone.model.visual is None
    assert backbone.model.language_model.layers is None
    assert backbone.lm_head is None
    assert not policy.training


def test_a_module_missing_from_the_file_is_reported(tmp_path):
    _, finetune = _save_finetune(tmp_path)
    config = PreTrainedConfig.from_pretrained(finetune)
    model_file = finetune / "model.safetensors"
    tensors = load_file(model_file)
    save_file({k: v for k, v in tensors.items() if ".action_head.vlln." not in k}, model_file)

    with pytest.raises(KeyError, match="vlln"):
        GrootPolicy.load_checkpoint_modules(str(model_file), config, [ACTION_HEAD])
