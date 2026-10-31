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

"""pi0, pi0.5 and pi0-FAST read their weights file with the download options from_pretrained was given."""

import contextlib

import pytest

pytest.importorskip("transformers")

import transformers.utils  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pi0 import PI0Config, PI0Policy, modeling_pi0  # noqa: E402
from lerobot.policies.pi05 import PI05Config, PI05Policy, modeling_pi05  # noqa: E402
from tests.policies.pi0_pi05.utils.meta_load import DOWNLOAD_ARGUMENTS, use_tiny_backbone  # noqa: E402


class Tokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


def pi0_fast(monkeypatch):
    pytest.importorskip("scipy")
    from lerobot.policies.pi0_fast import PI0FastConfig, PI0FastPolicy, modeling_pi0_fast

    monkeypatch.setattr(modeling_pi0_fast, "AutoProcessor", Tokenizer)
    monkeypatch.setattr(modeling_pi0_fast, "AutoTokenizer", Tokenizer)
    return modeling_pi0_fast, PI0FastPolicy, PI0FastConfig


class LookedUpError(Exception):
    pass


@pytest.mark.parametrize(
    "policy",
    [
        lambda monkeypatch: (modeling_pi0, PI0Policy, PI0Config),
        lambda monkeypatch: (modeling_pi05, PI05Policy, PI05Config),
        pi0_fast,
    ],
    ids=["pi0", "pi05", "pi0_fast"],
)
def test_the_weights_file_is_read_with_the_download_options(policy, tmp_path, monkeypatch):
    module, policy_cls, config_cls = policy(monkeypatch)
    use_tiny_backbone(monkeypatch, module)
    config = config_cls(image_resolution=(28, 28), device="cpu")
    config.input_features = {
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(8,))}
    calls = []

    def cached_file(path, filename, **kwargs):
        calls.append(kwargs)
        raise LookedUpError

    monkeypatch.setattr(transformers.utils, "cached_file", cached_file)
    # What the policy does after the lookup fails does not matter here, only what the lookup was given.
    with contextlib.suppress(LookedUpError):
        policy_cls.from_pretrained(tmp_path, config=config, **DOWNLOAD_ARGUMENTS)
    assert calls == [DOWNLOAD_ARGUMENTS]
