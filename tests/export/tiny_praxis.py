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

"""A tiny float32 Praxis-1 checkpoint, with the praxis tests' fake autoencoder and text encoder, for CPU tests."""

from pathlib import Path
from typing import Any

import torch

TASK = "pick up the red cube"


def use_tiny_praxis(patch: Any) -> None:
    """Give a loaded Praxis-1 policy the fake autoencoder and text encoder in place of the frozen ones.

    `patch.setattr(target, name, value)` makes the change: a pytest `monkeypatch`, or a namespace holding the
    builtin `setattr` in a process that stays tiny.
    """
    from lerobot.policies.praxis.modeling_praxis import PraxisPolicy
    from tests.policies.praxis.tiny_model import FakeAutoencoder, FakeTextEncoder

    def load_frozen_parts(policy: PraxisPolicy) -> None:
        policy.vae = FakeAutoencoder()
        policy.text_encoder = FakeTextEncoder()

    patch.setattr(PraxisPolicy, "_load_frozen_parts", load_frozen_parts)


def save_checkpoint(path: Path, bimanual: bool = False, **overrides: Any):
    """Save a tiny random Praxis-1 of the single-arm or the bimanual rig with its processors into `path`, and
    return its config."""
    from lerobot.policies.praxis.modeling_praxis import PraxisPolicy
    from lerobot.policies.praxis.processor_praxis import make_praxis_pre_post_processors
    from tests.policies.praxis.tiny_model import randomize_, tiny_bimanual_lerobot_config, tiny_lerobot_config

    make_config = tiny_bimanual_lerobot_config if bimanual else tiny_lerobot_config
    config = make_config(**{"dtype": torch.float32, **overrides})
    policy = PraxisPolicy(config)
    randomize_(policy.model)
    policy.save_pretrained(path)
    for processor in make_praxis_pre_post_processors(config):
        processor.save_pretrained(path)
    return config
