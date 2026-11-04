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

"""A tiny pi0 checkpoint with a local tokenizer, for export tests on the CPU without the Hub."""

from pathlib import Path
from typing import Any

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.export.tiny_pi05 import IMAGE, save_tokenizer, use_tiny_gemma


def use_tiny_pi0(patch: Any) -> None:
    """Give pi0 the four-layer Gemma of `use_tiny_pi05`, so the chain splits its language model into two parts."""
    from lerobot.policies.pi0 import modeling_pi0

    use_tiny_gemma(patch, modeling_pi0)


def save_checkpoint(path: Path, patch: Any, relative_actions: bool = False):
    """Save a tiny float32 pi0 with its processors and tokenizer into `path`, and return its config."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi0 import PI0Config, PI0Policy
    from tests.policies.common.meta_load import save_checkpoint as save_weights

    use_tiny_pi0(patch)
    save_tokenizer(path.parent / "tokenizer")
    config = PI0Config(
        device="cpu",
        dtype=torch.float32,
        image_resolution=(28, 28),
        chunk_size=4,
        n_action_steps=3,
        max_state_dim=8,
        max_action_dim=8,
        tokenizer_max_length=48,
        text_tokenizer_name=str(path.parent / "tokenizer"),
        use_relative_actions=relative_actions,
        input_features={
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    save_weights(PI0Policy, config, path)
    config.save_pretrained(path)
    stats = {
        OBS_STATE: {"mean": torch.zeros(3), "std": torch.ones(3)},
        ACTION: {"mean": torch.tensor([0.5, -1.0, 2.0]), "std": torch.tensor([1.0, 2.0, 3.0])},
    }
    for processor in make_pre_post_processors(config, dataset_stats=stats):
        processor.save_pretrained(path)
    return config
