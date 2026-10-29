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

"""A tiny SmolVLA checkpoint with a local tokenizer, for export tests on the CPU without the Hub."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

CAMERAS = (f"{OBS_IMAGES}.top", f"{OBS_IMAGES}.wrist")
VOCABULARY = ["[PAD]", "[UNK]", "[FAKE]", "[GLOBAL]", "pick", "up", "the", "cube", "open", "drawer"]


def use_tiny_smolvla(patch: Any) -> None:
    """Give SmolVLA a four-layer SmolVLM 32 wide with a one-layer vision tower, built without the Hub.

    `patch.setattr(target, name, value)` makes each change: a pytest `monkeypatch`, or a namespace holding the
    builtin `setattr` in a process that stays tiny.
    """
    from transformers import SmolVLMConfig

    from lerobot.policies.smolvla import smolvlm_with_expert

    config = SmolVLMConfig(
        text_config={
            "model_type": "llama",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 64,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 32,
            "patch_size": 8,
        },
        scale_factor=2,
    )
    # SmolVLA reads only the ids of two special tokens from the processor.
    tokens = SimpleNamespace(
        fake_image_token_id=VOCABULARY.index("[FAKE]"), global_image_token_id=VOCABULARY.index("[GLOBAL]")
    )
    patch.setattr(smolvlm_with_expert, "AutoConfig", SimpleNamespace(from_pretrained=lambda name: config))
    processor = SimpleNamespace(tokenizer=tokens)
    patch.setattr(
        smolvlm_with_expert, "AutoProcessor", SimpleNamespace(from_pretrained=lambda name: processor)
    )


def save_tokenizer(path: Path) -> None:
    """A word-level tokenizer of a few words."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(VOCABULARY)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(
        path
    )


def save_checkpoint(path: Path, patch: Any):
    """Save a tiny float32 SmolVLA with its processors and tokenizer into `path`, and return its config."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy

    use_tiny_smolvla(patch)
    save_tokenizer(path.parent / "tokenizer")
    config = SmolVLAConfig(
        device="cpu",
        vlm_model_name=str(path.parent / "tokenizer"),
        resize_imgs_with_padding=(32, 32),
        num_vlm_layers=4,
        chunk_size=4,
        n_action_steps=3,
        max_state_dim=8,
        max_action_dim=8,
        input_features={
            **{name: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 24, 32)) for name in CAMERAS},
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    torch.manual_seed(0)
    SmolVLAPolicy(config).save_pretrained(path)
    stats = {
        OBS_STATE: {"mean": torch.zeros(3), "std": torch.ones(3)},
        ACTION: {"mean": torch.tensor([1.0, -2.0, 3.0]), "std": torch.tensor([2.0, 0.5, 10.0])},
    }
    for processor in make_pre_post_processors(config, dataset_stats=stats):
        processor.save_pretrained(path)
    return config
