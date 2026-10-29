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

"""A tiny pi0.5 checkpoint with a local tokenizer, for export tests on the CPU without the Hub."""

from pathlib import Path
from typing import Any

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

IMAGE = "observation.images.base_0_rgb"
WORDS = ["Task", "State", "Action", ":", ",", ";", "-", ".", "pick", "up", "the", "cube", "open", "drawer"]


def use_tiny_pi05(patch: Any) -> None:
    """Give pi0.5 a four-layer Gemma 16 wide, so the chain splits its language model into two parts.

    `patch.setattr(target, name, value)` makes each change: a pytest `monkeypatch`, or a namespace holding the
    builtin `setattr` in a process that stays tiny.
    """
    from lerobot.policies.pi05 import modeling_pi05
    from tests.policies.pi0_pi05.utils.meta_load import use_tiny_backbone

    use_tiny_backbone(patch, modeling_pi05)
    tiny = modeling_pi05.GemmaConfig(width=16, depth=4, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8)
    patch.setattr(modeling_pi05, "get_gemma_config", lambda variant: tiny)
    paligemma = modeling_pi05.PaliGemmaForConditionalGenerationWithPiGemma

    def small_projector(config):
        # The policy fixes the image projector at Gemma 2B's width; this Gemma is narrower.
        config.vision_config.projection_dim = tiny.width
        config.vision_config.intermediate_size = 32
        return paligemma(config=config)

    patch.setattr(modeling_pi05, "PaliGemmaForConditionalGenerationWithPiGemma", small_projector)


def save_tokenizer(path: Path) -> None:
    """A word-level tokenizer that splits like Gemma where pi0.5's prompt needs it: it adds a start token,
    and splits the state's numbers the same inside the prompt as alone."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocabulary = {word: i for i, word in enumerate(["[PAD]", "[UNK]", "[BOS]", *WORDS])}
    vocabulary |= {str(n): len(vocabulary) + n for n in range(256)}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 2)])
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]"
    ).save_pretrained(path)


def save_checkpoint(path: Path, patch: Any, relative_actions: bool = False):
    """Save a tiny float32 pi0.5 with its processors and tokenizer into `path`, and return its config."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Config, PI05Policy
    from tests.policies.pi0_pi05.utils.meta_load import save_checkpoint as save_weights

    use_tiny_pi05(patch)
    save_tokenizer(path.parent / "tokenizer")
    config = PI05Config(
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
    renames = {f"time_mlp_in.{name}": f"action_time_mlp_in.{name}" for name in ("weight", "bias")}
    save_weights(PI05Policy, config, path, renames)
    config.save_pretrained(path)
    quantiles = {"mean": torch.zeros(3), "std": torch.ones(3)}
    stats = {
        OBS_STATE: {**quantiles, "q01": -torch.ones(3), "q99": torch.ones(3)},
        ACTION: {**quantiles, "q01": torch.full((3,), -2.0), "q99": torch.full((3,), 4.0)},
    }
    for processor in make_pre_post_processors(config, dataset_stats=stats):
        processor.save_pretrained(path)
    return config
