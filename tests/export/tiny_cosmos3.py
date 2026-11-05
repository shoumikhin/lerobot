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

"""A tiny Cosmos3 Edge checkpoint with a local word-level tokenizer, for export tests on the CPU without the Hub."""

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION

IMAGES = ("observation.images.image", "observation.images.image2")
IMAGE_SIZE = 32
ACTION_NAMES = [*(f"pos_{i}" for i in range(3)), *(f"rot6d_{i}" for i in range(6)), "gripper_0"]
SPECIAL = ["<|im_start|>", "<|im_end|>", "<|vision_start|>"]
# One user turn, then the generation prompt, as the Cosmos3 tokenizer step expects of a chat template.
CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message.role }} "
    "{% for part in message.content %}{{ part.text }}{% endfor %}<|im_end|> {% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant {% endif %}"
)


def save_tokenizer(path: Path) -> None:
    """A word-level tokenizer that splits the JSON prompt on whitespace and punctuation, with the special tokens."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["user", "assistant", "pick", "up", "the", "cube", "actions", "description", "fps", "resolution"]
    vocabulary = {word: i for i, word in enumerate(["[PAD]", "[UNK]", *words])}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="<|im_end|>"
    )
    fast.add_special_tokens({"additional_special_tokens": SPECIAL})
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(path)


def save_checkpoint(path: Path, *, chunk_size: int = 4, steps: int = 3, **overrides) -> None:
    """Save a random Cosmos3 Edge, a few units wide, with its processors, to `path`."""
    from lerobot.policies.cosmos3.configuration_cosmos3 import Nemotron3DenseVLMoTConfig, Wan2pt2VAEConfig
    from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors

    path.mkdir(parents=True)
    save_tokenizer(path / "tokenizer")
    text = {
        "vocab_size": 32,
        "num_hidden_layers": 2,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
    }
    config = make_policy_config(
        "cosmos3_edge",
        mot_backbone=Nemotron3DenseVLMoTConfig(
            text_config=text,
            vision_config={
                "num_hidden_layers": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 2,
            },
            projector_config={"input_hidden_size": 32, "merger_intermediate_size": 64, "out_hidden_size": 64},
        ),
        vae=Wan2pt2VAEConfig(
            base_dim=16,
            decoder_base_dim=16,
            z_dim=8,
            num_res_blocks=1,
            latents_mean=[0.0] * 8,
            latents_std=[1.0] * 8,
        ),
        device="cpu",
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        num_inference_steps=steps,
        max_action_dim=16,
        fps=20.0,
        action_feature_names=ACTION_NAMES,
        tokenizer_name=str(path / "tokenizer"),
        **overrides,
    )
    config.input_features = {
        camera: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE)) for camera in IMAGES
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(len(ACTION_NAMES),))}
    torch.manual_seed(0)
    policy = get_policy_class("cosmos3_edge")(config)
    generator = torch.Generator().manual_seed(1)
    q01 = -1 - torch.rand(len(ACTION_NAMES), generator=generator)
    stats = {ACTION: {"q01": q01, "q99": q01 + 2 + torch.rand(len(ACTION_NAMES), generator=generator)}}
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    policy.save_pretrained(path)
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)
