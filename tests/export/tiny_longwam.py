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

"""A tiny float32 Long-WAM checkpoint, with a tiny video VAE, text encoder and tokenizer, for CPU tests."""

from pathlib import Path

import torch

TASK = "pick place"


def save_checkpoint(path: Path, **overrides) -> None:
    """Save a tiny Long-WAM with its processors into `path`, and its tokenizer beside them."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.longwam.configuration_longwam import LongWAMConfig
    from lerobot.policies.longwam.modeling_longwam import LongWAMPolicy

    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "pick": 2, "place": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]").save_pretrained(
        path / "tokenizer"
    )
    cameras = ("observation.images.front", "observation.images.wrist")
    config = LongWAMConfig(
        # Cameras larger than `camera_size`, so the program resizes and crops them as Long-WAM does.
        input_features={
            **{name: PolicyFeature(FeatureType.VISUAL, (3, 20, 28)) for name in cameras},
            "observation.state": PolicyFeature(FeatureType.STATE, (2,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (3,))},
        device="cpu",
        dtype=torch.float32,
        push_to_hub=False,
        camera_keys=cameras,
        camera_size=(16, 16),
        past_obs_steps=16,
        future_video_steps=16,
        action_horizon=4,
        inference_action_horizon=6,
        n_action_steps=2,
        action_dim=3,
        proprio_dim=2,
        num_inference_steps=2,
        imagine_steps=1,
        action_hidden_dim=8,
        action_ffn_dim=32,
        tokenizer_name=str(path / "tokenizer"),
        tokenizer_max_length=8,
        prompt_template="{task}",
        video_config={
            "in_channels": 4,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "text_dim": 16,
            "freq_dim": 16,
            "ffn_dim": 32,
            "num_layers": 1,
            "patch_size": [1, 2, 2],
            "rope_max_seq_len": 32,
        },
        vae_config={
            "base_dim": 4,
            "z_dim": 4,
            "dim_mult": [1, 2, 2],
            "num_res_blocks": 1,
            "temperal_downsample": [True, True],
            "scale_factor_temporal": 4,
            "scale_factor_spatial": 4,
            "latents_mean": [0.0] * 4,
            "latents_std": [1.0] * 4,
        },
        text_encoder_config={
            "vocab_size": 4,
            "d_model": 16,
            "d_kv": 8,
            "d_ff": 32,
            "num_layers": 1,
            "num_heads": 2,
            "dropout_rate": 0,
            "feed_forward_proj": "gated-gelu",
        },
        **overrides,
    )
    stats = {
        key: {"min": torch.full((size,), -2.0), "max": torch.full((size,), 2.0)}
        for key, size in (("observation.state", config.proprio_dim), ("action", config.action_dim))
    }
    torch.manual_seed(0)
    LongWAMPolicy(config).save_pretrained(path)
    for processor in make_pre_post_processors(config, dataset_stats=stats):
        processor.save_pretrained(path)
