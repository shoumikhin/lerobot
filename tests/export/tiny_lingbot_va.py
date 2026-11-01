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

"""A tiny float32 LingBot-VA checkpoint, with a tiny VAE, text encoder and tokenizer, for CPU tests offline."""

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

CAMERAS = (f"{OBS_IMAGES}.image", f"{OBS_IMAGES}.image2")
TASK = "pick up the red cube"
WORDS = ["<pad>", "</s>", "<unk>", *(f"▁{word}" for word in ["pick", "up", "the", "red", "cube", "stop"])]


def save_frozen_modules(path: Path, text_dim: int) -> None:
    """Save the VAE, text encoder and tokenizer the policy loads from `wan_pretrained_path`, tiny."""
    from diffusers import AutoencoderKLWan
    from transformers import T5TokenizerFast, UMT5Config, UMT5EncoderModel

    torch.manual_seed(0)
    # Like the Wan2.2 VAE: 16 times smaller images, 4 frames per latent frame, 48 latent channels.
    AutoencoderKLWan(
        base_dim=8,
        decoder_base_dim=8,
        z_dim=48,
        dim_mult=[1, 2, 2, 2],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
        is_residual=True,
        patch_size=2,
        in_channels=12,
        out_channels=12,
        latents_mean=[0.1] * 48,
        latents_std=[2.0] * 48,
        scale_factor_spatial=16,
        scale_factor_temporal=4,
    ).save_pretrained(path / "vae")
    vocabulary = [(word, 0.0 if i < 3 else -1.0) for i, word in enumerate(WORDS)]
    T5TokenizerFast(vocab=vocabulary, extra_ids=0).save_pretrained(path / "tokenizer")
    config = UMT5Config(vocab_size=len(WORDS), d_model=text_dim, d_kv=8, d_ff=32, num_layers=1, num_heads=2)
    UMT5EncoderModel(config).save_pretrained(path / "text_encoder")


def save_checkpoint(path: Path):
    """Save a tiny LingBot-VA with its processors into `path`, its frozen modules beside it, and return its config.

    Its KV cache holds two chunks, so it frees its oldest entries from the third chunk on.
    """
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
    from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy

    save_frozen_modules(path.parent / "frozen", text_dim=32)
    config = LingBotVAConfig(
        num_attention_heads=2,
        attention_head_dim=24,
        action_dim=8,
        text_dim=32,
        freq_dim=16,
        ffn_dim=64,
        num_layers=2,
        frame_chunk_size=2,
        attn_window=4,
        num_inference_steps=3,
        action_num_inference_steps=4,
        height=64,
        width=64,
        max_sequence_length=8,
        used_action_channel_ids=[0, 1, 2, 3],
        obs_cam_keys=list(CAMERAS),
        image_hflip=True,
        wan_pretrained_path=str(path.parent / "frozen"),
        dtype=torch.float32,
        device="cpu",
        input_features={
            **{name: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 48)) for name in CAMERAS},
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))},
    )
    torch.manual_seed(0)
    policy = LingBotVAPolicy(config)
    # The action stream's time embedding starts as a copy of the video stream's; apart, a part that runs the wrong
    # one plays other actions.
    with torch.no_grad():
        for parameter in policy.transformer.condition_embedder_action.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    policy.save_pretrained(path)
    stats = {ACTION: {"q01": torch.tensor([-1.0, 0.0, 2.0, -5.0]), "q99": torch.tensor([1.0, 3.0, 2.5, 5.0])}}
    for processor in make_pre_post_processors(config, dataset_stats=stats):
        processor.save_pretrained(path)
    return config
