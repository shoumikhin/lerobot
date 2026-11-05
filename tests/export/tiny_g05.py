# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot in 2026.

"""A tiny G0.5 checkpoint with a local word-level tokenizer, for export tests on the CPU without the Hub."""

import json
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

IMAGES = ("observation.images.image", "observation.images.wrist_image")
IMAGE_SIZE = 16
WORDS = ["Embodiment:", "libero", "Task:", "State:", "Action:", ";", "pick", "up", "the", "cube"]
SPECIAL = ["<|endoftext|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
# The rows of each head: 16 wide with half of it rotated, split 2, 1, 1 over the three rope sections.
QWEN = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 256,
    "rope_parameters": {
        "rope_theta": 10000.0,
        "rope_type": "default",
        "mrope_section": [2, 1, 1],
        "mrope_interleaved": True,
        "partial_rotary_factor": 0.5,
    },
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 2,
}


def save_tokenizer(path: Path) -> dict[str, int]:
    """A word-level tokenizer whose special tokens come last, as in Qwen3.5's, and their ids."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocabulary = {word: i for i, word in enumerate(["[PAD]", "[UNK]", *WORDS])}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
    fast.add_special_tokens({"additional_special_tokens": SPECIAL})
    fast.save_pretrained(path)
    # G0.5 sizes its vocabulary from the added tokens the tokenizer config lists, as Qwen3.5's does.
    config_file = path / "tokenizer_config.json"
    config = json.loads(config_file.read_text())
    config["added_tokens_decoder"] = {
        str(i): {"content": t.content, "special": t.special} for i, t in fast.added_tokens_decoder.items()
    }
    config_file.write_text(json.dumps(config))
    return {token: fast.convert_tokens_to_ids(token) for token in SPECIAL}


def author_model_config(ids: dict[str, int], horizon: int, steps: int) -> dict:
    """The serialized model of a G0.5 checkpoint, a few units wide: a linear and a full attention layer."""
    return {
        "vision": {
            "depth": 1,
            "hidden_size": 16,
            "num_heads": 2,
            "patch_size": 4,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "intermediate_size": 32,
            "out_hidden_size": QWEN["hidden_size"],
            "num_position_embeddings": 16,
            "temporal_freq": 0,
        },
        "vlm": {**QWEN, "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"]},
        "action_expert": {
            **QWEN,
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "full_attention"],
            "input_dim": 20,
            "output_dim": 20,
        },
        "fm": {
            "time_convention": "pi_convention",
            "zero_pad_action_target": False,
            "final_action_clip_value": None,
            "action_causal": False,
            "num_inference_steps": steps,
            "horizon_steps": horizon,
            "action_dim": 20,
        },
        "ar": {"max_new_tokens": 4},
        "AT_CONFIG": {
            "parts_meta": {"right_control": 6, "right_gripper": 1},
            "rule_based_key_patterns": ["gripper"],
            "model_arch": {"n_codebooks": 1, "codebook_size": 4},
        },
        "pad_token_id": 0,
        "eos_token_id": ids["<|endoftext|>"],
        "image_token_index": ids["<|image_pad|>"],
        "proprio_dim": 20,
        "position_ids_type": "pi0fast",
        "attn_implementation": "eager",
        "input_preprocessor": {"pred_eov": False},
    }


def save_checkpoint(path: Path, *, horizon: int = 4, n_action_steps: int = 2, steps: int = 3):
    """Save a tiny float32 G0.5 LIBERO checkpoint, its processors and tokenizer into `path`; return its config."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.g05.configuration_g05 import G05Config, make_g05_prompt_template
    from lerobot.policies.g05.modeling_g05 import G05Policy

    path.mkdir(parents=True)
    ids = save_tokenizer(path / "hf_processor")
    config = G05Config(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            **{
                camera: PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMAGE_SIZE, IMAGE_SIZE))
                for camera in IMAGES
            },
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        device="cpu",
        dtype=torch.float32,
        action_head="flow",
        discrete_action=False,
        continuous_action=True,
        chunk_size=horizon,
        n_action_steps=n_action_steps,
        use_stepwise_action_norm=True,
        libero_gripper_binarize=True,
        camera_order=IMAGES,
        camera_sizes=dict.fromkeys(IMAGES, (IMAGE_SIZE, IMAGE_SIZE)),
        num_input_images=len(IMAGES),
        num_prompt_images=len(IMAGES),
        prompt_template=make_g05_prompt_template(len(IMAGES), predict_cot=False, flow_only=True),
        author_model_config=author_model_config(ids, horizon, steps),
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        },
    )
    config.pretrained_path = path
    torch.manual_seed(0)
    policy = G05Policy(config, checkpoint_dir=path)
    generator = torch.Generator().manual_seed(1)
    # Each step of the chunk has its own action statistics.
    action_q01 = -1 - torch.rand(horizon, 7, generator=generator)
    stats = {
        OBS_STATE: {"q01": -torch.ones(7), "q99": torch.ones(7)},
        ACTION: {"q01": action_q01, "q99": action_q01 + 2 + torch.rand(horizon, 7, generator=generator)},
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    policy.save_pretrained(path)
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)
    return config
