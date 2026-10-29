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

"""A tiny random GR00T N1.7 checkpoint with its own tiny Qwen3-VL processor, for export tests on the CPU."""

from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

SPECIAL = [
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
]
WORDS = [
    "<|endoftext|>",
    "[UNK]",
    "user",
    "pick",
    "up",
    "the",
    "red",
    "cube",
    "then",
    "stop",
    "perform",
    "task",
]
CHAT_TEMPLATE = (
    "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n"
    "{% for content in message['content'] %}{% if content['type'] == 'image' %}"
    "<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'text' %}{{ content['text'] }}"
    "{% endif %}{% endfor %}<|im_end|>\n{% endfor -%}"
)
TASK = "Pick up the red cube"
# The geometry each image mode resizes with, and the cameras its checkpoint takes.
MODES = {
    "albumentations": (
        {"image_target_size": [16, 16], "shortest_image_edge": 16, "crop_fraction": 0.75},
        {"observation.images.front": (3, 32, 32), "observation.images.wrist": (3, 32, 32)},
    ),
    "torchvision": (
        {"image_target_size": [16, 16], "image_crop_size": [12, 12]},
        {"observation.images.front": (3, 24, 40), "observation.images.wrist": (3, 24, 40)},
    ),
}


def write_backbone(path: Path):
    """A tiny Qwen3-VL: its config, a word-level tokenizer with Qwen's special tokens, and its processors."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3VLConfig
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    vocab = {token: i for i, token in enumerate(SPECIAL + WORDS)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(SPECIAL)
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="<|endoftext|>")
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(path)
    pixels = {"patch_size": 4, "temporal_patch_size": 2, "merge_size": 2}
    Qwen2VLImageProcessor(**pixels, min_pixels=64, max_pixels=4096).save_pretrained(path)
    Qwen3VLVideoProcessor(**pixels).save_pretrained(path)
    Qwen3VLConfig(
        image_token_id=vocab["<|image_pad|>"],
        video_token_id=vocab["<|video_pad|>"],
        vision_start_token_id=vocab["<|vision_start|>"],
        vision_end_token_id=vocab["<|vision_end|>"],
        tie_word_embeddings=True,
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 7,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "vocab_size": 64,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True},
            "tie_word_embeddings": True,
        },
        vision_config={
            "depth": 3,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 4,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "deepstack_visual_indexes": [0, 1, 2],
            "num_position_embeddings": 16,
        },
    ).save_pretrained(path)


def save_checkpoint(
    path: Path,
    mode: str,
    dataset_meta=None,
    use_bf16: bool = False,
    select_layer: int = 7,
    alternate_vl_dit: bool = True,
    **overrides,
):
    """A fine-tuned GR00T reading `select_layer` of its 7 Qwen3 layers (two language parts above 6), and 9
    diffusion blocks (two step parts), which alternate between image and text with `alternate_vl_dit`."""
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.policies.groot.processor_groot import GrootN17VLMEncodeStep

    geometry, cameras = MODES[mode]
    backbone = path / "backbone"
    write_backbone(backbone)
    base = path / "base"
    torch.manual_seed(0)
    GR00TN17(
        GR00TN17Config(
            model_name=str(backbone),
            select_layer=select_layer,
            backbone_embedding_dim=32,
            hidden_size=32,
            input_embedding_dim=32,
            max_state_dim=8,
            max_action_dim=6,
            action_horizon=4,
            max_num_embodiments=12,
            vl_self_attention_cfg={"num_layers": 1, "num_attention_heads": 2, "attention_head_dim": 16},
            diffusion_model_cfg={
                "positional_embeddings": None,
                "num_layers": 9,
                "num_attention_heads": 2,
                "attention_head_dim": 16,
                "norm_type": "ada_norm",
                "dropout": 0.0,
                "final_dropout": False,
                "output_dim": 32,
                "interleave_self_attention": True,
            },
            load_bf16=True,
            use_alternate_vl_dit=alternate_vl_dit,
        ),
        load_backbone_weights=False,
    ).to(torch.bfloat16).save_pretrained(base)
    config = GrootConfig(
        base_model_path=str(base),
        input_features={
            **{name: PolicyFeature(type=FeatureType.VISUAL, shape=shape) for name, shape in cameras.items()},
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(5,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(5,))},
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        max_state_dim=8,
        max_action_dim=6,
        use_bf16=use_bf16,
        **overrides,
    )
    policy = GrootPolicy(config)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(torch.randn_like(parameter), alpha=0.02)
    policy.save_pretrained(path / "finetune")
    stats = {
        OBS_STATE: {"min": torch.full((5,), -1.0), "max": torch.full((5,), 2.0)},
        ACTION: {"min": torch.full((5,), -10.0), "max": torch.full((5,), 30.0)},
    }
    if config.use_relative_actions:
        # Relative actions are normalized with statistics per chunk step.
        stats[ACTION] = {name: value.expand(4, 5) for name, value in stats[ACTION].items()}
    preprocessor, postprocessor = make_pre_post_processors(
        config, dataset_stats=stats, dataset_meta=dataset_meta
    )
    encode = next(s for s in preprocessor.steps if isinstance(s, GrootN17VLMEncodeStep))
    encode.model_name = str(backbone)
    encode.use_albumentations = mode == "albumentations"
    encode.image_crop_size = encode.shortest_image_edge = encode.crop_fraction = None
    for key, value in geometry.items():
        setattr(encode, key, value)
    for processor in (preprocessor, postprocessor):
        processor.save_pretrained(path / "finetune")
    return path / "finetune"
