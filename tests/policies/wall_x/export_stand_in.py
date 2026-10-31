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

"""A stand-in WALL-X for its export recipe, on WALL-X's preprocessor with a tiny Qwen2.5-VL processor."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("torchvision")
# The pixels are sized as qwen-vl-utils sizes them.
pytest.importorskip("qwen_vl_utils")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor  # noqa: E402
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig  # noqa: E402
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VisionTransformerPretrainedModel,
)

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.policies.wall_x.configuration_wall_x import WallXConfig  # noqa: E402
from lerobot.policies.wall_x.export_wall_x import WallXExportRecipe  # noqa: E402
from lerobot.policies.wall_x.modeling_wall_x import WallXPolicy  # noqa: E402
from lerobot.policies.wall_x.processor_wall_x import (  # noqa: E402
    WallXTokenizerStep,
    make_wall_x_pre_post_processors,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

CAMERA = "observation.images.front"
CHUNK = 2
SPECIAL = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
SPECIAL += ["<|propri|>", "<|action|>"]


def processor() -> SimpleNamespace:
    """A Qwen2.5-VL processor with WALL-X's special tokens, its other words unknown."""
    vocabulary = ["<pad>", "<unk>", *SPECIAL]
    words = Tokenizer(models.WordLevel({word: i for i, word in enumerate(vocabulary)}, unk_token="<unk>"))
    words.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=words, pad_token="<pad>", unk_token="<unk>", additional_special_tokens=SPECIAL
    )
    image_processor = Qwen2VLImageProcessor(patch_size=14, merge_size=2, temporal_patch_size=2)
    return SimpleNamespace(tokenizer=tokenizer, image_processor=image_processor)


class PixelWallX(WallXPolicy):
    """A stand-in WALL-X whose actions read the pixel values, the prompt, the state and the noise.

    It has a small vision tower where WALL-X has its own, for the recipe to build the vision grid for.
    """

    def __init__(self, config):
        PreTrainedPolicy.__init__(self, config)
        vision = Qwen2_5_VLVisionConfig(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=32,
            fullatt_block_indexes=[0],
        )
        visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(vision).eval()
        self.model = SimpleNamespace(visual=visual)

    def predict_action_chunk(self, batch, *, noise=None, vision_grid=None):
        vision_grid.check(self.model.visual, batch["image_grid_thw"])
        if noise is None:
            noise = torch.randn(1, CHUNK, self.config.max_action_dim, device=batch[OBS_STATE].device)
        reads = batch["pixel_values"].mean() + batch["input_ids"].float().mean()
        reads = reads + (batch["proprioception"] * batch["agent_pos_mask"]).mean()
        return (noise + reads)[..., : self.config.output_features[ACTION].shape[0]]


def make_recipe(device="cpu", task="pick up the cube") -> WallXExportRecipe:
    torch.manual_seed(0)
    identity = NormalizationMode.IDENTITY
    config = WallXConfig(
        device=device,
        recipe=None,
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        normalization_mapping={"VISUAL": identity, "STATE": identity, "ACTION": identity},
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 12)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    preprocessor, postprocessor = make_wall_x_pre_post_processors(config)
    (step,) = (step for step in preprocessor.steps if isinstance(step, WallXTokenizerStep))
    step._processor = processor()
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device=device), seed=0, dataset=None, task=task, tolerance=None
    )
    return WallXExportRecipe(PixelWallX(config).eval(), preprocessor, postprocessor, cfg)
