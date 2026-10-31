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

"""A stand-in Wall-OSS 0.5 for its export recipe, on its preprocessor with a tiny Qwen2.5-VL processor."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("torchvision")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor  # noqa: E402
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig  # noqa: E402
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VisionTransformerPretrainedModel,
)

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.policies.wall_oss_05.configuration_wall_oss_05 import WallOSS05Config  # noqa: E402
from lerobot.policies.wall_oss_05.export_wall_oss_05 import WallOSS05ExportRecipe  # noqa: E402
from lerobot.policies.wall_oss_05.modeling_wall_oss_05 import WallOSS05Policy  # noqa: E402
from lerobot.policies.wall_oss_05.processor_wall_oss_05 import (  # noqa: E402
    WALL_OSS_ACTION_TOKEN,
    WALL_OSS_IMAGE_TOKEN,
    WALL_OSS_PROPRIO_TOKEN,
    WallOSS05TokenizerStep,
    make_wall_oss_05_pre_post_processors,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

CAMERA = "observation.images.front"
CHUNK = 2
SPECIAL = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"]
WORDS = ["Proprioception", "Instruction", "pick", "up", "the", "cube", ":", "."]


def processor() -> SimpleNamespace:
    """A Qwen2.5-VL processor whose tokenizer writes each digit of a number alone, as Qwen's does."""
    special = [*SPECIAL, WALL_OSS_IMAGE_TOKEN, WALL_OSS_PROPRIO_TOKEN, WALL_OSS_ACTION_TOKEN]
    vocabulary = [*special, *WORDS, *map(str, range(10)), "-"]
    words = Tokenizer(models.WordLevel({word: i for i, word in enumerate(vocabulary)}, unk_token="<unk>"))
    digits = pre_tokenizers.Digits(individual_digits=True)
    words.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), digits])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=words,
        pad_token="<pad>",
        unk_token="<unk>",
        additional_special_tokens=special[2:],
        padding_side="left",
    )
    # As many pixels as the cameras', so the image processor does not resize them.
    image_processor = Qwen2VLImageProcessor(
        patch_size=14, merge_size=2, temporal_patch_size=2, min_pixels=224 * 448, max_pixels=224 * 448
    )
    return SimpleNamespace(tokenizer=tokenizer, image_processor=image_processor)


class PromptReader(WallOSS05Policy):
    """A stand-in Wall-OSS whose actions read the prompt, token by token, the pixels and the noise.

    It has a small vision tower where Wall-OSS has its own, for the recipe to build the vision grid for.
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
        # Weighted by position from the end, where Wall-OSS reads its action tokens, so a token out of place
        # moves the actions; padding outside the mask does not.
        ids, mask = batch["input_ids"], batch["attention_mask"]
        positions = torch.arange(ids.shape[1], 0, -1, device=noise.device)
        tokens = (ids * positions * mask).sum() + (batch["moe_token_types"] * positions).sum()
        reads = tokens.float() * 1e-4 + batch["pixel_values"].mean() + batch["dof_mask"].float().mean()
        return (noise + reads)[..., : self.config.output_features[ACTION].shape[0]]


def make_recipe(device="cpu", long_side=448, short_side=None) -> WallOSS05ExportRecipe:
    torch.manual_seed(0)
    identity = NormalizationMode.IDENTITY
    config = WallOSS05Config(
        device=device,
        recipe=None,
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        camera_key_mapping={CAMERA: "face_view"},
        normalization_mapping={"VISUAL": identity, "STATE": identity, "ACTION": identity},
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            CAMERA: PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, short_side or long_side // 2, long_side)
            ),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    preprocessor, postprocessor = make_wall_oss_05_pre_post_processors(config)
    (step,) = (step for step in preprocessor.steps if isinstance(step, WallOSS05TokenizerStep))
    step._processor = processor()
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device=device), seed=0, dataset=None, task="pick up the cube", tolerance=None
    )
    return WallOSS05ExportRecipe(PromptReader(config).eval(), preprocessor, postprocessor, cfg)
