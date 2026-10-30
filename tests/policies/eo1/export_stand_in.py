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

"""A stand-in EO-1 for its export recipe, on EO-1's preprocessor with the NumPy Qwen2-VL image processor."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("torchvision")

from transformers import Qwen2VLImageProcessorPil  # noqa: E402
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig  # noqa: E402
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VisionTransformerPretrainedModel,
)

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.export.recipe import ActionChunkRecipe  # noqa: E402
from lerobot.policies.eo1.configuration_eo1 import EO1Config  # noqa: E402
from lerobot.policies.eo1.export_eo1 import EO1ExportRecipe  # noqa: E402
from lerobot.policies.eo1.modeling_eo1 import EO1Policy  # noqa: E402
from lerobot.policies.eo1.processor_eo1 import make_eo1_pre_post_processors  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

CAMERA = "observation.images.top"
CHUNK = 4
TOKEN_IDS = {"<|state_pad|>": 5, "<|action_pad|>": 6}
IMAGE_TOKEN = 9


class Processor:
    """Like Qwen2.5-VL's processor: NumPy pixels, and one image token per merged patch, counted in Python."""

    def __init__(self):
        self.image_processor = Qwen2VLImageProcessorPil(patch_size=14, merge_size=2, temporal_patch_size=2)
        self.tokenizer = SimpleNamespace(add_tokens=lambda *a, **k: None, convert_tokens_to_ids=TOKEN_IDS.get)

    def apply_chat_template(self, messages, processor_kwargs, **kwargs):
        images = [
            b["image"] for row in messages for m in row for b in m["content"] if b.get("type") == "image"
        ]
        vision = self.image_processor(
            images=images,
            min_pixels=processor_kwargs["min_pixels"],
            max_pixels=processor_kwargs["max_pixels"],
            return_tensors="pt",
        )
        image_tokens = int(vision["image_grid_thw"].prod(-1).sum()) // 4
        state, action = TOKEN_IDS["<|state_pad|>"], TOKEN_IDS["<|action_pad|>"]
        input_ids = torch.tensor([[1, *[IMAGE_TOKEN] * image_tokens, state, 7, *[action] * CHUNK, 2]])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": vision["pixel_values"],
            "image_grid_thw": vision["image_grid_thw"],
            "mm_token_type_ids": input_ids.eq(IMAGE_TOKEN).int(),
        }


class PixelEO1(EO1Policy):
    """A stand-in EO-1 whose actions read the pixel values, the prompt, the state and the starting noise.

    It has a small vision tower where EO-1 has its own, for the recipe to build the vision grid for.
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
        self.model = SimpleNamespace(vlm_backbone=SimpleNamespace(model=SimpleNamespace(visual=visual)))

    def predict_action_chunk(self, batch, *, noise=None, vision_grid=None):
        # The processor step consumes the rendered messages; EO-1 passes every other entry to its model.
        assert "messages" not in batch
        # As EO-1 does, refuse a vision grid built for another tower or other images.
        vision_grid.check(self.model.vlm_backbone.model.visual, batch["image_grid_thw"])
        if noise is None:
            noise = torch.randn(
                1, self.config.chunk_size, self.config.max_action_dim, device=batch[OBS_STATE].device
            )
        reads = batch["pixel_values"].mean() + batch["input_ids"].float().mean() + batch[OBS_STATE].mean()
        reads = reads + batch["state_token_id"]
        return (noise + reads)[..., : self.config.output_features[ACTION].shape[0]]


def make_recipe(monkeypatch, side=56, device="cpu", use_fast_processor=False) -> ActionChunkRecipe:
    monkeypatch.setattr(
        "lerobot.policies.eo1.processor_eo1.Qwen2_5_VLProcessor.from_pretrained",
        lambda *args, **kwargs: Processor(),
    )
    torch.manual_seed(0)
    config = EO1Config(
        device=device,
        vlm_base="dummy-qwen",
        vlm_config={},
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        image_min_pixels=56 * 56,
        image_max_pixels=56 * 56,
        use_fast_processor=use_fast_processor,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, side, side)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    preprocessor, postprocessor = make_eo1_pre_post_processors(config)
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device=device), seed=0, dataset=None, task="pick up the cube", tolerance=None
    )
    return EO1ExportRecipe(PixelEO1(config).eval(), preprocessor, postprocessor, cfg)
