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

"""A stand-in LaWAM for its export recipe, on LaWAM's preprocessor with the Qwen2-VL image processor."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from transformers import Qwen2VLImageProcessor  # noqa: E402
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel  # noqa: E402

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.lawam.configuration_lawam import LaWAMConfig  # noqa: E402
from lerobot.policies.lawam.export_lawam import LaWAMExportRecipe  # noqa: E402
from lerobot.policies.lawam.modeling_lawam import LaWAMPolicy  # noqa: E402
from lerobot.policies.lawam.processor_lawam import (  # noqa: E402
    LaWAMQwenInputsProcessorStep,
    make_lawam_pre_post_processors,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

CAMERA = "observation.images.front"
CHUNK = 4
IMAGE_TOKEN, PLACEHOLDER_TOKEN = 9, 99


class Processor:
    """Like Qwen3-VL's processor: the image processor's pixels, and one image token per merged patch."""

    def __init__(self, placeholder: str):
        self.image_processor = Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2)
        self.placeholder = placeholder

    def apply_chat_template(self, messages, processor_kwargs, **kwargs):
        content = [block for row in messages for message in row for block in message["content"]]
        images = [block["image"] for block in content if block["type"] == "image"]
        vision = self.image_processor(
            images=images, do_rescale=False, device=processor_kwargs["device"], return_tensors="pt"
        )
        image_tokens = int(vision["image_grid_thw"].prod(-1).sum()) // 4
        texts = [block["text"] for block in content if block["type"] == "text"]
        placeholders = sum(text.count(self.placeholder) for text in texts)
        input_ids = torch.tensor([[1, *[IMAGE_TOKEN] * image_tokens, 7, *[PLACEHOLDER_TOKEN] * placeholders]])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": vision["pixel_values"],
            "image_grid_thw": vision["image_grid_thw"],
        }


class PixelLaWAM(nn.Module):
    """A stand-in for LaWAM's model whose actions read the pixel values, the prompt, the camera and the noise.

    It has a small vision tower where LaWAM has its own, for the recipe to build the vision grid for.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        vision = Qwen3VLVisionConfig(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=32,
            deepstack_visual_indexes=[],
        )
        visual = Qwen3VLVisionModel._from_config(vision).eval()
        self.policy_backend = SimpleNamespace(vlm=SimpleNamespace(model=SimpleNamespace(visual=visual)))
        self.action_dim = action_dim

    def predict_action(self, batch, *, noise=None, vision_grid=None):
        # As LaWAM does, refuse a vision grid built for another tower or other images.
        vision_grid.check(self.policy_backend.vlm.model.visual, batch["image_grid_thw"])
        if noise is None:
            noise = torch.randn(1, CHUNK, self.action_dim, device=batch["pixel_values"].device)
        reads = batch["pixel_values"].mean() + batch["input_ids"].float().mean()
        reads = reads + batch["primary_image"].mean()
        reads = reads + batch["act_placeholder_mask"].sum() + 2 * batch["flow_placeholder_mask"].sum()
        return noise + reads


def make_recipe(device="cpu") -> LaWAMExportRecipe:
    torch.manual_seed(0)
    identity = NormalizationMode.IDENTITY
    config = LaWAMConfig(
        device=device,
        chunk_size=CHUNK,
        action_horizon=CHUNK,
        n_action_steps=2,
        base_vlm="dummy-qwen",
        lam_image_hw=(32, 32),
        normalization_mapping={"VISUAL": identity, "STATE": identity, "ACTION": identity},
        input_features={
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    preprocessor, postprocessor = make_lawam_pre_post_processors(config, dataset_meta=SimpleNamespace(fps=10))
    (step,) = (step for step in preprocessor.steps if isinstance(step, LaWAMQwenInputsProcessorStep))
    step._processor, step._placeholder_token_id = Processor(step.placeholder_token), PLACEHOLDER_TOKEN
    policy = LaWAMPolicy(config, native_model=PixelLaWAM(config.flow_action_dim))
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device=device), seed=0, dataset=None, task="pick up the cube", tolerance=None
    )
    return LaWAMExportRecipe(policy.eval(), preprocessor, postprocessor, cfg)
