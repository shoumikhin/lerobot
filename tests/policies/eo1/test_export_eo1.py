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

"""The EO-1 export recipe, on EO-1's preprocessor with the NumPy Qwen2-VL image processor and a stand-in."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("torchvision")

from transformers import Qwen2VLImageProcessorPil  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.export.recipe import ActionChunkRecipe, random_frame  # noqa: E402
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
    """A stand-in EO-1 whose actions read the pixel values, the prompt, the state and the starting noise."""

    def __init__(self, config):
        PreTrainedPolicy.__init__(self, config)

    def predict_action_chunk(self, batch, *, noise=None):
        # The processor step consumes the rendered messages; EO-1 passes every other entry to its model.
        assert "messages" not in batch
        if noise is None:
            noise = torch.randn(
                1, self.config.chunk_size, self.config.max_action_dim, device=batch[OBS_STATE].device
            )
        reads = batch["pixel_values"].mean() + batch["input_ids"].float().mean() + batch[OBS_STATE].mean()
        reads = reads + batch["action_token_start"] + batch["state_token_id"]
        return (noise + reads)[..., : self.config.output_features[ACTION].shape[0]]


def make_recipe(monkeypatch, recipe_class=EO1ExportRecipe, side=56, device="cpu") -> ActionChunkRecipe:
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
    return recipe_class(PixelEO1(config).eval(), preprocessor, postprocessor, cfg)


def program_inputs(recipe: ActionChunkRecipe) -> tuple[torch.Tensor, ...]:
    frame = random_frame(recipe.config.input_features, torch.Generator().manual_seed(1))
    noise = torch.randn(recipe.test_case().noise.shape, generator=torch.Generator().manual_seed(2))
    return tuple(torch.from_numpy(x) for x in (*frame.values(), noise.numpy()))


def test_the_program_keeps_the_prompt_and_computes_the_pixels(monkeypatch):
    recipe = make_recipe(monkeypatch)

    [part] = recipe.parts()
    program = torch.export.export(part.module, part.example_inputs, strict=False)

    inputs = program_inputs(recipe)
    with torch.no_grad():
        torch.testing.assert_close(program.module()(*inputs), part.module(*inputs), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_the_pixels_are_the_processors_on_cuda_too(monkeypatch):
    recipe = make_recipe(monkeypatch, device="cuda")

    # The eager checks compare the per-frame pixels with what the processor step computes.
    [part] = recipe.parts()


def test_a_camera_size_the_processor_resizes_is_refused(monkeypatch):
    recipe = make_recipe(monkeypatch, side=64)

    with pytest.raises(ValueError, match="64x64 becomes 56x56"):
        list(recipe.parts())


def test_the_saved_preprocessor_cannot_be_traced(monkeypatch):
    recipe = make_recipe(monkeypatch, recipe_class=ActionChunkRecipe)
    [part] = recipe.parts()

    # The NumPy image processor reads the traced pixels.
    with pytest.raises(RuntimeError, match=r"\.numpy\(\) is not supported for tensor subclasses"):
        torch.export.export(part.module, part.example_inputs, strict=False)
