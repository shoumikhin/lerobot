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

"""The LaWAM export recipe, on LaWAM's preprocessor with a small Qwen-like processor and a stand-in model."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from transformers import Qwen2VLImageProcessor  # noqa: E402

from lerobot.export.recipe import ActionChunkRecipe, random_frame  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.lawam.export_lawam import LaWAMExportRecipe  # noqa: E402
from lerobot.policies.lawam.modeling_lawam import LaWAMPolicy  # noqa: E402
from lerobot.policies.lawam.processor_lawam import LaWAMQwenInputsProcessorStep  # noqa: E402
from tests.policies.lawam.test_lawam import _FakeNativeLaWAM, make_config  # noqa: E402

PLACEHOLDER = 99
IMAGE_TOKEN = 7


class GridPromptProcessor:
    """Builds the prompt as the Qwen processors do: one image token per merged patch of each image."""

    def __init__(self, image_tokens=None):
        self.image_processor = Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2)
        self.image_tokens = image_tokens or (lambda grid, images: int(grid.prod()) // 4)

    def apply_chat_template(self, messages, processor_kwargs, **kwargs):
        (content,) = (message["content"] for message in messages[0])
        images = [block["image"] for block in content if block["type"] == "image"]
        kwargs = {key: processor_kwargs[key] for key in ("do_rescale", "device", "return_tensors")}
        processed = self.image_processor(images=images, **kwargs)
        ids = [1, 2]
        for grid, image in zip(processed["image_grid_thw"], images, strict=True):
            ids += [IMAGE_TOKEN] * self.image_tokens(grid, image) + [3]
        ids += [PLACEHOLDER] * 16
        input_ids = torch.tensor([ids])
        return {**processed, "input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


class PixelLaWAM(_FakeNativeLaWAM):
    """A stand-in LaWAM whose actions read the pixel values, the prompt and the starting noise."""

    def predict_action(self, batch, noise=None):
        if noise is None:
            noise = torch.randn(1, self.chunk_size, self.action_dim)
        return noise + batch["pixel_values"].mean() + batch["input_ids"].float().mean()


def make_recipe(processor: GridPromptProcessor, recipe_class=LaWAMExportRecipe) -> ActionChunkRecipe:
    torch.manual_seed(0)
    config = make_config()
    preprocessor, postprocessor = make_pre_post_processors(
        config, dataset_stats=None, dataset_meta=SimpleNamespace(fps=20)
    )
    step = next(s for s in preprocessor.steps if isinstance(s, LaWAMQwenInputsProcessorStep))
    step._processor, step._placeholder_token_id = processor, PLACEHOLDER
    policy = LaWAMPolicy(config, native_model=PixelLaWAM()).eval()
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task="pick up the cube", tolerance=None
    )
    return recipe_class(policy, preprocessor, postprocessor, cfg)


def program_inputs(recipe: ActionChunkRecipe) -> tuple[torch.Tensor, ...]:
    frame = random_frame(recipe.config.input_features, torch.Generator().manual_seed(1))
    noise = torch.randn(recipe.test_case().noise.shape, generator=torch.Generator().manual_seed(2))
    return tuple(torch.from_numpy(x) for x in (*frame.values(), noise.numpy()))


def test_the_program_keeps_the_prompt_and_computes_the_pixels():
    recipe = make_recipe(GridPromptProcessor())

    [part] = recipe.parts()
    program = torch.export.export(part.module, part.example_inputs)

    inputs = program_inputs(recipe)
    with torch.no_grad():
        torch.testing.assert_close(program.module()(*inputs), part.module(*inputs), rtol=0, atol=0)


def test_the_saved_preprocessor_cannot_be_traced():
    recipe = make_recipe(GridPromptProcessor(), recipe_class=ActionChunkRecipe)
    [part] = recipe.parts()

    with pytest.raises(Exception, match="data-dependent"):
        torch.export.export(part.module, part.example_inputs)


def test_a_prompt_that_changes_with_the_pixels_is_refused():
    # One more image token when the first pixel is bright.
    processor = GridPromptProcessor(lambda grid, image: int(grid.prod()) // 4 + int(image[0, 0, 0] > 0.5))

    with pytest.raises(ValueError, match="prompt changed between two frames"):
        list(make_recipe(processor).parts())
