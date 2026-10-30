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

"""The WALL-X export recipe, on WALL-X's preprocessor with a small Qwen-like processor and a stand-in."""

import re
import zlib
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("peft")
pytest.importorskip("transformers")

from transformers import Qwen2VLImageProcessor  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.export.recipe import ActionChunkRecipe, random_frame  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.policies.wall_x import WallXConfig  # noqa: E402
from lerobot.policies.wall_x.export_wall_x import WallXExportRecipe  # noqa: E402
from lerobot.policies.wall_x.modeling_wall_x import WallXPolicy  # noqa: E402
from lerobot.policies.wall_x.processor_wall_x import WallXTokenizerStep  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

CAMERA = "observation.images.face_view"


class Tokenized(dict):
    __getattr__ = dict.__getitem__


class Tokenizer:
    """One id per special token such as `<|image_pad|>`, one per character otherwise."""

    pad_token_id = 0
    eos_token_id = 0

    @staticmethod
    def _ids(text):
        tokens = re.finditer(r"<\|[^|]+\|>|.", text, re.DOTALL)
        return [zlib.crc32(token.group(0).encode()) % 1000 + 1 for token in tokens]

    def encode(self, text, add_special_tokens=True):
        return self._ids(text)

    def convert_tokens_to_ids(self, token):
        return self._ids(token)[0]

    def __call__(self, texts, return_tensors=None, return_offsets_mapping=False, **kwargs):
        if isinstance(texts, str):
            return Tokenized(input_ids=self._ids(texts))
        input_ids = torch.tensor([self._ids(text) for text in texts])
        return Tokenized(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))


def make_recipe(recipe_class=WallXExportRecipe) -> ActionChunkRecipe:
    torch.manual_seed(0)
    config = WallXConfig(device="cpu", chunk_size=4, n_action_steps=4)
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
        CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    preprocessor, postprocessor = make_pre_post_processors(
        config, dataset_stats=None, dataset_meta=SimpleNamespace(fps=20)
    )
    step = next(s for s in preprocessor.steps if isinstance(s, WallXTokenizerStep))
    image_processor = Qwen2VLImageProcessor(patch_size=14, merge_size=2, temporal_patch_size=2)
    step._processor = SimpleNamespace(tokenizer=Tokenizer(), image_processor=image_processor)
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task="pick up the cube", tolerance=None
    )
    return recipe_class(PixelWallX(config).eval(), preprocessor, postprocessor, cfg)


class PixelWallX(WallXPolicy):
    """A stand-in WALL-X whose actions read the pixel values, the prompt, the state and the starting noise."""

    def __init__(self, config):
        PreTrainedPolicy.__init__(self, config)

    def predict_action_chunk(self, batch, *, noise=None):
        if noise is None:
            noise = torch.randn(1, self.config.chunk_size, self.config.max_action_dim)
        reads = batch["pixel_values"].mean() + batch["input_ids"].float().mean()
        reads = reads + batch["proprioception"].mean()
        return (noise + reads)[..., : self.config.output_features[ACTION].shape[0]]


def program_inputs(recipe: ActionChunkRecipe) -> tuple[torch.Tensor, ...]:
    frame = random_frame(recipe.config.input_features, torch.Generator().manual_seed(1))
    noise = torch.randn(recipe.test_case().noise.shape, generator=torch.Generator().manual_seed(2))
    return tuple(torch.from_numpy(x) for x in (*frame.values(), noise.numpy()))


def test_the_program_keeps_the_prompt_and_computes_the_pixels_and_state():
    recipe = make_recipe()

    [part] = recipe.parts()
    program = torch.export.export(part.module, part.example_inputs, strict=False)

    inputs = program_inputs(recipe)
    with torch.no_grad():
        torch.testing.assert_close(program.module()(*inputs), part.module(*inputs), rtol=0, atol=0)


def test_the_saved_preprocessor_cannot_be_traced():
    recipe = make_recipe(recipe_class=ActionChunkRecipe)
    [part] = recipe.parts()

    with pytest.raises(Exception, match="data-dependent|GuardOnDataDependentSymNode"):
        torch.export.export(part.module, part.example_inputs, strict=False)
