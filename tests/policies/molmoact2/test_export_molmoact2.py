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

"""The MolmoAct2 export recipe, on MolmoAct2's preprocessor with a tiny local processor and a stand-in policy."""

import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("einops")

from tokenizers.pre_tokenizers import ByteLevel  # noqa: E402
from transformers import Qwen2Tokenizer  # noqa: E402

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.export.recipe import ActionChunkRecipe, random_frame  # noqa: E402
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config  # noqa: E402
from lerobot.policies.molmoact2.export_molmoact2 import MolmoAct2ExportRecipe  # noqa: E402
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy  # noqa: E402
from lerobot.policies.molmoact2.molmoact2_hf_model.processing_molmoact2 import IMAGE_TOKENS  # noqa: E402
from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

CAMERA = "observation.images.top"
# The tiny processor's image size: 2x2 patches of 14 pixels, pooled into one image token.
SIDE = 28
CHUNK = 4


def write_checkpoint(path):
    """What MolmoAct2's pack step loads from a checkpoint: a byte-level Qwen2 tokenizer and the processor config."""
    vocab = {symbol: index for index, symbol in enumerate(sorted(ByteLevel.alphabet()))}
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=[])
    special = ["<|image|>", "<state_start>", "<state_end>", "<action_start>", "<action_end>", *IMAGE_TOKENS]
    special += [f"<state_{index}>" for index in range(256)]
    tokenizer.add_special_tokens({"additional_special_tokens": special})
    tokenizer.save_pretrained(path)
    image_processor = {
        "crop_mode": "resize",
        "size": {"height": SIDE, "width": SIDE},
        "patch_size": 14,
        "pooling_size": [2, 2],
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "resample": 2,
    }
    config = {"image_processor": image_processor, "use_single_crop_col_tokens": False}
    (path / "processor_config.json").write_text(json.dumps(config))


class PixelMolmoAct2(MolmoAct2Policy):
    """A stand-in MolmoAct2 whose actions read the pixel values, the prompt and the starting noise."""

    def __init__(self, config):
        PreTrainedPolicy.__init__(self, config)

    def predict_action_chunk(self, batch, *, noise=None):
        if noise is None:
            noise = torch.randn(
                1, CHUNK, self.config.expected_max_action_dim, device=batch["input_ids"].device
            )
        reads = batch["pixel_values"].mean() + batch["input_ids"].float().mean()
        reads = reads + batch["attention_mask"].float().mean()
        return (noise + reads)[..., : self.config.output_features[ACTION].shape[0]]


def make_recipe(tmp_path, recipe_class=MolmoAct2ExportRecipe, side=SIDE, device="cpu") -> ActionChunkRecipe:
    write_checkpoint(tmp_path)
    torch.manual_seed(0)
    config = MolmoAct2Config(
        device=device,
        checkpoint_path=str(tmp_path),
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, side, side)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        normalization_mapping=dict.fromkeys(("VISUAL", "STATE", "ACTION", "ENV"), NormalizationMode.IDENTITY),
    )
    preprocessor, postprocessor = make_molmoact2_pre_post_processors(config)
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device=device), seed=0, dataset=None, task="pick up the cube", tolerance=None
    )
    return recipe_class(PixelMolmoAct2(config).eval(), preprocessor, postprocessor, cfg)


def program_inputs(recipe: ActionChunkRecipe) -> tuple[torch.Tensor, ...]:
    frame = random_frame(recipe.config.input_features, torch.Generator().manual_seed(1))
    noise = torch.randn(recipe.test_case().noise.shape, generator=torch.Generator().manual_seed(2))
    return tuple(torch.from_numpy(x) for x in (*frame.values(), noise.numpy()))


def test_the_program_writes_the_state_into_the_prompt_and_computes_the_pixels(tmp_path):
    recipe = make_recipe(tmp_path)

    [part] = recipe.parts()
    program = torch.export.export(part.module, part.example_inputs, strict=False)

    # Another frame than the ones the recipe checked, so another state: the eager module checks its token ids
    # and pixels against the pack step, and the program must compute the same actions.
    inputs = program_inputs(recipe)
    with torch.no_grad():
        torch.testing.assert_close(program.module()(*inputs), part.module(*inputs), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_the_state_tokens_and_pixels_are_the_pack_steps_on_cuda_too(tmp_path):
    recipe = make_recipe(tmp_path, device="cuda")

    # The eager checks compare the per-frame token ids and pixels with what the pack step computes.
    [part] = recipe.parts()


def test_a_camera_size_the_processor_resizes_is_refused(tmp_path):
    recipe = make_recipe(tmp_path, side=32)

    with pytest.raises(ValueError, match="32x32 becomes 28x28"):
        list(recipe.parts())


def test_the_saved_preprocessor_cannot_be_traced(tmp_path):
    recipe = make_recipe(tmp_path, recipe_class=ActionChunkRecipe)
    [part] = recipe.parts()

    # The pack step reads the traced pixels with NumPy.
    with pytest.raises(RuntimeError, match=r"\.numpy\(\) is not supported for tensor subclasses"):
        torch.export.export(part.module, part.example_inputs, strict=False)
