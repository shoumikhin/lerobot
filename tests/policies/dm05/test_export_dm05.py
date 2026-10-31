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

"""The DM05 export recipe, on DM05's preprocessor with a tiny Gemma3 processor and a stand-in policy."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("torchvision")

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.dm05.configuration_dm05 import DM05Config  # noqa: E402
from lerobot.policies.dm05.export_dm05 import DM05ExportRecipe  # noqa: E402
from lerobot.policies.dm05.modeling_dm05 import DM05Policy  # noqa: E402
from lerobot.policies.dm05.processor_dm05 import (  # noqa: E402
    MODEL_INPUT_PREFIX,
    make_dm05_pre_post_processors,
)
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

CAMERA = "observation.images.front"
CHUNK = 2
SPECIAL = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
WORDS = ["user", "model", "Overall", "speed", "Task", "pick", "up", "the", "cube", "Head", "image", "States"]
IMAGE_TOKENS = ["<start_of_image>", "<end_of_image>", "<image_soft_token>"]


def save_processor(path: Path, split_words: bool = True) -> None:
    """A Gemma3 processor whose tokenizer writes each digit of a number as its own token, as Gemma's does."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Gemma3ImageProcessor, Gemma3Processor, PreTrainedTokenizerFast

    vocabulary = [*SPECIAL, *WORDS, *map(str, range(10)), ":", ".", *IMAGE_TOKENS]
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(vocabulary)}, unk_token="<unk>"))
    if split_words:
        digits = pre_tokenizers.Digits(individual_digits=True)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), digits])
    text = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
        image_token="<image_soft_token>",
        boi_token="<start_of_image>",
        eoi_token="<end_of_image>",
        additional_special_tokens=["<start_of_turn>", "<end_of_turn>", *IMAGE_TOKENS],
    )
    chat_template = """{% for message in messages %}<start_of_turn>{{ message['role'] }}
{% for item in message['content'] %}{% if item['type'] == 'image' %}<start_of_image>{% else %}{{ item['text'] }}{% endif %}{% endfor %}<end_of_turn>
{% endfor %}{% if add_generation_prompt %}<start_of_turn>model
{% endif %}"""
    image_processor = Gemma3ImageProcessor(size={"height": 16, "width": 16}, do_pan_and_scan=False)
    Gemma3Processor(image_processor, text, chat_template=chat_template, image_seq_length=4).save_pretrained(
        path
    )


class PromptReader(DM05Policy):
    """A stand-in DM05 whose actions read the prompt's tokens where they stand, the pixels and the noise."""

    def __init__(self, config):
        PreTrainedPolicy.__init__(self, config)

    def predict_action_chunk(self, batch, *, noise=None):
        inputs = {key.removeprefix(MODEL_INPUT_PREFIX): value for key, value in batch.items()}
        if noise is None:
            noise = torch.randn(1, CHUNK, self.config.max_action_dim, device=inputs[OBS_STATE].device)
        # Weighted by position, so a token out of place moves the actions; padding outside the mask does not.
        positions = torch.arange(1, inputs["input_ids"].shape[1] + 1, device=noise.device)
        tokens = (inputs["input_ids"] * positions * inputs["attention_mask"]).sum()
        types = (inputs["token_type_ids"] * positions).sum()
        reads = (tokens + types).float() * 1e-4 + inputs["pixel_values"].mean() + inputs[OBS_STATE].mean()
        return (noise + reads)[..., : self.config.output_features[ACTION].shape[0]]


def make_recipe(tmp_path, device="cpu", split_words=True) -> DM05ExportRecipe:
    save_processor(tmp_path / "processor", split_words)
    torch.manual_seed(0)
    identity = NormalizationMode.IDENTITY
    config = DM05Config(
        device=device,
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        processor_name_or_path=str(tmp_path / "processor"),
        normalization_mapping={"VISUAL": identity, "STATE": identity, "ACTION": identity},
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 12)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    preprocessor, postprocessor = make_dm05_pre_post_processors(config)
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device=device), seed=0, dataset=None, task="pick up the cube", tolerance=None
    )
    return DM05ExportRecipe(PromptReader(config).eval(), preprocessor, postprocessor, cfg)


def program_inputs(recipe: DM05ExportRecipe, state: list[float]) -> tuple[torch.Tensor, ...]:
    frame = {**recipe.test_case().frame, OBS_STATE: np.array(state, dtype=np.float32)}
    noise = np.random.default_rng(2).standard_normal(recipe.test_case().noise.shape, dtype=np.float32)
    return tuple(torch.from_numpy(x) for x in (*frame.values(), noise))


def test_the_program_writes_each_states_numbers_into_the_prompt(tmp_path):
    recipe = make_recipe(tmp_path)

    [part] = recipe.parts()
    program = torch.export.export(part.module, part.example_inputs, strict=False)

    # Bins 0, 5 and 191: numbers of one, one and three digits, where the test frame's take three each.
    inputs = program_inputs(recipe, [-1.0, -0.96, 0.5])
    with torch.no_grad():
        torch.testing.assert_close(program.module()(*inputs), part.module(*inputs), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_the_prompt_and_pixels_are_the_processors_on_cuda_too(tmp_path):
    recipe = make_recipe(tmp_path, device="cuda")

    # The eager checks compare the prompt and the pixels with what the saved steps make.
    [part] = recipe.parts()


def test_a_tokenizer_that_writes_the_state_otherwise_is_refused(tmp_path):
    # It reads all the text between two special tokens as one word, so a number alone is not its own token there.
    recipe = make_recipe(tmp_path, split_words=False)

    with pytest.raises(ValueError, match="does not hold the state's numbers"):
        list(recipe.parts())
