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

"""The FineART-VLA export recipe, on its text tokenizer step with a tiny local tokenizer and a stand-in policy."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.fineart_vla.configuration_fineart_vla import FineARTVLAConfig  # noqa: E402
from lerobot.policies.fineart_vla.export_fineart_vla import FineARTVLAExportRecipe  # noqa: E402
from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy  # noqa: E402
from lerobot.policies.fineart_vla.processor_fineart_vla import FineARTVLATextTokenizerStep  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.processor import (  # noqa: E402
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyProcessorPipeline,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

CAMERA = "observation.images.top"
CHUNK = 2
WORDS = ["User", "State", "pick", "up", "the", "cube", ":", ",", ";", "-"]


def save_tokenizer(path: Path, split_words: bool = True) -> None:
    """A tokenizer that adds a start token and writes each digit of a number as its own token, as Gemma's does."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocabulary = {word: i for i, word in enumerate(["<pad>", "<unk>", "<bos>", *WORDS, *map(str, range(10))])}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    if split_words:
        digits = pre_tokenizers.Digits(individual_digits=True)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), digits])
    tokenizer.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>", bos_token="<bos>"
    ).save_pretrained(path)


class PromptReader(FineARTVLAPolicy):
    """A stand-in FineART-VLA whose actions read the prompt's tokens where they stand, the state and the noise."""

    def __init__(self, config):
        PreTrainedPolicy.__init__(self, config)

    def predict_action_chunk(self, batch, *, noise=None):
        if noise is None:
            noise = torch.randn(1, CHUNK, self.config.max_action_dim, device=batch[OBS_STATE].device)
        tokens, mask = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        # Weighted by position, so a token out of place moves the actions.
        positions = torch.arange(1, tokens.shape[1] + 1, device=noise.device)
        reads = (tokens * positions * mask).sum().float() * 1e-4 + batch[OBS_STATE].mean()
        return (noise + reads)[..., : self.config.output_features[ACTION].shape[0]]


def make_recipe(tmp_path, device="cpu", split_words=True) -> FineARTVLAExportRecipe:
    save_tokenizer(tmp_path / "tokenizer", split_words)
    torch.manual_seed(0)
    config = FineARTVLAConfig(
        device=device,
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        enable_fast_action_loss=False,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    tokenizer = FineARTVLATextTokenizerStep(tokenizer_name=str(tmp_path / "tokenizer"), max_length=32)
    preprocessor = PolicyProcessorPipeline(
        steps=[AddBatchDimensionProcessorStep(), tokenizer, DeviceProcessorStep(device=device)]
    )
    postprocessor = PolicyProcessorPipeline(
        steps=[DeviceProcessorStep(device="cpu")],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device=device), seed=0, dataset=None, task="pick up the cube", tolerance=None
    )
    return FineARTVLAExportRecipe(PromptReader(config).eval(), preprocessor, postprocessor, cfg)


def program_inputs(recipe: FineARTVLAExportRecipe, state: list[float]) -> tuple[torch.Tensor, ...]:
    frame = {**recipe.test_case().frame, OBS_STATE: np.array(state, dtype=np.float32)}
    noise = np.random.default_rng(2).standard_normal(recipe.test_case().noise.shape, dtype=np.float32)
    return tuple(torch.from_numpy(x) for x in (*frame.values(), noise))


def test_the_program_writes_each_states_numbers_into_the_prompt(tmp_path):
    recipe = make_recipe(tmp_path)

    [part] = recipe.parts()
    program = torch.export.export(part.module, part.example_inputs, strict=False)

    # Bins -1, 6 and 255: numbers of two, one and three tokens, where the test frame's take three each.
    inputs = program_inputs(recipe, [-2.0, -0.95, 1.0])
    with torch.no_grad():
        torch.testing.assert_close(program.module()(*inputs), part.module(*inputs), rtol=0, atol=0)


def test_a_tokenizer_that_writes_the_state_otherwise_is_refused(tmp_path):
    # It reads all its text as one word, so a number alone is not its own token there.
    recipe = make_recipe(tmp_path, split_words=False)

    with pytest.raises(ValueError, match="does not hold the state's numbers once"):
        list(recipe.parts())
