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

"""A stand-in FineART-VLA for its export recipe, on FineART-VLA's preprocessor with a tiny tokenizer."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import AutoTokenizer, PreTrainedTokenizerFast  # noqa: E402

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.fineart_vla.configuration_fineart_vla import FineARTVLAConfig  # noqa: E402
from lerobot.policies.fineart_vla.export_fineart_vla import FineARTVLAExportRecipe  # noqa: E402
from lerobot.policies.fineart_vla.modeling_fineart_vla import FineARTVLAPolicy  # noqa: E402
from lerobot.policies.fineart_vla.processor_fineart_vla import (  # noqa: E402
    make_fineart_vla_pre_post_processors,
)
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

CAMERA = "observation.images.front"
CHUNK = 2
SPECIAL = ["<pad>", "<eos>", "<bos>", "<unk>"]
WORDS = ["User", "Assistant", "State", "pick", "up", "the", "cube", ":", ",", ";", "-"]


def tokenizer() -> PreTrainedTokenizerFast:
    """A tokenizer that writes each digit of a number as its own token, as PaliGemma's does."""
    vocabulary = [*SPECIAL, *WORDS, *map(str, range(10))]
    words = Tokenizer(models.WordLevel({word: i for i, word in enumerate(vocabulary)}, unk_token="<unk>"))
    digits = pre_tokenizers.Digits(individual_digits=True)
    words.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), digits])
    return PreTrainedTokenizerFast(
        tokenizer_object=words, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>", unk_token="<unk>"
    )


class PromptReader(FineARTVLAPolicy):
    """A stand-in FineART-VLA whose actions read the prompt, token by token, the pixels and the noise."""

    def __init__(self, config):
        PreTrainedPolicy.__init__(self, config)

    def predict_action_chunk(self, batch, *, noise=None):
        if noise is None:
            noise = torch.randn(1, CHUNK, self.config.max_action_dim, device=batch[OBS_STATE].device)
        # Weighted by position, so a token out of place moves the actions; padding outside the mask does not.
        tokens, mask = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        positions = torch.arange(1, tokens.shape[1] + 1, device=noise.device)
        reads = (tokens * positions * mask).sum().float() * 1e-4 + batch[CAMERA].mean()
        return (noise + reads)[..., : self.config.output_features[ACTION].shape[0]]


def make_recipe(
    monkeypatch, device="cpu", text_recipe=True, task="pick up the cube"
) -> FineARTVLAExportRecipe:
    # Also the tokenizer of pi0.5's preprocessor, which a checkpoint without a text recipe has.
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer())
    if text_recipe:
        pytest.importorskip("datasets", reason="recipes require lerobot[dataset]")
    torch.manual_seed(0)
    identity = NormalizationMode.IDENTITY
    config = FineARTVLAConfig(
        device=device,
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        tokenizer_max_length=32,
        enable_fast_action_loss=False,
        **({} if text_recipe else {"recipe": None}),
        normalization_mapping={"VISUAL": identity, "STATE": identity, "ACTION": identity},
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    preprocessor, postprocessor = make_fineart_vla_pre_post_processors(config)
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device=device), seed=0, dataset=None, task=task, tolerance=None
    )
    return FineARTVLAExportRecipe(PromptReader(config).eval(), preprocessor, postprocessor, cfg)
