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

"""A tiny pi0-FAST checkpoint with local tokenizers, for export tests on the CPU without the Hub."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

IMAGE = "observation.images.base_0_rgb"
WORDS = ["Task", "State", "Action", ":", ",", ";", "-", "|", "pick", "up", "the", "cube"]
FAST_TOKENS = 300
# The words at the end of the vocabulary that hold no FAST token, as PaliGemma's last 128 do.
SKIP_TOKENS = 4
VOCABULARY = 3 + len(WORDS) + 256 + FAST_TOKENS + SKIP_TOKENS
SCALE = 10.0
HORIZON, ACTION_DIM = 3, 3


class TinyFastProcessor:
    """FAST's processor for the tests: its byte-level BPE and its DCT settings, saved and loaded as FAST does.

    FAST's own processor class is code from the Hub, so the tests use this one in its place.
    """

    def __init__(self, bpe_tokenizer: Any, min_token: int, scale: float):
        self.bpe_tokenizer = bpe_tokenizer
        self.min_token = min_token
        self.scale = scale

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "TinyFastProcessor":
        from transformers import PreTrainedTokenizerFast

        settings = json.loads((Path(path) / "processor_config.json").read_text())
        return cls(PreTrainedTokenizerFast.from_pretrained(path), **settings)

    def save_pretrained(self, path: str | Path) -> None:
        self.bpe_tokenizer.save_pretrained(path)
        settings = {"min_token": self.min_token, "scale": self.scale}
        (Path(path) / "processor_config.json").write_text(json.dumps(settings))


def use_tiny_pi0_fast(patch: Any) -> None:
    """Give pi0-FAST a two-layer Gemma 16 wide with a small vocabulary, and the local FAST processor.

    `patch.setattr(target, name, value)` makes each change: a pytest `monkeypatch`, or a namespace holding the
    builtin `setattr` in a process that stays tiny.
    """
    from lerobot.policies.pi0_fast import modeling_pi0_fast
    from lerobot.processor import tokenizer_processor
    from tests.policies.common.meta_load import use_tiny_backbone

    use_tiny_backbone(patch, modeling_pi0_fast)
    tiny = modeling_pi0_fast.GemmaConfig(
        width=16, depth=2, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8
    )
    patch.setattr(modeling_pi0_fast, "get_gemma_config", lambda variant: tiny)
    paligemma = modeling_pi0_fast.PaliGemmaForConditionalGenerationWithPiGemma

    def small(config):
        # The policy fixes the vocabulary and the image projector at PaliGemma's sizes.
        config.text_config.vocab_size = VOCABULARY
        config.vision_config.projection_dim = tiny.width
        config.vision_config.intermediate_size = 32
        return paligemma(config=config)

    patch.setattr(modeling_pi0_fast, "PaliGemmaForConditionalGenerationWithPiGemma", small)
    # The policy loads FAST itself, and the processor step loads it again.
    for module in (modeling_pi0_fast, tokenizer_processor):
        patch.setattr(module, "AutoProcessor", TinyFastProcessor)


def save_tokenizer(path: Path) -> None:
    """A word-level tokenizer laid out like PaliGemma's for pi0-FAST: words, then the words that stand for FAST
    tokens, then the unused ones. It adds a start token and splits the state's numbers the same inside the prompt
    as alone."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    words = ["[PAD]", "[UNK]", "[BOS]", *WORDS, *map(str, range(256))]
    words += [f"<fast{i}>" for i in range(FAST_TOKENS)] + [f"<unused{i}>" for i in range(SKIP_TOKENS)]
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="[BOS] $A", special_tokens=[("[BOS]", 2)])
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]"
    ).save_pretrained(path)


def save_fast_tokenizer(path: Path) -> None:
    """A FAST tokenizer trained as FAST trains one: a byte-level BPE over the characters of the rounded DCT
    coefficients of action chunks, from an alphabet of every character the coefficients use."""
    from scipy.fft import dct
    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

    chunks = np.random.default_rng(0).uniform(-1, 1, (400, HORIZON, ACTION_DIM))
    coefficients = np.around(dct(chunks, axis=1, norm="ortho") * SCALE).astype(int).reshape(len(chunks), -1)
    min_token = int(coefficients.min())
    texts = ["".join(map(chr, row - min_token)) for row in coefficients]
    bpe = ByteLevelBPETokenizer()
    alphabet = [chr(i) for i in range(int(coefficients.max()) - min_token + 1)]
    trainer = BpeTrainer(vocab_size=FAST_TOKENS, initial_alphabet=alphabet, show_progress=False)
    bpe._tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=bpe, clean_up_tokenization_spaces=True)
    TinyFastProcessor(tokenizer, min_token, SCALE).save_pretrained(path)


def save_checkpoint(path: Path, patch: Any):
    """Save a tiny float32 pi0-FAST with its processors and tokenizers into `path`, and return its config."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi0_fast import PI0FastConfig, PI0FastPolicy
    from tests.policies.common.meta_load import save_checkpoint as save_weights

    use_tiny_pi0_fast(patch)
    save_tokenizer(path.parent / "tokenizer")
    save_fast_tokenizer(path.parent / "fast")
    config = PI0FastConfig(
        device="cpu",
        dtype=torch.float32,
        image_resolution=(28, 28),
        chunk_size=4,
        n_action_steps=HORIZON,
        max_state_dim=8,
        max_action_dim=8,
        tokenizer_max_length=48,
        max_decoding_steps=8,
        fast_skip_tokens=SKIP_TOKENS,
        # A random model does not start its answers with "Action:".
        validate_action_token_prefix=False,
        text_tokenizer_name=str(path.parent / "tokenizer"),
        action_tokenizer_name=str(path.parent / "fast"),
        input_features={
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    )
    save_weights(PI0FastPolicy, config, path)
    config.save_pretrained(path)
    stats = {
        name: {"mean": torch.zeros(3), "std": torch.ones(3), "q01": -torch.ones(3), "q99": torch.ones(3)}
        for name in (OBS_STATE, ACTION)
    }
    for processor in make_pre_post_processors(config, dataset_stats=stats):
        processor.save_pretrained(path)
    return config
