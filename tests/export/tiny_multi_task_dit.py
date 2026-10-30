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

"""Multi-Task DiT's CLIP encoders and tokenizer made locally, tiny, for tests that must not reach the Hub."""

from typing import Any

# The words of the test tasks, and the size of the images the tiny CLIP takes.
VOCABULARY = ["<pad>", "<unk>", "pick", "up", "the", "cube"]
IMAGE_SIZE = 32


def use_tiny_clip(patch: Any) -> None:
    """Give Multi-Task DiT tiny CLIP encoders and a word tokenizer in place of the ones it loads from the Hub.

    `patch.setattr(target, name, value)` makes each change: a pytest `monkeypatch`, or a namespace holding the
    builtin `setattr` in a process that stays tiny.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        CLIPTextConfig,
        CLIPTextModel,
        CLIPVisionConfig,
        CLIPVisionModel,
        PreTrainedTokenizerFast,
    )

    from lerobot.processor import tokenizer_processor

    vision = CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=IMAGE_SIZE,
    )
    text = CLIPTextConfig(
        vocab_size=len(VOCABULARY),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=8,
    )
    words = Tokenizer(models.WordLevel({w: i for i, w in enumerate(VOCABULARY)}, unk_token="<unk>"))
    words.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=words, pad_token="<pad>", unk_token="<unk>")
    patch.setattr(CLIPVisionModel, "from_pretrained", lambda *args, **kwargs: CLIPVisionModel(vision))
    patch.setattr(CLIPTextModel, "from_pretrained", lambda *args, **kwargs: CLIPTextModel(text))
    patch.setattr(tokenizer_processor.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
