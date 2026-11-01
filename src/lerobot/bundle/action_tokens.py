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

"""The actions of a policy that writes them as FAST tokens, decoded without PyTorch.

FAST (https://huggingface.co/physical-intelligence/fast) turns an action chunk into text: it takes the discrete
cosine transform (DCT) of each action component over the chunk, rounds the coefficients, writes each one as a
character, and compresses the text with a byte-level BPE tokenizer. pi0-FAST generates that text after `Action: `
and ends it with a marker. [`ActionDecoder`] undoes each step as the policy's `detokenize_actions` does, with the
tokens' text from the bundle's copy of the FAST tokenizer file and the inverse DCT as a NumPy matrix product.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import ActionTokens

# What a Transformers tokenizer with `clean_up_tokenization_spaces` replaces in the decoded text, in this order.
_CLEAN_UP = (
    (" .", "."),
    (" ?", "?"),
    (" !", "!"),
    (" ,", ","),
    (" ' ", "'"),
    (" n't", "n't"),
    (" 'm", "'m"),
    (" 's", "'s"),
    (" 've", "'ve"),
    (" 're", "'re"),
)


def _byte_level_bytes() -> dict[str, int]:
    """The byte each character of a byte-level token stands for, in GPT-2's table of bytes to characters."""
    printable = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("¡"), ord("¬") + 1),
        *range(ord("®"), ord("ÿ") + 1),
    ]
    others = [byte for byte in range(256) if byte not in printable]
    return {chr(byte): byte for byte in printable} | {chr(256 + i): byte for i, byte in enumerate(others)}


class ActionDecoder:
    """Turns the token ids a policy generated into its normalized actions, as a manifest's `action_tokens` says.

    Args:
        tokenizer (`dict[str, Any]`):
            The FAST tokenizer file, as JSON.
        spec (`ActionTokens`):
            The manifest's `action_tokens`.
    """

    def __init__(self, tokenizer: dict[str, Any], spec: ActionTokens) -> None:
        """Read the text of each token and make the inverse DCT."""
        model, decoder = tokenizer.get("model") or {}, tokenizer.get("decoder") or {}
        if model.get("type") != "BPE" or decoder.get("type") != "ByteLevel" or tokenizer.get("added_tokens"):
            raise ValueError(
                "The bundle's FAST tokenizer must be a byte-level BPE without added tokens, as FAST trains it. "
                "Export the policy again."
            )
        if not spec.prefix:
            raise ValueError("The bundle's action tokens name no answer prefix. Export the policy again.")
        self._spec = spec
        self._text = {token_id: token for token, token_id in model["vocab"].items()}
        self._bytes = _byte_level_bytes()
        n = spec.horizon
        time, frequency = np.arange(n)[:, None], np.arange(n)[None, :]
        # The orthonormal DCT-III, the inverse of the orthonormal DCT-II that FAST takes over the chunk.
        self._inverse_dct = np.sqrt(2 / n) * np.cos(np.pi * (2 * time + 1) * frequency / (2 * n))
        self._inverse_dct[:, 0] = np.sqrt(1 / n)

    @classmethod
    def read(cls, folder: Path, spec: ActionTokens) -> ActionDecoder:
        """The decoder of the bundle in `folder`."""
        return cls(json.loads((folder / spec.tokenizer).read_text()), spec)

    def __call__(self, token_ids: np.ndarray) -> np.ndarray:
        """The normalized actions of the generated `token_ids`, float32 `[horizon, action_dim]`."""
        spec = self._spec
        ids = [int(i) for i in np.asarray(token_ids).reshape(-1)]
        if spec.start and ids[: len(spec.start)] != spec.start:
            raise RuntimeError(
                'The exported policy generated tokens that do not start with "Action:", which its decoding '
                "requires (validate_action_token_prefix); the actions were not sent."
            )
        if spec.end in ids:
            ids = ids[: ids.index(spec.end)]
        ids = _drop(ids, spec.prefix)
        coefficients = self._coefficients([spec.id_offset - i for i in ids])
        return (self._inverse_dct @ (coefficients / spec.scale)).astype(np.float32)

    def _coefficients(self, fast_ids: list[int]) -> np.ndarray:
        """The DCT coefficients `[horizon, action_dim]`, one per character, cut or padded with zeros to fit."""
        spec = self._spec
        size = spec.horizon * spec.action_dim
        # The tokenizers library refuses a negative id, and the policy then decodes the chunk as zeros.
        if any(i < 0 for i in fast_ids):
            return np.zeros((spec.horizon, spec.action_dim))
        values = np.array([ord(c) for c in self._decode(fast_ids)], dtype=np.int64)[:size] + spec.min_token
        return np.pad(values, (0, size - len(values))).reshape(spec.horizon, spec.action_dim)

    def _decode(self, fast_ids: list[int]) -> str:
        """The text of the FAST tokens, as the tokenizers library decodes it, which skips ids it does not know."""
        data = bytearray()
        for token in (self._text[i] for i in fast_ids if i in self._text):
            if all(c in self._bytes for c in token):
                data.extend(self._bytes[c] for c in token)
            else:
                data.extend(token.encode())
        text = data.decode("utf-8", errors="replace")
        if self._spec.clean_up_spaces:
            for old, new in _CLEAN_UP:
                text = text.replace(old, new)
        return text


def _drop(ids: list[int], prefix: list[int]) -> list[int]:
    """`ids` without each occurrence of `prefix`, searched left to right as the policy removes them."""
    i = 0
    while i <= len(ids) - len(prefix):
        if ids[i : i + len(prefix)] == prefix:
            ids = ids[:i] + ids[i + len(prefix) :]
        else:
            i += 1
    return ids
