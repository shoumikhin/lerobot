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

"""The task as a bundle's programs take it: the embedding rows of its tokens, made without PyTorch.

A vision-language policy reads its task as token embeddings. The bundle holds the policy's tokenizer file
and its token embedding table, and its manifest says how the task text is cleaned and placed in the prompt
(`task_tokens`). The table is mapped into memory, so only the rows a task uses are read.

A loaded tokenizer holds about 300 MiB that the allocator keeps after it is deleted, so a task the bundle
has not seen is tokenized in a short-lived child process, once. The tasks known at export, such as the
dataset's, come with their token ids and never start one.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess  # nosec B404
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .manifest import TaskTokensSpec

# Each op takes the text and its arguments from the manifest.
TEXT_OPS: dict[str, Callable[..., str]] = {
    "strip": lambda text: text.strip(),
    "lower": lambda text: text.lower(),
    "replace": lambda text, old, new: text.replace(old, new),
    # Keeps the characters of the classes named: "alnum", "space", or a literal character.
    "keep": lambda text, *kinds: "".join(c for c in text if any(_is_kind(c, kind) for kind in kinds)),
    "default": lambda text, value: text or value,
    "template": lambda text, template: template.replace("{task}", text),
}

# Like the policies' Transformers tokenizers, the child turns off any padding and truncation the file stores.
TOKENIZE = """
import json, sys
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(sys.argv[1])
tokenizer.no_padding()
tokenizer.no_truncation()
print(json.dumps([tokenizer.encode(text).ids for text in json.load(sys.stdin)]))
"""


def _is_kind(character: str, kind: str) -> bool:
    if kind == "alnum":
        return character.isalnum()
    if kind == "space":
        return character.isspace()
    return character == kind


def clean(text_ops: list[list[str]], task: str) -> str:
    """The prompt text `text_ops` makes of `task`, op by op."""
    for name, *args in text_ops:
        task = TEXT_OPS[name](task, *args)
    return task


def tokenize(tokenizer_file: Path, texts: list[str]) -> list[list[int]]:
    """The token ids of each text, from a child process that loads `tokenizer_file`."""
    if importlib.util.find_spec("tokenizers") is None:
        raise RuntimeError(
            "This task was not known when the policy was exported, and tokenizing it needs the tokenizers "
            'package: pip install "tokenizers>=0.22". Or use a task the bundle lists in known_tasks.json.'
        )
    result = subprocess.run(  # nosec B603
        [sys.executable, "-I", "-c", TOKENIZE, str(tokenizer_file)],
        input=json.dumps(texts),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Tokenizing the task failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def read_table(path: Path) -> np.ndarray:
    """The one tensor of a safetensors file, memory-mapped: float32, or the raw bits of bfloat16."""
    with open(path, "rb") as file:
        header_size = int.from_bytes(file.read(8), "little")
        header = json.loads(file.read(header_size))
    header.pop("__metadata__", None)
    ((_, tensor),) = header.items()
    dtypes = {"BF16": np.uint16, "F32": np.float32}
    if tensor["dtype"] not in dtypes:
        raise ValueError(f"{path.name} holds {tensor['dtype']} embeddings; expected BF16 or F32.")
    start, _ = tensor["data_offsets"]
    return np.memmap(path, dtypes[tensor["dtype"]], "r", 8 + header_size + start, tuple(tensor["shape"]))


class TaskTokens:
    """Turns a task into the embedding rows of its tokens and their count, as a manifest's `task_tokens` says.

    Args:
        folder (`Path`):
            The bundle folder.
        spec (`TaskTokensSpec`):
            The manifest's `task_tokens`.
    """

    def __init__(self, folder: Path, spec: TaskTokensSpec):
        """Map the embedding table and read the known tasks; no tokenizer is loaded."""
        self._tokenizer_file = folder / spec.tokenizer
        self._table = read_table(folder / spec.table)
        self._text_ops = spec.text
        self.max_tokens = spec.max_tokens
        self._ids: dict[str, list[int]] = json.loads((folder / spec.known).read_text())
        if not self._ids:
            raise ValueError("The bundle's task tokens list no known task. Export the policy again.")

    def ids(self, task: str) -> list[int]:
        """The token ids of the prompt for `task`, tokenized in a child process the first time it is seen.

        Blocks for about a second for a new task, so call it outside any lock the control loop takes.
        """
        if task not in self._ids:
            known, known_ids = next(iter(self._ids.items()))
            checked, ids = tokenize(
                self._tokenizer_file, [clean(self._text_ops, known), clean(self._text_ops, task)]
            )
            if checked != known_ids:
                raise RuntimeError(
                    "The tokenizers library on this machine splits the exported tasks into other tokens than "
                    "the export did. Install the version the export used."
                )
            self._ids[task] = ids
        return self._ids[task]

    def error(self, task: str) -> str | None:
        """Why the programs cannot take `task`, or `None` if they can."""
        count = len(self.ids(task))
        if count > self.max_tokens:
            return (
                f"The task {task!r} is {count} tokens long, but this exported policy takes at most "
                f"{self.max_tokens}. Shorten the task, or export the policy again for longer ones."
            )
        return None

    def __call__(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        """The task's embedding rows as float32, padded with zeros to `max_tokens`, and how many are real."""
        ids = self.ids(task)
        rows = np.zeros((self.max_tokens, self._table.shape[1]), np.float32)
        looked_up = np.asarray(self._table[ids])
        if looked_up.dtype == np.uint16:
            # bfloat16 is the upper half of a float32, so this is exact.
            looked_up = (looked_up.astype(np.uint32) << 16).view(np.float32)
        rows[: len(ids)] = looked_up
        return rows, np.array([len(ids)], dtype=np.int64)
