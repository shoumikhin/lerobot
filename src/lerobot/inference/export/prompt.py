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

"""The task as an exported vision-language policy's programs take it: token embeddings, made on the host.

The policy's own tokenizer turns the task into token ids, and the ids pick their rows from the language
model's full embedding table. The table is a file in the folder that is mapped into memory, so only the
rows a task uses are ever read, and the board can drop them again: the programs hold no copy of it.

A loaded tokenizer holds about 300 MiB, which the allocator keeps even after it is deleted, so it runs in a
child process for each new task, about a second, and leaves the rollout's memory as it was. The folder keeps
the ids of the task it was exported with, so a rollout that runs that task never starts one.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

TASK_EMBEDDINGS = "task_embeddings"
TASK_LENGTH = "task_length"


# The policies tokenize with the `tokenizers` library behind their Transformers tokenizer, which turns off
# the padding and truncation the tokenizer file stores, so the child does too.
TOKENIZE = """
import json, sys
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file(sys.argv[1])
tokenizer.no_padding()
tokenizer.no_truncation()
print(json.dumps([tokenizer.encode(text).ids for text in json.load(sys.stdin)]))
"""


def tokenize(tokenizer_file: Path, texts: list[str]) -> list[list[int]]:
    """The token ids of each text, from a child process that loads `tokenizer_file`."""
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
    """The one tensor of a safetensors file, memory-mapped, as float32 or as the raw bits of bfloat16."""
    with open(path, "rb") as file:
        header_size = int.from_bytes(file.read(8), "little")
        header = json.loads(file.read(header_size))
    header.pop("__metadata__", None)
    ((_, tensor),) = header.items()
    dtypes = {"BF16": np.uint16, "F32": np.float32}
    if tensor["dtype"] not in dtypes:
        raise ValueError(f"{path} holds {tensor['dtype']} embeddings; expected BF16 or F32.")
    start, _ = tensor["data_offsets"]
    return np.memmap(path, dtypes[tensor["dtype"]], "r", 8 + header_size + start, tuple(tensor["shape"]))


class TaskPrompt:
    """Turns a task into the `task_embeddings` and `task_length` inputs, as `export.json`'s `prompt` describes.

    `prompt` names the tokenizer file, the embedding table, the text around the task, how the task is
    cleaned first (`default_task` for an empty one, `strip`, `replace`, and `formalize`: lowercase, then only
    letters, digits, spaces and underscores), and how many tokens the programs take.
    """

    def __init__(self, folder: Path, prompt: dict):
        """Read `prompt` and map the folder's embedding table; no tokenizer is loaded yet."""
        self._tokenizer_file = folder / prompt["tokenizer"]
        self._table = read_table(folder / prompt["embeddings"])
        self._text = prompt["text"]
        self._strip = prompt.get("strip", False)
        self._replace = prompt.get("replace", {})
        self._formalize = prompt.get("formalize", False)
        self._default = prompt.get("default_task")
        self.max_tokens = prompt["max_tokens"]
        # The task the folder was exported with, which its test case was computed for, and its ids.
        self.task = prompt["task"]
        self._ids = {self.task: prompt["task_ids"]}

    def text(self, task: str) -> str:
        """The task, cleaned and placed in the policy's text as its own processor does."""
        if not task and self._default is not None:
            task = self._default
        if self._strip:
            task = task.strip()
        for old, new in self._replace.items():
            task = task.replace(old, new)
        if self._formalize:
            task = "".join(c for c in task.lower() if c.isalnum() or c.isspace() or c == "_")
        # Not str.format, which would read braces in the task.
        return self._text.replace("{task}", task)

    def token_ids(self, task: str) -> list[int]:
        """The task's token ids, from a child process the first time a task is seen."""
        if task not in self._ids:
            exported, ids = tokenize(self._tokenizer_file, [self.text(self.task), self.text(task)])
            # Other ids for the exported task mean this tokenizer is not the one the policy was exported with.
            if exported != self._ids[self.task]:
                raise RuntimeError(
                    "This board's tokenizers library splits the task the policy was exported with into other "
                    "tokens than the export did. Install the version the export used."
                )
            self._ids[task] = ids
        return self._ids[task]

    def check(self, task: str) -> list[int]:
        """The task's token ids, or a ValueError if the programs cannot take that many."""
        ids = self.token_ids(task)
        if len(ids) > self.max_tokens:
            raise ValueError(
                f"The task {task!r} is {len(ids)} tokens long, but this exported policy takes at most "
                f"{self.max_tokens}. Shorten the task, or export the policy again for longer ones."
            )
        return ids

    def __call__(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        """The task's embedding rows, padded with zeros to `max_tokens`, and how many of them are real."""
        ids = self.check(task)
        rows = np.zeros((self.max_tokens, self._table.shape[1]), np.float32)
        looked_up = np.asarray(self._table[ids])
        if looked_up.dtype == np.uint16:
            # bfloat16 is the upper half of a float32, so this is exact.
            looked_up = (looked_up.astype(np.uint32) << 16).view(np.float32)
        rows[: len(ids)] = looked_up
        return rows, np.array([len(ids)], dtype=np.int64)
