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

import numpy as np
import pytest
import torch

from lerobot.inference.export.prompt import TaskPrompt

pytest.importorskip("tokenizers")

WORDS = ["[UNK]", "Task:", "pick", "up", "the", "cube", "stack", "blocks", ",", "State:"]


@pytest.fixture
def folder(tmp_path):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    # Tokenizer files can store padding and truncation, which the policies' tokenizers turn off.
    tokenizer.enable_padding(length=8)
    tokenizer.enable_truncation(max_length=4)
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    torch.manual_seed(0)
    save_file({"weight": torch.randn(len(WORDS), 4).to(torch.bfloat16)}, tmp_path / "table.safetensors")
    return tmp_path


def prompt(folder, task="cube", **spec):
    """A prompt exported for `task`, with the ids the tokenizer gives it."""
    spec = {"tokenizer": "tokenizer.json", "embeddings": "table.safetensors", "max_tokens": 6, **spec}
    text = TaskPrompt(folder, {**spec, "task": task, "task_ids": []}).text(task)
    ids = [WORDS.index(word) if word in WORDS else 0 for word in text.split()]
    return TaskPrompt(folder, {**spec, "task": task, "task_ids": ids})


def test_task_rows_are_the_table_rows_of_its_tokens_exactly(folder):
    from safetensors.torch import load_file

    table = load_file(folder / "table.safetensors")["weight"]
    rows, length = prompt(folder, text="Task: {task} , State:", strip=True, replace={"_": " "})(" pick_up ")
    ids = [WORDS.index(w) for w in ("Task:", "pick", "up", ",", "State:")]
    assert length.tolist() == [len(ids)]
    np.testing.assert_array_equal(rows[: len(ids)], table[ids].float().numpy())
    assert not rows[len(ids) :].any()


def test_an_empty_task_becomes_the_default_task(folder):
    assert prompt(folder, text="{task}", default_task="stack the blocks").token_ids("") == [6, 4, 7]


def test_a_formalized_task_is_lowercase_letters_digits_and_spaces(folder):
    assert prompt(folder, text="{task}", formalize=True).token_ids("Pick UP, the cube!") == [2, 3, 4, 5]


def test_a_task_longer_than_the_programs_take_is_refused(folder):
    with pytest.raises(ValueError, match="7 tokens long, but this exported policy takes at most 6"):
        prompt(folder, text="{task}")("pick up the cube pick up the")


def test_the_exported_task_needs_no_tokenizer(folder):
    (folder / "tokenizer.json").unlink()
    assert prompt(folder, text="{task}").token_ids("cube") == [5]


def test_a_task_seen_before_needs_no_tokenizer(folder):
    exported = prompt(folder, text="{task}")
    ids = exported.token_ids("pick up the cube")
    (folder / "tokenizer.json").unlink()
    assert exported.token_ids("cube") == [5]
    assert exported.token_ids("pick up the cube") == ids == [2, 3, 4, 5]


def test_a_tokenizer_that_splits_the_exported_task_otherwise_is_refused(folder):
    exported = TaskPrompt(
        folder,
        {"tokenizer": "tokenizer.json", "embeddings": "table.safetensors", "max_tokens": 6, "text": "{task}"}
        | {"task": "cube", "task_ids": [4, 5]},
    )
    with pytest.raises(RuntimeError, match="other tokens than the export did"):
        exported.token_ids("pick")
