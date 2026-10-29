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

"""A bundle's task tokens: the prompt text, the token rows, new tasks, and the engine's task changes."""

import json
import threading
from dataclasses import asdict

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file, save_file as save_numpy
from safetensors.torch import save_file
from torch import Tensor, nn

from lerobot.inference.bundle import Bundle, BundleInferenceEngine, task_tokens as task_tokens_module
from lerobot.inference.bundle.manifest import (
    EMBEDDINGS_FILE,
    KNOWN_TASKS,
    MANIFEST,
    TASK_EMBEDDINGS,
    TASK_LENGTH,
    TOKENIZER,
    BundleManifest,
    TaskTokensSpec,
)
from lerobot.inference.bundle.task_tokens import TaskTokens, clean
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.export.torch_backend import make_bundle

TEXT = [["strip"], ["replace", "_", " "], ["template", "task: {task} ."]]
WORDS = ["task:", ".", "pick", "place", "the", "cube", "box", "and", "then", "it"]
# The ids of the prompts "task: pick the cube ." and "task: place the box ." in the test tokenizer.
PICK = [0, 2, 4, 5, 1]
PLACE = [0, 3, 4, 6, 1]


def write_tokenizer(path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    model = tokenizers.models.WordLevel({word: i for i, word in enumerate(WORDS)} | {"[UNK]": 99}, "[UNK]")
    tokenizer = tokenizers.Tokenizer(model)
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path))


def write_task_files(
    folder, table: Tensor, known: dict[str, list[int]], max_tokens: int = 6
) -> TaskTokensSpec:
    write_tokenizer(folder / TOKENIZER)
    save_file({"embeddings": table}, folder / EMBEDDINGS_FILE)
    (folder / KNOWN_TASKS).write_text(json.dumps(known))
    return TaskTokensSpec(TEXT, TOKENIZER, EMBEDDINGS_FILE, KNOWN_TASKS, max_tokens)


@pytest.mark.parametrize(
    "ops, task, expected",
    [
        (TEXT, "  pick_the_cube ", "task: pick the cube ."),
        ([["lower"], ["keep", "alnum", "space", "_"]], "Pick, the CUBE_2!", "pick the cube_2"),
        ([["default", "do it"], ["template", "{task}."]], "", "do it."),
    ],
)
def test_text_ops_make_the_prompt(ops, task, expected):
    assert clean(ops, task) == expected


@pytest.mark.parametrize(
    "change, error",
    [
        ({"text": [["shout"]]}, r"\['shout'\].*Upgrade LeRobot"),
        ({"vocabulary": "v.json"}, r"\['vocabulary'\].*Upgrade LeRobot"),
    ],
)
def test_manifest_refuses_task_tokens_it_does_not_know(change, error):
    spec = asdict(TaskTokensSpec(TEXT, TOKENIZER, EMBEDDINGS_FILE, KNOWN_TASKS, 6)) | change
    data = {
        "backend": "b",
        "runtime": "r",
        "programs": [{"name": "m", "file": "m", "inputs": [], "outputs": [ACTION]}],
        "action_scale": [1.0],
        "tolerance": 0.1,
        "task_tokens": spec,
        "format_version": 1,
    }
    with pytest.raises(ValueError, match=error):
        BundleManifest.from_dict(data)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_task_rows_come_from_the_table(tmp_path, dtype):
    table = torch.randn(len(WORDS), 4).to(dtype)
    tokens = TaskTokens(tmp_path, write_task_files(tmp_path, table, {"pick the cube": PICK}))
    rows, length = tokens("pick the cube")
    assert length.tolist() == [5] and rows.shape == (6, 4) and rows.dtype == np.float32
    np.testing.assert_array_equal(rows[:5], table[PICK].float().numpy())
    np.testing.assert_array_equal(rows[5:], 0)


def test_a_new_task_is_tokenized_once_in_a_child(tmp_path, monkeypatch):
    tokens = TaskTokens(
        tmp_path, write_task_files(tmp_path, torch.randn(len(WORDS), 4), {"pick the cube": PICK})
    )
    calls = []
    tokenize = task_tokens_module.tokenize
    monkeypatch.setattr(task_tokens_module, "tokenize", lambda *args: calls.append(args) or tokenize(*args))
    assert tokens.ids("pick the cube") == PICK and not calls
    assert tokens.ids("place_the box") == PLACE
    assert tokens.ids("place_the box") == PLACE and len(calls) == 1
    assert "at most 6" in tokens.error("pick the cube and then place it")


def test_a_new_task_without_the_tokenizers_package_names_the_install(tmp_path, monkeypatch):
    tokens = TaskTokens(
        tmp_path, write_task_files(tmp_path, torch.randn(len(WORDS), 4), {"pick the cube": PICK})
    )
    find_spec = task_tokens_module.importlib.util.find_spec
    monkeypatch.setattr(
        task_tokens_module.importlib.util,
        "find_spec",
        lambda name, *args: None if name == "tokenizers" else find_spec(name, *args),
    )
    assert tokens.ids("pick the cube") == PICK
    with pytest.raises(RuntimeError, match='pip install "tokenizers'):
        tokens.ids("place the box")


def test_a_tokenizer_that_splits_differently_is_refused(tmp_path):
    tokens = TaskTokens(
        tmp_path, write_task_files(tmp_path, torch.randn(len(WORDS), 4), {"pick the cube": [0, 2]})
    )
    with pytest.raises(RuntimeError, match="splits the exported tasks into other tokens"):
        tokens.ids("place the box")


class TaskScale(nn.Module):
    """Actions that depend on the task: the state times the sum of the task's rows."""

    def forward(self, state: Tensor, embeddings: Tensor, length: Tensor) -> Tensor:
        return (state * embeddings.sum() + length.float())[None]


def task_bundle(tmp_path):
    frame = {OBS_STATE: np.array([1.0, 2.0], np.float32)}
    table = torch.randn(len(WORDS), 4)
    rows = torch.zeros(6, 4)
    rows[:5] = table[PICK]
    bundle = make_bundle(
        tmp_path / "bundle",
        TaskScale(),
        {**frame, TASK_EMBEDDINGS: rows.numpy(), TASK_LENGTH: np.array([5])},
        task="pick the cube",
    )
    spec = write_task_files(bundle.path, table, {"pick the cube": PICK, "place the box": PLACE})
    data = json.loads((bundle.path / MANIFEST).read_text())
    data["task_tokens"] = asdict(spec)
    # The test case holds the frame only; the engine makes the task's values.
    case = load_file(bundle.path / "test.safetensors")
    del case[TASK_EMBEDDINGS], case[TASK_LENGTH]
    save_numpy(case, bundle.path / "test.safetensors")
    (bundle.path / MANIFEST).write_text(json.dumps(data))
    return bundle.path, table


def test_engine_runs_the_task_it_is_given(tmp_path):
    path, table = task_bundle(tmp_path)
    engine = BundleInferenceEngine(Bundle.read(path))
    state = np.array([1.0, 2.0], np.float32)
    np.testing.assert_allclose(
        engine.get_action({OBS_STATE: state}), state * table[PICK].sum().item() + 5, rtol=1e-5
    )
    assert engine.set_task("place_the box")
    engine.reset()
    place = table[PLACE].sum().item()
    np.testing.assert_allclose(engine.get_action({OBS_STATE: state}), state * place + 5, rtol=1e-5)
    assert not engine.set_task("pick the cube and then place it")


def test_tokenizing_a_new_task_does_not_hold_the_control_loop(tmp_path, monkeypatch):
    path, _ = task_bundle(tmp_path)
    engine = BundleInferenceEngine(Bundle.read(path))
    started, release = threading.Event(), threading.Event()
    tokenize = task_tokens_module.tokenize

    def slow_tokenize(*args):
        started.set()
        release.wait(10)
        return tokenize(*args)

    monkeypatch.setattr(task_tokens_module, "tokenize", slow_tokenize)
    results = []
    worker = threading.Thread(target=lambda: results.append(engine.set_task("place_the cube")))
    worker.start()
    assert started.wait(10)
    # While the child tokenizes, the robot keeps getting actions and a newer, known task is taken.
    assert engine.get_action({OBS_STATE: np.array([1.0, 2.0], np.float32)}) is not None
    assert engine.set_task("place the box")
    release.set()
    worker.join(10)
    assert results == [False] and engine.task == "place the box"
