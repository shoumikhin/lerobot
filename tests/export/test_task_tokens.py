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

"""A bundle's task tokens: the prompt text, the token rows, new tasks, and the task an engine runs."""

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file, save_file as save_numpy
from safetensors.torch import save_file
from torch import Tensor, nn

from lerobot.bundle import task_tokens as task_tokens_module
from lerobot.bundle.engine import BundleEngine
from lerobot.bundle.manifest import (
    EMBEDDINGS,
    KNOWN_TASKS,
    MANIFEST,
    TASK_EMBEDDINGS,
    TASK_LENGTH,
    TOKENIZER,
    Bundle,
    BundleManifest,
    TaskTokens,
)
from lerobot.bundle.task_tokens import TaskEncoder, clean
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.export.torch_backend import make_bundle, record_places
from tests.torch_free import assert_runs_without_torch, refusing_torch

OPS = [["strip"], ["replace", "_", " "], ["template", "task: {task} ."]]
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


def write_task_files(folder, table: Tensor, known: dict[str, list[int]], max_tokens: int = 6) -> TaskTokens:
    write_tokenizer(folder / TOKENIZER)
    save_file({"embeddings": table}, folder / EMBEDDINGS)
    (folder / KNOWN_TASKS).write_text(json.dumps(known))
    return TaskTokens(OPS, TOKENIZER, EMBEDDINGS, KNOWN_TASKS, max_tokens)


@pytest.mark.parametrize(
    "ops, task, expected",
    [
        (OPS, "  pick_the_cube ", "task: pick the cube ."),
        ([["lower"], ["keep", "alnum", "space", "_"]], "Pick, the CUBE_2!", "pick the cube_2"),
        ([["default", "do it"], ["template", "{task}."]], "", "do it."),
    ],
)
def test_prompt_ops_make_the_prompt(ops, task, expected):
    assert clean(ops, task) == expected


@pytest.mark.parametrize(
    "change, error",
    [
        ({"prompt_ops": [["shout"]]}, r"\['shout'\].*Upgrade LeRobot"),
        ({"vocabulary": "v.json"}, r"\['vocabulary'\].*Upgrade LeRobot"),
    ],
)
def test_manifest_refuses_task_tokens_it_does_not_know(change, error):
    spec = asdict(TaskTokens(OPS, TOKENIZER, EMBEDDINGS, KNOWN_TASKS, 6)) | change
    data = {
        "backend": "b",
        "runtime": "r",
        "programs": {"m": {"file": "m"}},
        "runs": [{"name": "m", "program": "m", "inputs": [], "outputs": [ACTION]}],
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
    tokens = TaskEncoder(tmp_path, write_task_files(tmp_path, table, {"pick the cube": PICK}))
    rows, length = tokens("pick the cube")
    assert length.tolist() == [5] and rows.shape == (6, 4) and rows.dtype == np.float32
    np.testing.assert_array_equal(rows[:5], table[PICK].float().numpy())
    np.testing.assert_array_equal(rows[5:], 0)


def test_a_new_task_is_tokenized_once_in_a_child(tmp_path, monkeypatch):
    tokens = TaskEncoder(
        tmp_path, write_task_files(tmp_path, torch.randn(len(WORDS), 4), {"pick the cube": PICK})
    )
    calls = []
    tokenize = task_tokens_module.tokenize
    monkeypatch.setattr(task_tokens_module, "tokenize", lambda *args: calls.append(args) or tokenize(*args))
    assert tokens.ids("pick the cube") == PICK and not calls
    assert tokens.ids("place_the box") == PLACE
    assert tokens.ids("place_the box") == PLACE and len(calls) == 1
    with pytest.raises(ValueError, match="at most 6"):
        tokens.check("pick the cube and then place it")


def test_a_new_task_without_the_tokenizers_package_names_the_install(tmp_path, monkeypatch):
    tokens = TaskEncoder(
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


def test_a_new_task_is_tokenized_with_the_tokenizers_package_this_process_sees(tmp_path, monkeypatch):
    # Found only through PYTHONPATH, as a package in the user site or installed with `pip install --target`.
    package = tmp_path / "site" / "tokenizers"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from types import SimpleNamespace\n"
        "class Tokenizer:\n"
        "    from_file = staticmethod(lambda path: Tokenizer())\n"
        "    no_padding = no_truncation = lambda self: None\n"
        "    encode = lambda self, text: SimpleNamespace(ids=[7] * len(text.split()))\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(package.parent))
    monkeypatch.syspath_prepend(str(package.parent))
    assert task_tokens_module.tokenize(tmp_path / TOKENIZER, ["place the box"]) == [[7, 7, 7]]


def test_a_tokenizer_that_splits_differently_is_refused(tmp_path):
    tokens = TaskEncoder(
        tmp_path, write_task_files(tmp_path, torch.randn(len(WORDS), 4), {"pick the cube": [0, 2]})
    )
    with pytest.raises(RuntimeError, match="splits the exported tasks into other tokens"):
        tokens.ids("place the box")


def test_task_rows_are_made_without_torch(tmp_path):
    """A known task and a new one; the child process that tokenizes the new one must not load PyTorch either."""
    spec = write_task_files(tmp_path, torch.randn(len(WORDS), 4), {"pick the cube": PICK})
    assert_runs_without_torch(
        """
        import json
        import sys
        from pathlib import Path

        from lerobot.bundle import task_tokens
        from lerobot.bundle.manifest import TaskTokens

        task_tokens.TOKENIZE = sys.argv[3]
        encoder = task_tokens.TaskEncoder(Path(sys.argv[1]), TaskTokens(**json.loads(sys.argv[2])))
        lengths = [encoder(task)[1].tolist() for task in ("pick the cube", "place_the box")]
        assert lengths == [[5], [5]], lengths
        """,
        str(tmp_path),
        json.dumps(asdict(spec)),
        refusing_torch(task_tokens_module.TOKENIZE),
    )


class TaskScale(nn.Module):
    """Actions that depend on the task: the state times the sum of the task's rows."""

    def forward(self, state: Tensor, embeddings: Tensor, length: Tensor) -> Tensor:
        return (state * embeddings.sum() + length.float())[None]


def task_bundle(tmp_path):
    frame = {OBS_STATE: np.array([1.0, 2.0], np.float32)}
    # Seeded, so that no expected action lands near zero, where rtol=1e-5 is finer than float32 rounding.
    table = torch.randn(len(WORDS), 4, generator=torch.Generator().manual_seed(1))
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
    state = np.array([1.0, 2.0], np.float32)
    pick, place = state * table[PICK].sum().item() + 5, state * table[PLACE].sum().item() + 5
    np.testing.assert_allclose(
        BundleEngine(Bundle.read(path)).get_action({OBS_STATE: state}), pick, rtol=1e-5
    )
    engine = BundleEngine(Bundle.read(path), task="place_the box")
    np.testing.assert_allclose(engine.get_action({OBS_STATE: state}), place, rtol=1e-5)
    with pytest.raises(ValueError, match="tokens long"):
        BundleEngine(Bundle.read(path), task="pick the cube and then place it")


def test_engine_refuses_a_run_that_writes_the_task_values(tmp_path):
    path, _ = task_bundle(tmp_path)
    data = json.loads((path / MANIFEST).read_text())
    run = data["runs"][0]
    data["runs"] = [{**run, "name": "retask", "outputs": [TASK_EMBEDDINGS]}, run]
    (path / MANIFEST).write_text(json.dumps(data))
    with pytest.raises(
        ValueError, match=r"run 'retask' writes \['task_embeddings'\], which the bundle makes"
    ):
        BundleEngine(Bundle.read(path))


def test_engine_places_the_task_values_once(tmp_path, monkeypatch):
    path, _ = task_bundle(tmp_path)
    placed = record_places(monkeypatch)
    engine = BundleEngine(Bundle.read(path), task="place_the box")
    loaded = len(placed)
    for _ in range(3):
        engine.reset()
        engine.get_action({OBS_STATE: np.array([1.0, 2.0], np.float32)})
    # At load: the test case's state and task, then the run's task once. Each chunk copies only the state.
    assert placed[loaded - 2 : loaded] == [(6, 4), (1,)] and placed[loaded:] == [(2,)] * 3
