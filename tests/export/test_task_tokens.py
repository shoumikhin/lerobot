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
import os
import stat
from collections import Counter
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file, save_file as save_numpy
from safetensors.torch import save_file
from torch import Tensor, nn

from lerobot.bundle import target as target_module, task_tokens as task_tokens_module
from lerobot.bundle.engine import BundleEngine
from lerobot.bundle.manifest import (
    EMBEDDINGS,
    FORMAT_VERSION,
    KNOWN_TASKS,
    MANIFEST,
    PROMPT_OPS,
    TASK_EMBEDDINGS,
    TASK_LENGTH,
    TEST_CASE,
    TOKENIZER,
    Bundle,
    BundleManifest,
    ProgramFiles,
    ProgramRun,
    TaskTokens,
)
from lerobot.bundle.task_tokens import TaskEncoder, clean
from lerobot.export.bundle import write_bundle
from lerobot.export.recipe import TaskTokenData
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.export.torch_backend import make_bundle, record_places
from tests.torch_free import assert_runs_without_torch, refusing_torch

OPS = [["strip"], ["replace", "_", " "], ["template", "task: {task} ."]]
WORDS = ["task:", ".", "pick", "place", "the", "cube", "box", "and", "then", "it"]
# The ids of the prompts "task: pick the cube ." and "task: place the box ." in the test tokenizer.
PICK = [0, 2, 4, 5, 1]
PLACE = [0, 3, 4, 6, 1]


def write_task_files(folder, table: Tensor, known: dict[str, list[int]]) -> TaskTokens:
    tokenizers = pytest.importorskip("tokenizers")
    model = tokenizers.models.WordLevel({word: i for i, word in enumerate(WORDS)} | {"[UNK]": 99}, "[UNK]")
    tokenizer = tokenizers.Tokenizer(model)
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(folder / TOKENIZER))
    save_file({"embeddings": table}, folder / EMBEDDINGS)
    (folder / KNOWN_TASKS).write_text(json.dumps(known))
    return TaskTokens(OPS, TOKENIZER, EMBEDDINGS, KNOWN_TASKS, 6)


def write_task_bundle(tmp_path, tokens: TaskTokenData | None, spec: TaskTokens | None, frame=None):
    """Write a bundle of the exported task "pick the cube" with `write_bundle`, as the export does."""
    folder = tmp_path / "bundle"
    folder.mkdir(parents=True)
    manifest = BundleManifest(
        backend="b",
        runtime="r",
        programs={"m": ProgramFiles("m")},
        runs=[ProgramRun("m", "m", [OBS_STATE], [ACTION])],
        task_tokens=spec,
        task="pick the cube",
        action_scale=[1.0],
        tolerance=0.1,
    )
    config = SimpleNamespace(save_pretrained=lambda path: (path / "config.json").write_text("{}"))
    frame = {OBS_STATE: np.zeros(1, np.float32)} if frame is None else frame
    case = SimpleNamespace(frame=frame, actions=np.zeros((1, 1), np.float32), noise=None)
    write_bundle(SimpleNamespace(path=folder, manifest=manifest), case, config, task_tokens=tokens)
    return folder


def test_the_export_writes_task_files_that_others_can_read_and_the_bundle_reads_back(tmp_path):
    """safetensors' own save_file makes files only their owner can read."""
    table = torch.randn(len(WORDS), 2).to(torch.bfloat16)
    spec = write_task_files(tmp_path, table, {})
    tokens = TaskTokenData(OPS, tmp_path / TOKENIZER, table, {"pick the cube": PICK}, 6)
    folder = write_task_bundle(tmp_path, tokens, spec)
    (tmp_path / "new").touch()
    usual = stat.S_IMODE((tmp_path / "new").stat().st_mode)
    assert stat.S_IMODE((folder / EMBEDDINGS).stat().st_mode) == usual
    encoder = TaskEncoder(folder, spec)
    assert encoder.ids("pick the cube") == PICK and encoder.ids("place the box") == PLACE
    np.testing.assert_array_equal(encoder("pick the cube")[0][:5], table[PICK].float().numpy())


@pytest.mark.parametrize(
    "known, frame, table, error",
    [
        ({"place the box": PLACE}, None, None, "known tasks do not hold the exported task 'pick the cube'"),
        (None, {TASK_LENGTH: np.array([5])}, None, r"frame holds \['task_length'\], which the bundle makes"),
        (None, None, torch.ones(len(WORDS), 2).half(), "embedding table is torch.float16"),
    ],
)
def test_the_export_refuses_task_tokens_a_bundle_would_fail_on(tmp_path, known, frame, table, error):
    spec = write_task_files(tmp_path, torch.ones(len(WORDS), 2), {})
    table = torch.ones(len(WORDS), 2) if table is None else table
    tokens = TaskTokenData(OPS, tmp_path / TOKENIZER, table, known or {"pick the cube": PICK}, 6)
    with pytest.raises(ValueError, match=error):
        write_task_bundle(tmp_path, tokens, spec, frame)


def test_the_export_refuses_task_files_without_task_tokens_or_the_other_way_round(tmp_path):
    spec = write_task_files(tmp_path, torch.ones(len(WORDS), 2), {})
    tokens = TaskTokenData(OPS, tmp_path / TOKENIZER, torch.ones(len(WORDS), 2), {"pick the cube": PICK}, 6)
    for given, named in [(tokens, None), (None, spec)]:
        with pytest.raises(ValueError, match="exactly when its manifest names task token files"):
            write_task_bundle(tmp_path / str(given is None), given, named)


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


def test_the_manifest_lists_the_prompt_ops_the_bundle_runs():
    assert task_tokens_module._PROMPT_OPS.keys() == PROMPT_OPS.keys()


@pytest.mark.parametrize(
    "change, error",
    [
        ({"prompt_ops": [["shout"]]}, r"\['shout'\].*Upgrade LeRobot"),
        ({"padding": "left"}, r"\['padding'\].*Upgrade LeRobot"),
        ({"prompt_ops": [["keep", "alnum", "digit"]]}, r"\['digit'\].*Upgrade LeRobot"),
        ({"prompt_ops": [["replace", "_"]]}, r"\['task_tokens'\] in a form"),
        ({"prompt_ops": [["template"]]}, r"\['task_tokens'\] in a form"),
        ({"prompt_ops": [["strip", "x"]]}, r"\['task_tokens'\] in a form"),
        ({"prompt_ops": [["keep"]]}, r"\['task_tokens'\] in a form"),
        ({"prompt_ops": [["replace", "_", 1]]}, r"\['task_tokens'\] in a form"),
        ({"max_tokens": 0}, r"\['task_tokens'\] in a form"),
        ({"max_tokens": "6"}, r"\['task_tokens'\] in a form"),
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
        "format_version": FORMAT_VERSION,
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
    # A task that cleans to the prompt of a known one needs no tokenizer either.
    assert tokens.ids("pick the cube") == PICK and tokens.ids(" pick_the cube") == PICK and not calls
    assert tokens.ids("place_the box") == PLACE
    assert tokens.ids("place the box") == PLACE and len(calls) == 1
    # "task: pick the cube then ." is exactly the budget of 6 tokens.
    tokens.ids("pick the cube then")
    with pytest.raises(ValueError, match="at most 6"):
        tokens.ids("pick the cube and then place it")


@pytest.mark.parametrize("ids", [[0, 2, len(WORDS)], [0, -1]])
def test_known_ids_outside_the_table_are_refused(tmp_path, ids):
    tokens = TaskEncoder(tmp_path, write_task_files(tmp_path, torch.randn(len(WORDS), 4), {"pick it": ids}))
    with pytest.raises(ValueError, match="embedding table of 10 rows does not hold"):
        tokens("pick it")


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
    with pytest.raises(ValueError, match='pip install "tokenizers'):
        tokens.ids("place the box")


def test_a_new_task_is_tokenized_with_the_tokenizers_package_on_pythonpath(tmp_path, monkeypatch):
    # Found only through PYTHONPATH, as a package installed with `pip install --target`: the child reads it, and
    # the search path of this process, where the check before the child runs looks, gets it too.
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


def test_a_new_task_is_tokenized_without_the_modules_of_the_current_folder(tmp_path, monkeypatch):
    # The current folder can be the bundle's, which may hold a file named like a module the child imports.
    spec = write_task_files(tmp_path, torch.randn(len(WORDS), 4), {"pick the cube": PICK})
    folder = tmp_path / "current"
    folder.mkdir()
    (folder / "tokenizers.py").write_text("raise SystemExit('tokenizers.py of the current folder ran')\n")
    monkeypatch.chdir(folder)
    # An empty PYTHONPATH entry, as `PYTHONPATH=$PYTHONPATH:/app` leaves when it was not set, names it too.
    monkeypatch.setenv("PYTHONPATH", os.pathsep + os.environ.get("PYTHONPATH", ""))
    assert TaskEncoder(tmp_path, spec).ids("place the box") == PLACE


@pytest.mark.parametrize("entry", [".", "sub"])
def test_a_new_task_is_tokenized_without_the_modules_of_the_bundle_folder(tmp_path, monkeypatch, entry):
    """Also when Python started elsewhere, with the bundle folder, or one in it, on PYTHONPATH."""
    bundle = tmp_path / "bundle"
    (bundle / "sub").mkdir(parents=True)
    spec = write_task_files(bundle, torch.randn(len(WORDS), 4), {"pick the cube": PICK})
    for folder in (bundle, bundle / "sub"):
        (folder / "tokenizers.py").write_text("raise SystemExit('tokenizers.py of the bundle ran')\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(bundle / entry), os.environ.get("PYTHONPATH", "")]))
    assert TaskEncoder(bundle, spec).ids("place the box") == PLACE


@pytest.mark.parametrize(
    "known", [{"pick the cube": [0, 2]}, {"pick the cube": PICK, "place the box": [0, 3]}]
)
def test_a_tokenizer_that_splits_any_known_task_differently_is_refused(tmp_path, known):
    tokens = TaskEncoder(tmp_path, write_task_files(tmp_path, torch.randn(len(WORDS), 4), known))
    with pytest.raises(ValueError, match="splits the tasks known at export into other tokens"):
        tokens.ids("pick it")


def test_task_tokens_without_a_known_task_or_with_a_table_of_another_dtype_are_refused(tmp_path):
    spec = write_task_files(tmp_path, torch.randn(len(WORDS), 4), {})
    with pytest.raises(ValueError, match="list no known task"):
        TaskEncoder(tmp_path, spec)
    spec = write_task_files(tmp_path, torch.randn(len(WORDS), 4).half(), {"pick the cube": PICK})
    with pytest.raises(ValueError, match="holds F16 embeddings; expected BF16 or F32"):
        TaskEncoder(tmp_path, spec)


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
    """Actions that depend on the task: the state times the sum of the task's rows, plus its token count. Each
    chunk holds two of them."""

    def forward(self, state: Tensor, embeddings: Tensor, length: Tensor) -> Tensor:
        return (state * embeddings.sum() + length.float())[None].expand(2, -1)


def expected(state: np.ndarray, table: Tensor, ids: list[int]) -> np.ndarray:
    """The action TaskScale computes for the task of the token `ids`."""
    return state * table[ids].sum().item() + len(ids)


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
    case = load_file(bundle.path / TEST_CASE)
    del case[TASK_EMBEDDINGS], case[TASK_LENGTH]
    save_numpy(case, bundle.path / TEST_CASE)
    (bundle.path / MANIFEST).write_text(json.dumps(data))
    return bundle.path, table


def test_engine_runs_the_task_it_is_given(tmp_path):
    path, table = task_bundle(tmp_path)
    state = np.array([1.0, 2.0], np.float32)
    pick, place = (expected(state, table, ids) for ids in (PICK, PLACE))
    np.testing.assert_allclose(
        BundleEngine(Bundle.read(path)).get_action({OBS_STATE: state}), pick, rtol=1e-5
    )
    engine = BundleEngine(Bundle.read(path), task="place_the box")
    np.testing.assert_allclose(engine.get_action({OBS_STATE: state}), place, rtol=1e-5)
    with pytest.raises(ValueError, match="tokens long"):
        BundleEngine(Bundle.read(path), task="pick the cube and then place it")


def test_engine_refuses_an_exported_task_longer_than_the_programs_take(tmp_path):
    path, _ = task_bundle(tmp_path)
    data = json.loads((path / MANIFEST).read_text())
    data["task_tokens"]["max_tokens"] = 4
    (path / MANIFEST).write_text(json.dumps(data))
    with pytest.raises(ValueError, match="'pick the cube' is 5 tokens long"):
        BundleEngine(Bundle.read(path), task="pick it")


def test_engine_refuses_a_bundle_for_another_gpu_before_it_tokenizes_a_task(tmp_path, monkeypatch):
    path, _ = task_bundle(tmp_path)
    data = json.loads((path / MANIFEST).read_text())
    data["target"] = {"device": "cuda", "gpu": "Orin", "compute_capability": "8.7"}
    (path / MANIFEST).write_text(json.dumps(data))
    monkeypatch.setattr(target_module, "cuda_gpus", lambda: [("H100", "9.0")])
    monkeypatch.setattr(task_tokens_module, "tokenize", lambda *args: pytest.fail("tokenized"))
    with pytest.raises(RuntimeError, match="built for Orin"):
        BundleEngine(Bundle.read(path), task="pick it")


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
    assert Counter(placed[:loaded]) == Counter([(2,), (6, 4), (1,), (6, 4), (1,)])
    assert placed[loaded:] == [(2,)] * 3
    # The exported task, which the test case ran, is not placed again.
    del placed[:]
    BundleEngine(Bundle.read(path))
    assert Counter(placed) == Counter([(2,), (6, 4), (1,)])
