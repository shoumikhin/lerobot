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

"""pi0.5's task tokens, end to end on a tiny pi0.5, with the test backend."""

import filecmp
import json

import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from lerobot.bundle.engine import BundleEngine  # noqa: E402
from lerobot.bundle.manifest import Bundle  # noqa: E402
from lerobot.scripts.lerobot_export import export  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from tests.export.plugins import use_plugins  # noqa: E402
from tests.export.tiny_pi05 import IMAGE, save_checkpoint  # noqa: E402

TASK = "pick up the cube"
OTHER_TASK = "open_the drawer\n"


@pytest.fixture(scope="module", autouse=True)
def tiny_pi05_plugins():
    """Add the test backend and make pi0.5 tiny in the export's steps, which run in processes of their own."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        use_plugins(monkeypatch, "torch_backend", "tiny_pi05")
        yield


def run_export(checkpoint, output_dir, *args: str) -> Bundle:
    export(["--backend.type=torch_test", f"--policy.path={checkpoint}", f"--output_dir={output_dir}", *args])
    return Bundle.read(output_dir)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pi05")
    with pytest.MonkeyPatch.context() as patch:
        save_checkpoint(folder / "checkpoint", patch)
    return folder / "checkpoint"


def test_task_budget_follows_max_task_tokens(checkpoint, tmp_path):
    bundle = run_export(checkpoint, tmp_path / "bundle", f"--task={TASK}", "--max_task_tokens=12")
    tokens = bundle.manifest.task_tokens
    assert tokens.max_tokens == 12
    # The checkpoint's own tokenizer file, byte for byte.
    assert filecmp.cmp(
        bundle.path / tokens.tokenizer, checkpoint.parent / "tokenizer" / "tokenizer.json", False
    )
    with pytest.raises(ValueError, match="takes at most 12"):
        BundleEngine(bundle, task="pick up the cube pick up the cube pick up the cube")


@pytest.mark.dataset
def test_export_knows_the_dataset_tasks_and_exports_the_first(checkpoint, tmp_path):
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets.io_utils import write_info

    features = {
        IMAGE: {"dtype": "image", "shape": (28, 28, 3), "names": ["height", "width", "channels"]},
        OBS_STATE: {"dtype": "float32", "shape": (3,), "names": None},
        ACTION: {"dtype": "float32", "shape": (3,), "names": None},
    }
    root = tmp_path / "dataset"
    metadata = LeRobotDatasetMetadata.create(
        "user/tiny", fps=10, features=features, root=root, use_videos=False
    )
    metadata.save_episode_tasks([OTHER_TASK, TASK])
    # Recorded episodes count their tasks this way; this dataset holds tasks only.
    metadata.info.total_tasks = 2
    write_info(metadata.info, root)
    bundle = run_export(
        checkpoint, tmp_path / "bundle", "--dataset.repo_id=user/tiny", f"--dataset.root={root}"
    )
    assert bundle.manifest.task == OTHER_TASK
    known = json.loads((bundle.path / bundle.manifest.task_tokens.known_tasks).read_text())
    assert list(known) == [OTHER_TASK, TASK]
    BundleEngine(bundle)


@pytest.mark.parametrize(
    "args, error",
    [
        ([], "give the one to export with --task"),
        ([f"--task={TASK}", "--max_task_tokens=3"], "more than the 3 of --max_task_tokens"),
        ([f"--task={TASK}", "--max_task_tokens=100"], "at most"),
    ],
)
def test_export_refuses_a_task_the_prompt_cannot_hold(checkpoint, tmp_path, capfd, args, error):
    with pytest.raises(RuntimeError, match="trace step failed"):
        run_export(checkpoint, tmp_path / "bundle", *args)
    assert error in capfd.readouterr().err
