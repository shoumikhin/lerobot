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

"""pi0.5 exported as a chain of programs, end to end on a tiny pi0.5, with the test backend."""

import filecmp
import json

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

import tests.export.torch_backend  # noqa: E402,F401  # registers the test backend
from lerobot.inference.bundle import Bundle, BundleInferenceEngine  # noqa: E402
from lerobot.inference.bundle.manifest import NOISE  # noqa: E402
from lerobot.policies import make_pre_post_processors, prepare_observation_for_inference  # noqa: E402
from lerobot.policies.pi05 import PI05Policy  # noqa: E402
from lerobot.scripts.lerobot_export import export  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from tests.export.tiny_pi05 import IMAGE, save_checkpoint, use_tiny_pi05  # noqa: E402

# The export's steps run in processes of their own: these register the test backend and make pi0.5 tiny there.
IMPORTS = ["tests.export.torch_backend", "tests.export.tiny_pi05_steps"]
TASK = "pick up the cube"
OTHER_TASK = "open_the drawer\n"


def run_export(checkpoint, output_dir, *args: str) -> Bundle:
    export(
        ["--backend.type=torch_test", f"--policy.path={checkpoint}", f"--output_dir={output_dir}", *args],
        IMPORTS,
    )
    return Bundle.read(output_dir)


@pytest.fixture(scope="module", params=[False, True], ids=["absolute", "relative"])
def exported(request, tmp_path_factory):
    folder = tmp_path_factory.mktemp("pi05")
    with pytest.MonkeyPatch.context() as patch:
        config = save_checkpoint(folder / "checkpoint", patch, relative_actions=request.param)
    bundle = run_export(folder / "checkpoint", folder / "bundle", f"--task={TASK}")
    return folder / "checkpoint", config, bundle


def policy_actions(checkpoint, config, frame, task, noise, monkeypatch):
    """The actions PyTorch's pi0.5 plays for one chunk, as lerobot-rollout runs it."""
    use_tiny_pi05(monkeypatch)
    policy = PI05Policy.from_pretrained(checkpoint, config=config).eval()
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(checkpoint))
    with torch.no_grad():
        batch = preprocessor(prepare_observation_for_inference(dict(frame), torch.device("cpu"), task))
        actions = policy.predict_action_chunk(batch, noise=torch.from_numpy(noise)[None])
        return postprocessor(actions[:, : config.n_action_steps])[0].numpy()


def bundle_actions(engine, bundle, frame, steps, seed):
    """The actions the bundle plays for one chunk, and the starting noise it drew."""
    np.random.seed(seed)
    actions = np.stack([engine.get_action(dict(frame)) for _ in range(steps)])
    noise = np.random.RandomState(seed).standard_normal(bundle.manifest.noise_shape).astype(np.float32)
    return actions, noise


def test_manifest_runs_the_step_ten_times_from_one_program(exported):
    checkpoint, config, bundle = exported
    manifest = bundle.manifest
    names = [p.name for p in manifest.programs]
    steps = [f"step_{i}" for i in range(10)]
    assert names == ["embed", "language_model_0", "language_model_1", *steps, "actions"]
    assert {p.file for p in manifest.programs if p.name in steps} == {"step.pt2"}
    assert len({p.file for p in manifest.programs}) == 5
    assert manifest.programs[0].inputs == [IMAGE, OBS_STATE, "task_embeddings", "task_length"]
    assert manifest.programs[-2].outputs == ["x_10"] and manifest.programs[-1].inputs[0] == "x_10"
    assert manifest.programs[3].inputs[-2:] == [NOISE, "timestep_0"]
    constants = load_file(bundle.path / manifest.constants)
    np.testing.assert_allclose(
        [constants[f"timestep_{i}"][0] for i in range(10)], [1.0 - i / 10 for i in range(10)], rtol=1e-6
    )
    tokens = manifest.task_tokens
    assert tokens.text[-1] == ["template", "Task: {task}, State:"]
    # The checkpoint's own tokenizer file, byte for byte.
    assert filecmp.cmp(
        bundle.path / tokens.tokenizer, checkpoint.parent / "tokenizer" / "tokenizer.json", False
    )
    assert manifest.task == TASK and manifest.noise_shape == [config.chunk_size, config.max_action_dim]


def test_bundle_plays_the_actions_of_the_pytorch_policy_for_any_task(exported, monkeypatch):
    checkpoint, config, bundle = exported
    engine = BundleInferenceEngine(bundle)
    frame = {
        name: value
        for name, value in load_file(bundle.path / bundle.manifest.test).items()
        if name in (IMAGE, OBS_STATE)
    }
    frame[OBS_STATE] = np.array([0.3, -0.7, 0.9], np.float32)
    actions, noise = bundle_actions(engine, bundle, frame, config.n_action_steps, 0)
    expected = policy_actions(checkpoint, config, frame, TASK, noise, monkeypatch)
    np.testing.assert_allclose(actions, expected, rtol=0, atol=1e-4)

    assert engine.set_task(OTHER_TASK)
    engine.reset()
    actions, noise = bundle_actions(engine, bundle, frame, config.n_action_steps, 1)
    expected = policy_actions(checkpoint, config, frame, OTHER_TASK, noise, monkeypatch)
    np.testing.assert_allclose(actions, expected, rtol=0, atol=1e-4)
    # The task changes the actions, so the comparison above would catch a program that ignored it.
    assert not np.allclose(
        expected, policy_actions(checkpoint, config, frame, TASK, noise, monkeypatch), atol=1e-3
    )


def test_task_budget_follows_max_task_tokens(exported, tmp_path):
    checkpoint, _, _ = exported
    bundle = run_export(checkpoint, tmp_path / "bundle", f"--task={TASK}", "--max_task_tokens=12")
    assert bundle.manifest.task_tokens.max_tokens == 12
    engine = BundleInferenceEngine(bundle)
    assert not engine.set_task("pick up the cube pick up the cube pick up the cube")


@pytest.mark.dataset
def test_export_knows_the_dataset_tasks_and_exports_the_first(exported, tmp_path):
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets.io_utils import write_info

    checkpoint, _, _ = exported
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
    known = json.loads((bundle.path / bundle.manifest.task_tokens.known).read_text())
    assert list(known) == [OTHER_TASK, TASK]
    BundleInferenceEngine(bundle)


@pytest.mark.parametrize(
    "args, error",
    [
        ([], "give the one to export with --task"),
        ([f"--task={TASK}", "--max_task_tokens=3"], "more than the 3 of --max_task_tokens"),
        ([f"--task={TASK}", "--max_task_tokens=100"], "at most"),
    ],
)
def test_export_refuses_a_task_the_prompt_cannot_hold(exported, tmp_path, capfd, args, error):
    checkpoint, _, _ = exported
    with pytest.raises(RuntimeError, match="trace step failed"):
        run_export(checkpoint, tmp_path / "bundle", *args)
    assert error in capfd.readouterr().err
