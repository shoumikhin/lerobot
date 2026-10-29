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

"""The recipes that export a policy as a chain of programs, end to end on tiny checkpoints with the test
backend: for a new frame and each task, the bundle plays the chunk that the PyTorch policy predicts from the
same noise."""

import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")

from lerobot.bundle.engine import BundleEngine  # noqa: E402
from lerobot.bundle.manifest import Bundle  # noqa: E402
from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.export.recipe import random_frame  # noqa: E402
from lerobot.policies import make_pre_post_processors, prepare_observation_for_inference  # noqa: E402
from lerobot.policies.factory import get_policy_class  # noqa: E402
from lerobot.scripts.lerobot_export import export  # noqa: E402
from tests.export import tiny_pi05  # noqa: E402
from tests.export.plugins import use_plugins  # noqa: E402

TASK = "pick up the cube"


@dataclass
class Chain:
    """A tiny checkpoint that `save` writes into a folder, and how its export runs and plays.

    `plugins` make the policy tiny in the export's own processes, as `tiny` does in this one. The bundle is
    exported with the first of `tasks` and plays each of them. `programs` are its program files, and `lora`
    the modules of a PEFT adapter for it.
    """

    save: Callable[[Path, pytest.MonkeyPatch], Path]
    tasks: tuple[str, ...]
    programs: list[str]
    plugins: tuple[str, ...] = ()
    tiny: Callable[[pytest.MonkeyPatch], None] | None = None
    lora: list[str] | None = None


def save_pi05(relative_actions: bool) -> Callable[[Path, pytest.MonkeyPatch], Path]:
    def save(folder: Path, patch: pytest.MonkeyPatch) -> Path:
        tiny_pi05.save_checkpoint(folder / "checkpoint", patch, relative_actions=relative_actions)
        return folder / "checkpoint"

    return save


PI05 = Chain(
    save_pi05(relative_actions=False),
    tasks=(TASK, "open_the drawer\n"),
    programs=["actions", "embed", "language_model_0", "language_model_1", "step"],
    plugins=("tiny_pi05",),
    tiny=tiny_pi05.use_tiny_pi05,
    lora=["action_out_proj"],
)
CHAINS = {
    "pi05_absolute": PI05,
    "pi05_relative": replace(PI05, save=save_pi05(relative_actions=True), lora=None),
}


def run_export(chain: Chain, checkpoint: Path, output_dir: Path, patch: pytest.MonkeyPatch) -> None:
    use_plugins(patch, "torch_backend", *chain.plugins)
    args = [f"--policy.path={checkpoint}", f"--output_dir={output_dir}", f"--task={chain.tasks[0]}"]
    export(["--backend.type=torch_test", *args])


@pytest.fixture(scope="module", params=list(CHAINS))
def exported(request, tmp_path_factory):
    chain, folder = CHAINS[request.param], tmp_path_factory.mktemp(request.param)
    with pytest.MonkeyPatch.context() as patch:
        checkpoint = chain.save(folder, patch)
        run_export(chain, checkpoint, folder / "bundle", patch)
    return chain, checkpoint, Bundle.read(folder / "bundle")


def test_the_bundle_plays_the_chunk_the_policy_predicts(exported, monkeypatch):
    """Another frame than the export's test case, and each task, against the policy's own chunk under the
    bundle's noise, as lerobot-rollout runs it."""
    chain, checkpoint, bundle = exported
    if chain.tiny is not None:
        chain.tiny(monkeypatch)
    policy = get_policy_class(PreTrainedConfig.from_pretrained(checkpoint).type).from_pretrained(checkpoint)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    frame = random_frame(policy.config.input_features, torch.Generator().manual_seed(1))
    steps = policy.config.n_action_steps

    def predict(task: str, noise: np.ndarray) -> np.ndarray:
        observation = prepare_observation_for_inference(dict(frame), torch.device("cpu"), task)
        with torch.no_grad():
            chunk = policy.eval().predict_action_chunk(
                preprocessor(observation), noise=torch.from_numpy(noise)[None]
            )
            return postprocessor(chunk[:, :steps])[0].numpy()

    scale = np.asarray(bundle.manifest.action_scale)
    for seed, task in enumerate(chain.tasks):
        engine = BundleEngine(bundle, task=task)
        np.random.seed(seed)
        actions = np.stack([engine.get_action(dict(frame)) for _ in range(steps)])
        noise = np.random.RandomState(seed).standard_normal(bundle.manifest.noise_shape).astype(np.float32)
        expected = predict(task, noise)
        np.testing.assert_allclose(actions / scale, expected / scale, rtol=0, atol=bundle.manifest.tolerance)
    if len(chain.tasks) > 1:
        # The task changes the actions by more than the tolerance, so the comparison would catch a program that
        # ignored it.
        assert np.abs((expected - predict(chain.tasks[0], noise)) / scale).max() > bundle.manifest.tolerance
    assert sorted(bundle.manifest.programs) == chain.programs


@pytest.mark.parametrize("name", [name for name, chain in CHAINS.items() if chain.lora])
def test_the_export_refuses_a_peft_adapter(name, tmp_path, monkeypatch, capfd):
    peft = pytest.importorskip("peft")
    chain = CHAINS[name]
    checkpoint, adapter = chain.save(tmp_path, monkeypatch), tmp_path / "adapter"
    # What PEFT training saves: the config and processors, and an adapter in place of model.safetensors.
    shutil.copytree(checkpoint, adapter, ignore=shutil.ignore_patterns("model.safetensors"))
    peft.LoraConfig(target_modules=chain.lora, base_model_name_or_path=str(checkpoint)).save_pretrained(
        adapter
    )
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.use_peft = True
    config.save_pretrained(adapter)
    with pytest.raises(RuntimeError, match="trace step failed"):
        run_export(chain, adapter, tmp_path / "bundle", monkeypatch)
    assert "is a PEFT adapter" in capfd.readouterr().err
