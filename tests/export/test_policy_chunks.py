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

import importlib.util
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

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
from tests.export import tiny_groot, tiny_pi05, tiny_smolvla  # noqa: E402
from tests.export.plugins import use_plugins  # noqa: E402

TASK = "pick up the cube"
GROOT_EXTRAS = pytest.mark.skipif(
    not all(importlib.util.find_spec(name) for name in ("diffusers", "tree")),
    reason="GR00T needs diffusers and dm-tree, from lerobot[groot]",
)


@dataclass
class Chain:
    """A tiny checkpoint that `save` writes into a folder, and how its export runs and plays.

    `plugins` make the policy tiny in the export's own processes, as `tiny` does in this one. The bundle is
    exported with the first of `tasks` and `args`, and plays each of them. `programs` are its program names,
    `lora` the modules of a PEFT adapter for it, and `check` checks what else it holds.
    """

    save: Callable[[Path, pytest.MonkeyPatch], Path]
    tasks: tuple[str, ...]
    programs: list[str]
    args: tuple[str, ...] = ()
    plugins: tuple[str, ...] = ()
    tiny: Callable[[pytest.MonkeyPatch], None] | None = None
    lora: list[str] | None = None
    check: Callable[[torch.nn.Module, Bundle], None] | None = None
    marks: tuple[pytest.MarkDecorator, ...] = ()


def save_pi05(relative_actions: bool) -> Callable[[Path, pytest.MonkeyPatch], Path]:
    def save(folder: Path, patch: pytest.MonkeyPatch) -> Path:
        tiny_pi05.save_checkpoint(folder / "checkpoint", patch, relative_actions=relative_actions)
        return folder / "checkpoint"

    return save


def save_smolvla(folder: Path, patch: pytest.MonkeyPatch) -> Path:
    tiny_smolvla.save_checkpoint(folder / "checkpoint", patch)
    return folder / "checkpoint"


def programs_hold_only_their_weights(
    owners: dict[str, tuple[str, ...]], transformed: tuple[str, ...] = ()
) -> Callable[[torch.nn.Module, Bundle], None]:
    """A check that each program holds the parameters of the modules `owners` names for it and no other
    parameter of the policy, found by value among its weights and constants. A part boundary that moves, or
    a weight copied into a second program, fails it. `transformed` names parameters a program holds computed
    into another form, which no value matches."""

    def key(tensor: torch.Tensor) -> tuple:
        tensor = tensor.detach().contiguous()
        return tuple(tensor.shape), tensor.dtype, tensor.reshape(-1).view(torch.uint8).numpy().tobytes()

    def check(policy: torch.nn.Module, bundle: Bundle) -> None:
        parameters = {key(p) for p in policy.parameters()}
        for name, files in bundle.manifest.programs.items():
            program = torch.export.load(bundle.path / files.file)
            constants = [t for t in program.constants.values() if isinstance(t, torch.Tensor)]
            held = Counter(k for k in map(key, [*program.state_dict.values(), *constants]) if k in parameters)
            own = Counter(
                key(p)
                for module in owners.get(name, ())
                for parameter, p in policy.get_submodule(module).named_parameters()
                if not any(part in parameter for part in transformed)
            )
            assert held == own, name

    return check


ACTION_PROJECTIONS = ("model.action_in_proj", "model.action_out_proj")
VLM = "model.vlm_with_expert.vlm.model"
smolvla_programs_hold_only_their_weights = programs_hold_only_their_weights(
    {
        "embed": (
            f"{VLM}.vision_model",
            f"{VLM}.connector",
            f"{VLM}.text_model.embed_tokens",
            "model.state_proj",
        ),
        "language_model": (f"{VLM}.text_model.layers", f"{VLM}.text_model.norm"),
        "step": (
            "model.vlm_with_expert.lm_expert",
            *ACTION_PROJECTIONS,
            "model.action_time_mlp_in",
            "model.action_time_mlp_out",
        ),
    }
)


PI05 = Chain(
    save_pi05(relative_actions=False),
    tasks=(TASK, "open_the drawer\n"),
    programs=["actions", "embed", "language_model_0", "language_model_1", "step"],
    plugins=("tiny_pi05",),
    tiny=tiny_pi05.use_tiny_pi05,
    lora=["action_out_proj"],
)
GROOT = Chain(
    lambda folder, patch: tiny_groot.save_checkpoint(folder, "torchvision"),
    # A task the export never saw.
    tasks=(tiny_groot.TASK, "stop then pick up the cube"),
    programs=["actions", "keys_values", "language_model_0", "language_model_1", "step_0", "step_1", "vision"],
    # On this random tiny GR00T the chained parts and the whole policy differ by up to 3.7e-05 on a busy CPU,
    # as float32 sums run in another order, which is over the 2e-05 default for real checkpoints.
    args=("--tolerance=1e-4",),
    marks=(GROOT_EXTRAS,),
)
CHAINS = {
    "pi05_absolute": PI05,
    "pi05_relative": replace(PI05, save=save_pi05(relative_actions=True), lora=None),
    "groot_torchvision": GROOT,
    "smolvla": Chain(
        save_smolvla,
        tasks=(TASK,),
        programs=["actions", "embed", "language_model", "step"],
        plugins=("tiny_smolvla",),
        tiny=tiny_smolvla.use_tiny_smolvla,
        check=smolvla_programs_hold_only_their_weights,
    ),
}


def run_export(chain: Chain, checkpoint: Path, output_dir: Path, patch: pytest.MonkeyPatch) -> None:
    use_plugins(patch, "torch_backend", *chain.plugins)
    args = [
        f"--policy.path={checkpoint}",
        f"--output_dir={output_dir}",
        f"--task={chain.tasks[0]}",
        *chain.args,
    ]
    export(["--backend.type=torch_test", *args])


@pytest.fixture(
    scope="module", params=[pytest.param(name, marks=chain.marks) for name, chain in CHAINS.items()]
)
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
    if chain.check is not None:
        chain.check(policy, bundle)


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


def test_the_smolvla_recipe_refuses_a_chain_that_computes_something_else(tmp_path, monkeypatch):
    from lerobot.policies.smolvla.export_smolvla import SmolVLAExportRecipe, SmolVLAStep

    checkpoint = save_smolvla(tmp_path, monkeypatch)
    policy = get_policy_class("smolvla").from_pretrained(checkpoint).eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    recipe = SmolVLAExportRecipe(
        policy,
        preprocessor,
        postprocessor,
        SimpleNamespace(
            backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task=TASK, tolerance=None
        ),
    )
    step = SmolVLAStep.forward
    monkeypatch.setattr(SmolVLAStep, "forward", lambda self, *inputs: step(self, *inputs) + 1e-3)
    with pytest.raises(ValueError, match="computes something else as a chain of parts"):
        next(recipe.parts())
