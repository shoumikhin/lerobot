# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot in 2026.

"""The G0.5 export recipe on a tiny random G0.5 with its own word-level tokenizer, with the test backend."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from lerobot.bundle.engine import BundleEngine  # noqa: E402
from lerobot.bundle.manifest import NOISE, Bundle  # noqa: E402
from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.export.backends.executorch_tensorrt import ExecuTorchTensorRTConfig  # noqa: E402
from lerobot.export.recipe import load_policy  # noqa: E402
from lerobot.policies.g05.export_g05 import G05Actions, G05ExportRecipe  # noqa: E402
from lerobot.scripts.lerobot_export import export  # noqa: E402
from tests.export.plugins import use_plugins  # noqa: E402
from tests.export.tiny_g05 import save_checkpoint  # noqa: E402

TASK = "pick up the cube"
STEPS = 3


@pytest.fixture(scope="module", autouse=True)
def torch_test_backend():
    """Add the test backend to the export's steps, which run in processes of their own."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        use_plugins(monkeypatch, "torch_backend")
        yield


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    folder = tmp_path_factory.mktemp("g05") / "checkpoint"
    save_checkpoint(folder, steps=STEPS)
    return folder


def recipe_config(checkpoint):
    """What `lerobot-export` hands the recipe, for a recipe built in this process."""
    policy = PreTrainedConfig.from_pretrained(checkpoint)
    policy.pretrained_path = checkpoint
    return SimpleNamespace(
        policy=policy, backend=SimpleNamespace(device="cpu"), task=TASK, dataset=None, tolerance=None, seed=0
    )


def test_exports_a_prefix_a_step_per_timestep_and_the_actions(checkpoint, tmp_path):
    output = tmp_path / "bundle"
    export(
        [
            "--backend.type=torch_test",
            f"--policy.path={checkpoint}",
            f"--output_dir={output}",
            f"--task={TASK}",
        ]
    )
    bundle = Bundle.read(output)
    manifest = bundle.manifest
    assert list(manifest.programs) == ["prefix", "step", "actions"]
    assert [run.name for run in manifest.runs] == ["prefix", *(f"step_{i}" for i in range(STEPS)), "actions"]
    assert manifest.runs[1].inputs[-2:] == [NOISE, "timestep_0"]
    assert manifest.task == TASK
    # The tiny checkpoint names no actions, as the released ones do.
    assert manifest.action_names is None
    # Loading replays the test case through the compiled programs.
    BundleEngine(bundle)


def test_the_chain_plays_what_the_policy_plays(checkpoint):
    recipe = G05ExportRecipe.load(recipe_config(checkpoint))
    for part in recipe.parts():
        # --quantize changes the weights of each part in place once it is traced, which the parts share.
        for weight in part.module.parameters():
            weight.data.zero_()
    # The test case is compared with the policy's own predict_action_chunk on the same frame and noise.
    case = recipe.test_case()
    assert case.actions.shape == (recipe.policy.config.n_action_steps, 7)
    untouched = G05ExportRecipe.load(recipe_config(checkpoint))
    for _ in untouched.parts():
        pass
    np.testing.assert_array_equal(case.actions, untouched.test_case().actions)


@pytest.mark.parametrize("tensorrt", [True, False], ids=["tensorrt", "other_backend"])
def test_tensorrt_may_split_the_prefix_and_step_builds(checkpoint, tensorrt):
    """Only with TensorRT, since only `executorch_tensorrt` reads compile options."""
    recipe = G05ExportRecipe.load(recipe_config(checkpoint))
    if tensorrt:
        recipe.cfg.backend = ExecuTorchTensorRTConfig()
    options = {part.name: part.compile_options for part in recipe.parts()}
    expected = {"enable_resource_partitioning": True} if tensorrt else {}
    assert options["prefix"] == expected
    assert options["step"] == expected


def test_the_action_scale_holds_every_played_step(checkpoint):
    recipe = G05ExportRecipe.load(recipe_config(checkpoint))
    for _ in recipe.parts():
        pass
    case = recipe.test_case()
    # Each step unnormalizes with statistics of its own, while the checks divide every step by one scale.
    chunk = torch.zeros(1, recipe.policy.config.chunk_size, 20)
    spread = (recipe.postprocessor(chunk + 1) - recipe.postprocessor(chunk))[0, : len(case.actions)]
    np.testing.assert_allclose(case.action_scale, spread.abs().min(0).values.numpy(), rtol=1e-6)


def test_actions_unnormalize_each_step_with_its_own_statistics(checkpoint):
    recipe = G05ExportRecipe.load(recipe_config(checkpoint))
    chunk = torch.randn(recipe.policy.config.chunk_size, 20, generator=torch.Generator().manual_seed(0))
    played = G05Actions(recipe.postprocessor, recipe.policy.config.n_action_steps)(chunk)
    # lerobot-rollout unnormalizes the queued actions one by one, each with its step's statistics.
    recipe.postprocessor.reset()
    expected = torch.cat([recipe.postprocessor(action[None]) for action in chunk[: len(played)]])
    torch.testing.assert_close(played, expected)


@pytest.mark.parametrize(
    "field, value",
    [
        ("action_head", "actioncodec"),
        ("runtime_system", "system2"),
        ("n_obs_steps", 2),
        ("use_relative_actions", True),
    ],
)
def test_refuses_what_the_chain_does_not_run(checkpoint, field, value):
    cfg = recipe_config(checkpoint)
    policy, preprocessor, postprocessor = load_policy(cfg)
    setattr(policy.config, field, value)
    with pytest.raises(ValueError, match="G0.5 exports"):
        G05ExportRecipe(policy, preprocessor, postprocessor, cfg)
