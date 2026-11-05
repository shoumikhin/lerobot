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

"""The Cosmos3 export recipe on a tiny random Cosmos3 Edge with its own word-level tokenizer, with the test backend."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("diffusers")
pytest.importorskip("tokenizers")

from diffusers import UniPCMultistepScheduler  # noqa: E402

from lerobot.bundle.engine import BundleEngine  # noqa: E402
from lerobot.bundle.manifest import NOISE, Bundle  # noqa: E402
from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.export.recipe import load_policy  # noqa: E402
from lerobot.policies.cosmos3.export_cosmos3 import (  # noqa: E402
    Cosmos3ExportRecipe,
    domain_projection,
    solver_coefficients,
)
from lerobot.policies.cosmos3.modeling_cosmos3 import DomainAwareLinear  # noqa: E402
from lerobot.scripts.lerobot_export import export  # noqa: E402
from tests.export.plugins import use_plugins  # noqa: E402
from tests.export.tiny_cosmos3 import save_checkpoint  # noqa: E402

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
    folder = tmp_path_factory.mktemp("cosmos3") / "checkpoint"
    save_checkpoint(folder, steps=STEPS)
    return folder


def recipe_config(checkpoint):
    """What `lerobot-export` hands the recipe, for a recipe built in this process."""
    policy = PreTrainedConfig.from_pretrained(checkpoint)
    policy.pretrained_path = checkpoint
    return SimpleNamespace(
        policy=policy, backend=SimpleNamespace(device="cpu"), task=TASK, dataset=None, tolerance=None, seed=0
    )


def unipc(order: int = 2, **options) -> UniPCMultistepScheduler:
    """A flow UniPC scheduler of 6 steps, set up the way Cosmos3 sets up its own."""
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=1.0, solver_order=order, **options
    )
    sigmas = np.linspace(0.999, 0.0, 7)[:-1]
    scheduler.set_timesteps(sigmas=5 * sigmas / (1 + 4 * sigmas))
    return scheduler


def test_exports_an_encoder_a_step_per_timestep_and_the_actions(checkpoint, tmp_path):
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
    assert list(manifest.programs) == ["encode", "step", "actions"]
    assert [run.name for run in manifest.runs] == ["encode", *(f"step_{i}" for i in range(STEPS)), "actions"]
    assert manifest.runs[0].inputs[-1] == NOISE
    # The first step starts the solver's history from zeros, and each later one reads the one before.
    assert manifest.runs[1].inputs[1:-2] == ["zeros"] * 3
    assert manifest.runs[2].inputs[1:-2] == manifest.runs[1].outputs[1:]
    # Loading replays the test case through the compiled programs.
    BundleEngine(bundle)


def test_the_chain_plays_what_the_policy_plays(checkpoint):
    recipe = Cosmos3ExportRecipe.load(recipe_config(checkpoint))
    for _ in recipe.parts():
        pass
    # The test case is compared with the policy's own predict_action_chunk on the same frame and noise.
    case = recipe.test_case()
    assert case.actions.shape == (recipe.policy.config.n_action_steps, 10)


def test_the_held_domain_projection_matches_the_policy_for_that_domain():
    torch.manual_seed(0)
    layer = DomainAwareLinear(4, 3, num_domains=5)
    torch.nn.init.normal_(layer.bias.weight)
    x = torch.randn(1, 6, 4)
    torch.testing.assert_close(domain_projection(layer, x, 2), layer(x, torch.tensor([2])))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_the_measured_matrices_replay_the_scheduler_with_its_history(order):
    timesteps, matrices = solver_coefficients(lambda: unipc(order))
    scheduler = unipc(order)
    assert [float(t) for t in timesteps] == [float(t) for t in scheduler.timesteps]
    generator = torch.Generator().manual_seed(2)
    state = expected = torch.randn(1, 8, generator=generator, dtype=torch.float64)
    history = [torch.zeros_like(state)] * (order + 1)
    for timestep, matrix in zip(scheduler.timesteps, matrices, strict=True):
        velocity = torch.randn(state.shape, generator=generator, dtype=torch.float64)
        expected = scheduler.step(velocity, timestep, expected).prev_sample
        state, *history = matrix @ torch.cat([state, velocity, *history])
        state, history = state[None], [h[None] for h in history]
        torch.testing.assert_close(state, expected, rtol=1e-12, atol=1e-12)


def test_refuses_a_scheduler_whose_step_is_not_linear():
    # Thresholding clips the predicted sample, which no fixed matrix can do.
    with pytest.raises(ValueError, match="solver step is not linear in its inputs"):
        solver_coefficients(lambda: unipc(thresholding=True))


@pytest.mark.parametrize(("field", "value"), [("guidance_scale", 2.0), ("n_obs_steps", 2)])
def test_refuses_what_the_chain_does_not_run(checkpoint, field, value):
    cfg = recipe_config(checkpoint)
    policy, preprocessor, postprocessor = load_policy(cfg)
    setattr(policy.config, field, value)
    with pytest.raises(ValueError, match="Cosmos3 exports"):
        Cosmos3ExportRecipe(policy, preprocessor, postprocessor, cfg)
