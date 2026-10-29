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

"""SmolVLA exported as a chain of programs, end to end on a tiny SmolVLA, with the test backend."""

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file

pytest.importorskip("transformers")

import tests.export.torch_backend  # noqa: E402,F401  # registers the test backend
from lerobot.inference.bundle import Bundle, BundleInferenceEngine  # noqa: E402
from lerobot.inference.bundle.manifest import NOISE  # noqa: E402
from lerobot.policies import make_pre_post_processors, prepare_observation_for_inference  # noqa: E402
from lerobot.policies.smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.policies.smolvla.export_smolvla import (  # noqa: E402
    EMBED_MODULES,
    LANGUAGE_MODEL_MODULES,
    STEP_MODULES,
)
from lerobot.scripts.lerobot_export import export  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402
from tests.export.tiny_smolvla import CAMERAS, save_checkpoint, use_tiny_smolvla  # noqa: E402

# The export's steps run in processes of their own: these register the test backend and make SmolVLA tiny there.
IMPORTS = ["tests.export.torch_backend", "tests.export.tiny_smolvla_steps"]
TASK = "pick up the cube"


def run_export(checkpoint, output_dir, *args: str) -> Bundle:
    export(
        ["--backend.type=torch_test", f"--policy.path={checkpoint}", f"--output_dir={output_dir}", *args],
        IMPORTS,
    )
    return Bundle.read(output_dir)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    folder = tmp_path_factory.mktemp("smolvla")
    with pytest.MonkeyPatch.context() as patch:
        config = save_checkpoint(folder / "checkpoint", patch)
    bundle = run_export(folder / "checkpoint", folder / "bundle", f"--task={TASK}")
    return folder / "checkpoint", config, bundle


def load_policy(checkpoint, config, monkeypatch) -> SmolVLAPolicy:
    use_tiny_smolvla(monkeypatch)
    return SmolVLAPolicy.from_pretrained(checkpoint, config=config).eval()


def test_manifest_runs_the_step_once_per_timestep_from_one_program(exported):
    _, config, bundle = exported
    manifest = bundle.manifest
    steps = [f"step_{i}" for i in range(config.num_steps)]
    assert [p.name for p in manifest.programs] == ["embed", "language_model", *steps, "actions"]
    files = {p.file for p in manifest.programs}
    assert files == {"embed.pt2", "language_model.pt2", "step.pt2", "actions.pt2"}
    embed, language_model, first, *_, last, actions = manifest.programs
    assert embed.inputs == [*CAMERAS, OBS_STATE] and language_model.inputs == embed.outputs
    assert language_model.outputs == ["past_keys", "past_values"]
    assert first.inputs == ["prefix_pad_masks", "past_keys", "past_values", NOISE, "timestep_0"]
    assert last.outputs == ["x_10"] and actions.inputs == ["x_10"]
    constants = load_file(bundle.path / manifest.constants)
    np.testing.assert_allclose(
        [constants[f"timestep_{i}"][0] for i in range(10)], [1.0 - i / 10 for i in range(10)], rtol=1e-6
    )
    assert manifest.task == TASK and manifest.noise_shape == [config.chunk_size, config.max_action_dim]


def test_each_program_holds_only_the_weights_it_runs(exported, monkeypatch):
    checkpoint, config, bundle = exported
    policy = load_policy(checkpoint, config, monkeypatch)

    def size(modules) -> int:
        return sum(p.numel() for name in modules for p in policy.get_submodule(name).parameters())

    for file, own in (
        ("embed.pt2", size(EMBED_MODULES)),
        ("language_model.pt2", size(LANGUAGE_MODEL_MODULES)),
        ("step.pt2", size(STEP_MODULES)),
        ("actions.pt2", 0),
    ):
        program = torch.export.load(bundle.path / file)
        weights = sum(t.numel() for t in program.state_dict.values())
        constants = sum(t.numel() for t in program.constants.values() if isinstance(t, torch.Tensor))
        # Besides its weights a program holds small tables, like the normalization statistics; any other
        # module of the policy would add at least a few hundred values.
        assert own <= weights + constants < own + 100, file


def test_bundle_plays_the_actions_of_the_pytorch_policy(exported, monkeypatch):
    checkpoint, config, bundle = exported
    engine = BundleInferenceEngine(bundle)
    rng = np.random.default_rng(1)
    frame = {name: rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for name in CAMERAS}
    frame[OBS_STATE] = np.array([0.3, -0.7, 0.9], np.float32)
    np.random.seed(0)
    actions = np.stack([engine.get_action(dict(frame)) for _ in range(config.n_action_steps)])
    noise = np.random.RandomState(0).standard_normal(bundle.manifest.noise_shape).astype(np.float32)

    policy = load_policy(checkpoint, config, monkeypatch)
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(checkpoint))
    with torch.no_grad():
        batch = preprocessor(prepare_observation_for_inference(dict(frame), torch.device("cpu"), TASK))
        expected = policy.predict_action_chunk(batch, noise=torch.from_numpy(noise)[None])
        expected = postprocessor(expected[:, : config.n_action_steps])[0].numpy()
    np.testing.assert_allclose(actions, expected, rtol=0, atol=1e-4)
