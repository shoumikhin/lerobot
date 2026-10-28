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

"""`lerobot-export` and the bundle it writes, end to end on a tiny ACT, with the test backend."""

import sys

import numpy as np
import pytest
import torch

import tests.export.torch_backend  # noqa: F401  # registers the test backend
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.inference.export import ExportBundle, ExportInferenceEngine
from lerobot.inference.sync import SyncInferenceEngine
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.scripts.lerobot_export import export
from lerobot.utils.constants import ACTION, OBS_STATE

IMAGE = "observation.images.front"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    config = ACTConfig(
        input_features={
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        chunk_size=4,
        n_action_steps=4,
        use_vae=False,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        pretrained_backbone_weights=None,
        device="cpu",
    )
    stats = {
        IMAGE: {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.2)},
        OBS_STATE: {"mean": torch.zeros(3), "std": torch.ones(3)},
        ACTION: {"mean": torch.full((3,), 10.0), "std": torch.full((3,), 30.0)},
    }
    path = tmp_path_factory.mktemp("act")
    ACTPolicy(config).save_pretrained(path)
    for processor in make_pre_post_processors(config, dataset_stats=stats):
        processor.save_pretrained(path)
    return path


def run_export(*args: str) -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(sys, "argv", ["lerobot-export", "--backend.type=torch_test", *args])
        export()


@pytest.fixture(scope="module")
def bundle(checkpoint, tmp_path_factory):
    path = tmp_path_factory.mktemp("export") / "act_torch_test"
    run_export(f"--policy.path={checkpoint}", f"--output_dir={path}")
    return ExportBundle.read(path)


def test_bundle_plays_the_actions_of_the_pytorch_policy(checkpoint, bundle):
    manifest = bundle.manifest
    assert (manifest.backend, manifest.runtime, manifest.task, manifest.noise_shape) == (
        "torch_test",
        "torch_test",
        None,
        None,
    )
    prepared = [f"prepared.{IMAGE}", f"prepared.{OBS_STATE}"]
    assert [(p.inputs, p.outputs) for p in manifest.programs] == [
        ([IMAGE, OBS_STATE], prepared),
        (prepared, [ACTION]),
    ]

    policy = ACTPolicy.from_pretrained(checkpoint)
    reference = SyncInferenceEngine(
        policy, *make_pre_post_processors(policy.config, checkpoint), {}, [], "", "cpu", ""
    )
    engine = ExportInferenceEngine(bundle)
    rng = np.random.default_rng(0)
    frame = {
        IMAGE: rng.integers(0, 256, (32, 32, 3), dtype=np.uint8),
        OBS_STATE: rng.random(3, dtype=np.float32),
    }
    for _ in range(policy.config.n_action_steps + 1):
        np.testing.assert_allclose(
            engine.get_action(dict(frame)), reference.get_action(dict(frame)).numpy(), atol=1e-5
        )


def test_export_writes_nothing_over_tolerance(checkpoint, tmp_path):
    output_dir = tmp_path / "bundle"
    with pytest.raises(RuntimeError, match="failed its check"):
        run_export(f"--policy.path={checkpoint}", f"--output_dir={output_dir}", "--backend.offset=1.0")
    assert list(tmp_path.iterdir()) == []
