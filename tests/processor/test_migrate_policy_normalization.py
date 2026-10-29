#!/usr/bin/env python

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

import json

import draccus
import pytest
import torch
from safetensors.torch import save_file

from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class
from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline, UnnormalizerProcessorStep
from lerobot.processor.migrate_policy_normalization import main as migrate
from lerobot.utils.import_utils import _diffusers_available

COMMON_LEGACY_FIELDS = {
    "device": "cpu",
    "input_features": {
        "observation.image": {"type": "VISUAL", "shape": [3, 32, 32]},
        "observation.state": {"type": "STATE", "shape": [2]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [2]}},
    "pretrained_backbone_weights": None,
    "use_group_norm": True,
    "crop_shape": [28, 28],
    "optimizer_betas": [0.95, 0.999],
}

# Small configs in the format of the published legacy checkpoints (lerobot/diffusion_pusht, lerobot/vqbet_pusht).
LEGACY_CONFIGS = {
    "diffusion": {
        "type": "diffusion",
        **COMMON_LEGACY_FIELDS,
        "normalization_mapping": {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"},
        "down_dims": [16, 32],
    },
    "vqbet": {
        "type": "vqbet",
        **COMMON_LEGACY_FIELDS,
        "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"},
        "gpt_n_layer": 1,
        "mlp_hidden_dim": 1024,  # removed from VQBeTConfig since these checkpoints were saved
    },
}

LEGACY_STATS = {
    "observation.image": {"mean": torch.rand(3, 1, 1), "std": torch.rand(3, 1, 1) + 0.1},
    "observation.state": {"min": -torch.rand(2), "max": torch.rand(2)},
    "action": {"min": -torch.rand(2), "max": torch.rand(2)},
}


@pytest.mark.parametrize(
    "policy_type",
    [
        pytest.param(
            "diffusion", marks=pytest.mark.skipif(not _diffusers_available, reason="diffusers not installed")
        ),
        "vqbet",
    ],
)
def test_migrates_legacy_checkpoint(policy_type, tmp_path, monkeypatch):
    legacy_config = LEGACY_CONFIGS[policy_type]
    config_class = PreTrainedConfig.get_choice_class(policy_type)
    config_dict = {k: v for k, v in legacy_config.items() if k not in ("type", "mlp_hidden_dim")}
    weights = get_policy_class(policy_type)(draccus.decode(config_class, config_dict)).state_dict()

    stats = {k: v for k, v in LEGACY_STATS.items() if policy_type == "diffusion" or k != "observation.image"}
    buffers = {}
    for feature, feature_stats in stats.items():
        modules = (
            ["normalize_targets", "unnormalize_outputs"] if feature == "action" else ["normalize_inputs"]
        )
        for module in modules:
            for stat, value in feature_stats.items():
                buffers[f"{module}.buffer_{feature.replace('.', '_')}.{stat}"] = value.clone()

    legacy_dir, output_dir = tmp_path / "legacy", tmp_path / "migrated"
    legacy_dir.mkdir()
    (legacy_dir / "config.json").write_text(json.dumps(legacy_config))
    save_file(weights | buffers, legacy_dir / "model.safetensors")

    monkeypatch.setattr(
        "sys.argv", ["migrate", "--pretrained-path", str(legacy_dir), "--output-dir", str(output_dir)]
    )
    monkeypatch.setattr("lerobot.common.train_utils.ModelCard.validate", lambda _self: None)
    migrate()

    policy = get_policy_class(policy_type).from_pretrained(output_dir, strict=True)
    migrated_weights = policy.state_dict()
    assert migrated_weights.keys() == weights.keys()
    assert all(torch.equal(migrated_weights[k], weights[k]) for k in weights)

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        output_dir, config_filename="policy_preprocessor.json"
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        output_dir, config_filename="policy_postprocessor.json"
    )
    normalizer = next(s for s in preprocessor.steps if isinstance(s, NormalizerProcessorStep))
    unnormalizer = next(s for s in postprocessor.steps if isinstance(s, UnnormalizerProcessorStep))
    for step, expected in ((normalizer, stats), (unnormalizer, {"action": stats["action"]})):
        assert step.stats.keys() >= expected.keys()
        for feature, feature_stats in expected.items():
            for stat, value in feature_stats.items():
                assert torch.equal(torch.as_tensor(step.stats[feature][stat]), value)
