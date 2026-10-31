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

"""The export of a tiny DM05 with its real Gemma3 core, where the program tests use a stand-in."""

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("torchvision")

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.dm05.modeling_dm05 import DM05Policy  # noqa: E402
from lerobot.policies.dm05.processor_dm05 import make_dm05_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from tests.policies.dm05.export_stand_in import CAMERA, save_processor  # noqa: E402
from tests.policies.dm05.test_dm05_integration import _dm05_config, _tiny_core_config  # noqa: E402

CHUNK = 2


def test_lerobot_export_writes_a_bundle_that_plays_the_policys_chunk(tmp_path, monkeypatch):
    """Transformers traces the core's Gemma3 attention mask to an op that a program save cannot write as it is,
    so only the real core shows that the bundle gets written."""
    from lerobot.bundle.engine import BundleEngine
    from lerobot.bundle.manifest import Bundle
    from lerobot.export.recipe import random_frame
    from lerobot.policies import prepare_observation_for_inference
    from lerobot.scripts.lerobot_export import export
    from tests.export.plugins import use_plugins

    save_processor(tmp_path / "processor")
    core = _tiny_core_config().to_dict()
    # The image tokens of the tiny processor's vocabulary.
    core["vlm_config"].update(boi_token_index=30, eoi_token_index=31, image_token_index=32)
    config = _dm05_config(
        core_config=core,
        processor_name_or_path=str(tmp_path / "processor"),
        dtype=torch.float32,
        vlm_gradient_checkpointing=False,
        ae_gradient_checkpointing=False,
    )
    config.max_state_dim = config.max_action_dim = 4
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(4,))}
    identity = NormalizationMode.IDENTITY
    config.normalization_mapping = {"VISUAL": identity, "STATE": identity, "ACTION": identity}
    torch.manual_seed(0)
    policy = DM05Policy(config).eval()
    preprocessor, postprocessor = make_dm05_pre_post_processors(config)
    checkpoint = tmp_path / "checkpoint"
    for obj in (policy, preprocessor, postprocessor):
        obj.save_pretrained(checkpoint)
    use_plugins(monkeypatch, "torch_backend")
    task = "pick up the cube"
    export(
        [
            "--backend.type=torch_test",
            f"--policy.path={checkpoint}",
            f"--output_dir={tmp_path / 'bundle'}",
            f"--task={task}",
        ]
    )

    bundle = Bundle.read(tmp_path / "bundle")
    frame = random_frame(policy.config.input_features, torch.Generator().manual_seed(1))
    engine = BundleEngine(bundle, task=task)
    np.random.seed(0)
    actions = np.stack([engine.get_action(dict(frame)) for _ in range(CHUNK)])
    noise = np.random.RandomState(0).standard_normal(bundle.manifest.noise_shape).astype(np.float32)
    observation = prepare_observation_for_inference(dict(frame), torch.device("cpu"), task)
    with torch.no_grad():
        chunk = policy.predict_action_chunk(preprocessor(observation), noise=torch.from_numpy(noise)[None])
        expected = postprocessor(chunk[:, :CHUNK])[0].numpy()
    scale = np.asarray(bundle.manifest.action_scale)
    np.testing.assert_allclose(actions / scale, expected / scale, rtol=0, atol=bundle.manifest.tolerance)
