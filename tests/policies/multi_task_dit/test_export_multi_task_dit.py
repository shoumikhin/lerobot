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

"""Multi-Task DiT's export recipe on a tiny policy whose CLIP encoders and tokenizer are made locally."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from transformers import CLIPTextModel  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig  # noqa: E402
from lerobot.policies.multi_task_dit.export_multi_task_dit import MultiTaskDiTExportRecipe  # noqa: E402
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (  # noqa: E402
    CLIPTextEncoder,
    MultiTaskDiTPolicy,
)
from lerobot.policies.multi_task_dit.processor_multi_task_dit import (  # noqa: E402
    make_multi_task_dit_pre_post_processors,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from tests.export.tiny_multi_task_dit import use_tiny_clip  # noqa: E402

IMAGE = "observation.images.front"


def make_recipe(monkeypatch, **sampler) -> MultiTaskDiTExportRecipe:
    use_tiny_clip(monkeypatch)
    config = MultiTaskDiTConfig(
        input_features={
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        n_obs_steps=2,
        horizon=8,
        n_action_steps=4,
        num_train_timesteps=10,
        num_integration_steps=3,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        timestep_embed_dim=16,
        image_crop_shape=None,
        tokenizer_max_length=8,
        device="cpu",
        **sampler,
    )
    stats = {
        IMAGE: {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.2)},
        OBS_STATE: {"min": torch.full((2,), -2.0), "max": torch.full((2,), 3.0)},
        ACTION: {"min": torch.full((2,), -2.0), "max": torch.full((2,), 3.0)},
    }
    torch.manual_seed(0)
    policy = MultiTaskDiTPolicy(config)
    # Untrained, its adaLN gates are zero, so the denoiser would ignore the frames, the task and the timestep.
    for name, parameter in policy.named_parameters():
        if "adaLN_modulation" in name:
            torch.nn.init.normal_(parameter, std=0.1)
    preprocessor, postprocessor = make_multi_task_dit_pre_post_processors(config, dataset_stats=stats)
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task="pick up the cube", tolerance=None
    )
    return MultiTaskDiTExportRecipe(policy.eval(), preprocessor, postprocessor, cfg)


@pytest.mark.parametrize("objective", ["diffusion", "flow_matching"])
def test_the_parts_compute_what_the_policy_samples(monkeypatch, objective):
    # The recipe checks that its parts, chained, compute the policy's chunk before it yields them.
    parts = list(make_recipe(monkeypatch, objective=objective).parts())
    assert [part.name for part in parts] == ["encoder", "step", "actions"]


def test_the_encoder_program_holds_the_task_features_and_the_policy_keeps_its_text_encoder(monkeypatch):
    recipe = make_recipe(monkeypatch, objective="flow_matching")
    encoder = next(part for part in recipe.parts() if part.name == "encoder").module
    assert not any(isinstance(module, CLIPTextModel) for module in encoder.modules())
    assert isinstance(recipe.policy.observation_encoder.text_encoder, CLIPTextEncoder)


def test_rk4_integration_is_refused(monkeypatch):
    recipe = make_recipe(monkeypatch, objective="flow_matching", integration_method="rk4")
    with pytest.raises(ValueError, match="Only Euler integration can be exported, not 'rk4'"):
        list(recipe.parts())
