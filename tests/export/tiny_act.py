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

"""A tiny ACT with one camera, which exports in seconds."""

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

IMAGE = f"{OBS_IMAGES}.front"


def act_config(image: str = IMAGE) -> ACTConfig:
    return ACTConfig(
        input_features={
            image: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
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


def act_stats(image: str = IMAGE, action_std: tuple[float, ...] = (30.0, 30.0, 30.0)) -> dict:
    return {
        image: {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.2)},
        OBS_STATE: {"mean": torch.zeros(3), "std": torch.ones(3)},
        ACTION: {"mean": torch.full((3,), 10.0), "std": torch.tensor(action_std)},
    }


def save_checkpoint(path, image: str = IMAGE) -> None:
    """Save the tiny ACT with random weights and its processors to `path`."""
    config = act_config(image)
    ACTPolicy(config).save_pretrained(path)
    for processor in make_pre_post_processors(config, dataset_stats=act_stats(image)):
        processor.save_pretrained(path)
