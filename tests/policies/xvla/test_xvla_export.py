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

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.xvla.configuration_xvla import XVLAConfig  # noqa: E402
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_LANGUAGE_TOKENS, OBS_STATE  # noqa: E402

TINY_FLORENCE = {
    "model_type": "florence2",
    "bos_token_id": 0,
    "eos_token_id": 2,
    "pad_token_id": 1,
    "projection_dim": 64,
    "vocab_size": 51289,
    "text_config": {
        "vocab_size": 51289,
        "d_model": 64,
        "encoder_layers": 1,
        "decoder_layers": 1,
        "num_hidden_layers": 1,
        "encoder_attention_heads": 2,
        "decoder_attention_heads": 2,
        "encoder_ffn_dim": 128,
        "decoder_ffn_dim": 128,
        "max_position_embeddings": 64,
    },
    "vision_config": {
        "model_type": "davit",
        "drop_path_rate": 0.0,
        "patch_size": [7, 3, 3, 3],
        "patch_stride": [4, 2, 2, 2],
        "patch_padding": [3, 1, 1, 1],
        "patch_prenorm": [False, True, True, True],
        "dim_embed": [16, 32, 64, 128],
        "num_heads": [2, 2, 4, 4],
        "num_groups": [2, 2, 4, 4],
        "depths": [1, 1, 1, 1],
        "window_size": 2,
        "projection_dim": 64,
        "visual_temporal_embedding": {"type": "COSINE", "max_temporal_embeddings": 100},
        "image_pos_embed": {"type": "learned_abs_2d", "max_pos_embeddings": 50},
        "image_feature_source": ["spatial_avg_pool", "temporal_avg_pool"],
    },
}


CAMS = [f"{OBS_IMAGES}.cam{i}" for i in range(3)]


def tiny_policy() -> XVLAPolicy:
    config = XVLAConfig(
        input_features={
            **{cam: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)) for cam in CAMS},
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(20,))},
        florence_config=TINY_FLORENCE,
        hidden_size=32,
        depth=1,
        num_heads=2,
        chunk_size=4,
        n_action_steps=4,
        num_image_views=3,
        resize_imgs_with_padding=None,
        num_denoising_steps=2,
        device="cpu",
    )
    torch.manual_seed(0)
    return XVLAPolicy(config).eval()


def test_export_with_a_padding_view_has_fixed_shapes():
    policy = tiny_policy()
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(3, 1000, (1, 8), generator=gen)
    state = torch.randn(1, 8, generator=gen)
    images = torch.rand(1, 2, 3, 64, 64, generator=gen)
    noise = torch.randn(1, policy.config.chunk_size, policy.model.dim_action, generator=gen)

    class Predict(nn.Module):
        def forward(self, ids, state, images, noise):
            # Only two of the three cameras are present, so the model sees one padding view.
            batch = {OBS_LANGUAGE_TOKENS: ids, OBS_STATE: state, CAMS[0]: images[:, 0], CAMS[1]: images[:, 1]}
            return policy.predict_action_chunk(batch, noise=noise)

    model = Predict()
    ep = torch.export.export(model, (ids, state, images, noise))
    # Fixed-shape backends such as Torch-TensorRT cannot run sizes that depend on tensor values, and a check
    # on a tensor value makes a GPU program read it back to the host on every run.
    assert not [n for n in ep.graph.nodes if "nonzero" in str(n.target)]
    assert not ep.range_constraints
    assert torch.equal(ep.module()(ids, state, images, noise), model(ids, state, images, noise))


def test_forward_vlm_takes_the_first_views_given_a_count_and_any_mask_without_one():
    model = tiny_policy().model
    gen = torch.Generator().manual_seed(0)
    ids = torch.randint(3, 1000, (2, 8), generator=gen)
    images = torch.rand(2, 3, 3, 64, 64, generator=gen)
    prefix = torch.tensor([[1, 1, 0], [1, 1, 0]])
    counted = model.forward_vlm(ids, images, prefix, num_valid_views=2)
    read = model.forward_vlm(ids, images, prefix)
    assert counted.keys() == read.keys() and all(torch.equal(counted[k], read[k]) for k in read)
    unread = model.forward_vlm(ids, images, torch.ones(2, 3), num_valid_views=2)
    assert all(torch.equal(counted[k], unread[k]) for k in counted)
    model.forward_vlm(ids, images, torch.tensor([[1, 0, 1], [1, 1, 1]]))
    with pytest.raises(ValueError, match="At least one image view"):
        model.forward_vlm(ids, images, torch.zeros(2, 3), num_valid_views=0)
