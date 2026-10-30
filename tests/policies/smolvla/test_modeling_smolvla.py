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

from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching


@pytest.mark.parametrize("resize", [(4, 4), None])
def test_prepare_images_normalizes_uint8_inputs_before_resizing(resize):
    policy = SimpleNamespace(
        config=SimpleNamespace(
            image_features={"observation.images.camera": object()},
            resize_imgs_with_padding=resize,
            empty_cameras=0,
        )
    )
    image = torch.tensor([[[[0, 255], [128, 64]]]], dtype=torch.uint8).repeat(1, 3, 1, 1)

    images, image_masks = SmolVLAPolicy.prepare_images(policy, {"observation.images.camera": image})

    assert images[0].dtype == torch.float32
    assert images[0].min() >= -1.0
    assert images[0].max() <= 1.0
    assert image_masks[0].tolist() == [True]
    if resize is None:
        torch.testing.assert_close(images[0], image.float() / 255.0 * 2.0 - 1.0)


def test_embed_prefix_concatenates_one_dtype_the_one_torch_cat_promotes_to(monkeypatch):
    """bfloat16 image and text embeddings with a float32 state embedding make a float32 prefix, as in eager mode,
    from tensors that already share that dtype, as ExecuTorch needs to lower the concatenation."""
    vlm = SimpleNamespace(
        embed_image=lambda image: torch.full((1, 2, 4), 0.5, dtype=torch.bfloat16),
        embed_language_tokens=lambda tokens: torch.full((1, tokens.shape[1], 4), 0.25, dtype=torch.bfloat16),
    )
    torch.manual_seed(0)
    model = SimpleNamespace(
        vlm_with_expert=vlm, add_image_special_tokens=False, state_proj=torch.nn.Linear(3, 4), prefix_length=0
    )
    concatenated = []
    cat = torch.cat
    monkeypatch.setattr(
        torch, "cat", lambda tensors, dim: concatenated.append(list(tensors)) or cat(tensors, dim)
    )

    embs, _, _ = VLAFlowMatching.embed_prefix(
        model,
        [torch.zeros(1, 3, 8, 8)],
        [torch.ones(1, dtype=torch.bool)],
        torch.zeros(1, 5, dtype=torch.long),
        torch.ones(1, 5, dtype=torch.bool),
        torch.ones(1, 3),
    )

    with torch.no_grad():
        state = model.state_proj(torch.ones(1, 3))[:, None]
    expected = cat(
        [vlm.embed_image(None) * 2.0, vlm.embed_language_tokens(torch.zeros(1, 5)) * 2.0, state], 1
    )
    assert embs.dtype == torch.float32
    torch.testing.assert_close(embs, expected, rtol=0, atol=0)
    assert {tensor.dtype for tensor in concatenated[0]} == {torch.float32}
