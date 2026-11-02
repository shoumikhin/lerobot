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

"""LingBot-VA's export recipe: the cache slots its programs take and free, and what it refuses to export."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from packaging.version import Version

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig  # noqa: E402
from lerobot.policies.lingbot_va.export_lingbot_va import (  # noqa: E402
    LingBotVAExportRecipe,
    _Allocate,
    _ClearPredicted,
    _Write,
)
from lerobot.policies.lingbot_va.utils import WanAttention  # noqa: E402
from lerobot.utils.import_utils import is_package_available  # noqa: E402


def test_static_cache_takes_and_frees_slots_as_the_policy_does():
    torch.manual_seed(0)
    heads, head_dim, batch, latent, action, window = 2, 4, 2, 6, 4, 4
    slots = (window // 2) * (latent + action)
    attention = WanAttention(dim=heads * head_dim, heads=heads, dim_head=head_dim)
    attention.init_kv_cache("pos", slots, heads, head_dim, "cpu", torch.float32, batch)
    cache = attention.attn_caches["pos"]
    keys = torch.zeros(1, batch, slots, heads, head_dim)
    values = torch.zeros_like(keys)
    valid = torch.zeros(slots, dtype=torch.bool)
    ids = torch.full((slots,), -1)
    predicted = torch.zeros(slots, dtype=torch.bool)
    allocate = {latent: _Allocate(latent), action: _Allocate(action)}
    write = {latent: _Write(latent), action: _Write(action)}
    tokens = max(latent, action)

    def check():
        assert torch.equal(cache["mask"], valid)
        assert torch.equal(cache["id"][valid], ids[valid])
        assert torch.equal(cache["is_pred"][valid], predicted[valid])
        assert torch.equal(cache["k"][:, valid], keys[0][:, valid])
        assert torch.equal(cache["v"][:, valid], values[0][:, valid])

    for chunk in range(30):
        if chunk:
            attention.clear_pred_cache("pos")
            valid = _ClearPredicted()(valid, predicted)
        for count, passes in ((latent, 0), (action, 0), (latent, 3), (action, 3)):
            # The observed latents and actions are written once; the predicted ones after passes that write
            # and then free their slots, as the denoising steps do.
            is_pred = passes > 0
            new_keys = torch.randn(1, batch, tokens, heads, head_dim)
            new_values = torch.randn_like(new_keys)
            for _ in range(passes):
                taken = attention.update_cache(
                    "pos", torch.randn(batch, count, heads, head_dim), new_values[0, :, :count], False
                )
                attention.restore_cache("pos", taken)
            attention.update_cache("pos", new_keys[0, :, :count], new_values[0, :, :count], is_pred)
            valid, taken, cache_id = allocate[count](valid, ids)
            keys, values, valid, ids, predicted = write[count](
                keys,
                values,
                valid,
                ids,
                predicted,
                taken,
                cache_id,
                new_keys,
                new_values,
                torch.tensor([is_pred]),
            )
            check()
    # The first chunk's observed latents and actions were freed for newer ones.
    assert int(ids[valid].min()) > 1


# ExecuTorch 1.6 reads programs traced by PyTorch 2.15 or newer.
EXECUTORCH = is_package_available("executorch") and Version(torch.__version__).release >= (2, 15)


@pytest.mark.skipif(not EXECUTORCH, reason="needs ExecuTorch and PyTorch 2.15")
def test_cache_allocation_runs_in_executorch_on_the_cpu(tmp_path):
    from lerobot.bundle.executorch import ExecuTorchProgram
    from lerobot.export.backends.executorch_xnnpack import ExecuTorchXNNPACKConfig

    allocate = _Allocate(3)
    path = tmp_path / "allocate.pte"
    empty = (torch.zeros(8, dtype=torch.bool), torch.full((8,), -1))
    ExecuTorchXNNPACKConfig().compile(torch.export.export(allocate, empty), path)
    program = ExecuTorchProgram(path)
    states = [
        ([0, 0, 0, 0, 0, 0, 0, 0], [-1, -1, -1, -1, -1, -1, -1, -1]),
        ([1, 0, 1, 0, 0, 1, 1, 0], [0, -1, 1, -1, -1, 1, 2, -1]),
        # One slot is free, so the two slots of the oldest id are freed for the other two.
        ([1, 1, 1, 0, 1, 1, 1, 1], [3, 1, 1, -1, 2, 2, 3, 2]),
    ]
    for mask, slot_ids in states:
        valid, ids = torch.tensor(mask, dtype=torch.bool), torch.tensor(slot_ids)
        actual = program(valid.numpy(), ids.numpy())
        for value, expected in zip(actual, allocate(valid, ids), strict=True):
            np.testing.assert_array_equal(program.to_numpy(value), expected.numpy())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"camera_layout": "robotwin_tshape"}, "two streaming VAEs"),
        ({"frame_chunk_size": 1}, "frame_chunk_size is 1"),
        ({"action_per_frame": 6}, "multiple of 4"),
    ],
)
def test_export_refuses_what_it_does_not_run(change, message):
    config = LingBotVAConfig(**change)
    with pytest.raises(ValueError, match=message):
        LingBotVAExportRecipe(SimpleNamespace(config=config), None, None, None)
