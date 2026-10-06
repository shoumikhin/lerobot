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

from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize(
    "cameras,t,p,m,gh,gw", [(2, 2, 16, 2, 16, 16), (1, 1, 2, 2, 4, 4), (2, 3, 3, 2, 4, 6)]
)
def test_patch_packing_preserves_order_with_at_most_eight_dimensions(monkeypatch, cameras, t, p, m, gh, gw):
    pytest.importorskip("transformers")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples" / "export"))
    from groot_recipe import GrootObservation

    class Observation(GrootObservation):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.frame_processor = torch.nn.Identity()
            self.cameras = [f"observation.images.camera{i}" for i in range(cameras)]
            self.frame_names = [*self.cameras, "observation.state"]
            self.normalize = self.rescale = self.normalize_state = False
            self.patch_size, self.temporal_patch_size, self.merge_size = p, t, m
            self.grid_h, self.grid_w = gh, gw
            self.max_state_dim = 6

    raw = torch.arange(cameras * gh * p * gw * p * 3).remainder(256).to(torch.uint8)
    raw = raw.reshape(cameras, gh * p, gw * p, 3)
    frame = (*raw.unbind(0), torch.arange(6).float())
    images = (raw.float() / 255).permute(0, 3, 1, 2)
    images = (images.clamp(0, 1) * 255).trunc()
    expected = images[:, None].repeat(1, t, 1, 1, 1)
    expected = expected.reshape(cameras, 1, t, 3, gh // m, m, p, gw // m, m, p)
    expected = expected.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).reshape(-1, 3 * t * p * p)

    observation = Observation()
    pixels, state = observation(*frame)
    torch.testing.assert_close(pixels, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, frame[-1][None, None], rtol=0, atol=0)
    program = torch.export.export(observation, frame).run_decompositions()
    ranks = [
        node.meta["val"].ndim
        for node in program.graph.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    ]
    assert max(ranks) <= 8
