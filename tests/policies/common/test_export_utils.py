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

import torch

from lerobot.policies.common.export_utils import is_non_strict_exporting


class Probe(torch.nn.Module):
    def forward(self, x):
        return x + 1 if is_non_strict_exporting() else x - 1


def test_is_non_strict_exporting_is_true_only_while_non_strict_export_traces():
    probe, x = Probe(), torch.zeros(2)
    assert torch.equal(probe(x), x - 1)
    # fullgraph, so Dynamo traces the check, where PyTorch 2.7 to 2.11 also report is_exporting().
    assert torch.equal(torch.compile(probe, backend="eager", fullgraph=True)(x), x - 1)
    assert torch.equal(torch.export.export(probe, (x,), strict=False).module()(x), x + 1)
    assert torch.equal(torch.export.export(probe, (x,), strict=True).module()(x), x - 1)
