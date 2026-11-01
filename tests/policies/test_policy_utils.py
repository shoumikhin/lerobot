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

from lerobot.policies.utils import is_exporting


class ReportsExporting(torch.nn.Module):
    """Adds 1 when traced while `is_exporting()` holds; the branch is taken at trace time."""

    def forward(self, x):
        return x + 1 if is_exporting() else x


def test_is_exporting_holds_only_while_non_strict_export_traces():
    x = torch.zeros(1)
    assert torch.equal(ReportsExporting()(x), x)
    assert torch.equal(torch.compile(ReportsExporting(), backend="eager", fullgraph=True)(x), x)
    assert torch.equal(torch.export.export(ReportsExporting(), (x,), strict=True).module()(x), x)
    assert torch.equal(torch.export.export(ReportsExporting(), (x,), strict=False).module()(x), x + 1)
