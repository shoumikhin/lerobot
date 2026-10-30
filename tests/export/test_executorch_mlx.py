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

"""The executorch_mlx backend, which ExecuTorch has only on macOS."""

import sys

import pytest

from lerobot.export.backends.executorch_mlx import ExecuTorchMLXConfig


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_mlx_is_refused_before_tracing_off_a_mac(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    with pytest.raises(ValueError, match="executorch_mlx compiles only on a Mac"):
        ExecuTorchMLXConfig()
    monkeypatch.setattr(sys, "platform", "darwin")
    ExecuTorchMLXConfig()
