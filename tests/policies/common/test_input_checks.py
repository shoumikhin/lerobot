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

from lerobot.policies.common.input_checks import check_input


class Positive(torch.nn.Module):
    def forward(self, x):
        check_input(x > 0, lambda: "x must be positive", error_type=AssertionError)
        return x * 2


def test_check_input_raises_in_eager_mode_even_with_the_transformers_checks_off(monkeypatch):
    check_input(torch.ones(3) > 0, "never raised")
    check_input(True, "never raised")
    monkeypatch.setenv("TRANSFORMERS_DISABLE_TORCH_CHECK", "1")
    with pytest.raises(ValueError, match="not all positive"):
        check_input(torch.tensor([1.0, -1.0]) > 0, "not all positive")
    with pytest.raises(AssertionError, match="x must be positive"):
        Positive()(torch.tensor([-1.0]))


def test_check_input_is_a_runtime_assert_in_an_exported_program():
    pytest.importorskip("transformers")
    x = torch.tensor([1.0, 2.0])
    program = torch.export.export(Positive(), (x,), strict=False).module()
    assert torch.equal(program(x), x * 2)
    with pytest.raises(RuntimeError):
        program(torch.tensor([1.0, -2.0]))
