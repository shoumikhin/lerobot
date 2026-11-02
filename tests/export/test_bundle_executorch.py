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

"""The ExecuTorch runtime of a bundle."""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from lerobot.bundle import executorch
from tests.export.backends import needs_executorch


@pytest.mark.parametrize("preset", [None, "torch"])
def test_executorch_returns_its_own_results_without_changing_the_environment(tmp_path, preset):
    # A stand-in for ExecuTorch's bindings that prints the setting they load with.
    (tmp_path / "executorch").mkdir()
    (tmp_path / "executorch" / "__init__.py").write_text("")
    (tmp_path / "executorch" / "runtime.py").write_text(
        f"import os\nprint('loaded with', os.environ.get({executorch.TENSOR_OUTPUT!r}))\n"
    )
    env = {name: value for name, value in os.environ.items() if name != executorch.TENSOR_OUTPUT}
    if preset is not None:
        env[executorch.TENSOR_OUTPUT] = preset
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    code = (
        "import os\nfrom lerobot.bundle import executorch\nexecutorch.prepare()\n"
        "print('then', os.environ.get(executorch.TENSOR_OUTPUT))"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    # The bindings load while the setting holds; a process started later, such as an export, gets PyTorch's tensors.
    assert result.stdout.splitlines() == ["loaded with executorch", f"then {preset}"]


@needs_executorch
def test_a_scalar_input_keeps_its_rank(tmp_path):
    from executorch import exir

    class Double(torch.nn.Module):
        def forward(self, x):
            return x * 2

    program = exir.to_edge(torch.export.export(Double(), (torch.tensor(3.0),))).to_executorch()
    (tmp_path / "double.pte").write_bytes(program.buffer)
    double = executorch.ExecuTorchProgram(tmp_path / "double.pte")
    assert double.to_numpy(double(np.array(3.0, np.float32))[0]).tolist() == 6.0
