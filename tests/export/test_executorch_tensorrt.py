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

"""The executorch_tensorrt backend, on a GPU."""

import subprocess
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.utils.import_utils import is_package_available

needs_tensorrt = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_package_available("torch_tensorrt")
    or not is_package_available("executorch"),
    reason="needs a CUDA GPU, ExecuTorch and Torch-TensorRT",
)

# Loads the program the way a bundle does, in a process without PyTorch.
RUN = """
import sys
from pathlib import Path

import numpy as np

from lerobot.bundle.runtime import prepare_runtime

path, data, inputs, outputs, *runtime_imports = sys.argv[1:]
program = prepare_runtime("executorch", runtime_imports).load(Path(path), Path(data) if data else None)
(result,) = program(np.load(inputs))
np.save(outputs, program.to_numpy(result))
assert "torch" not in sys.modules
"""


def run_without_torch(program, data, x, tmp_path):
    """The program's output for `x`, loaded with its data file in a process without PyTorch."""
    from lerobot.export.backends.executorch_tensorrt import ExecuTorchTensorRTConfig

    np.save(tmp_path / "x.npy", x.cpu().numpy())
    subprocess.run(
        [
            sys.executable,
            "-c",
            RUN,
            str(program),
            str(data or ""),
            str(tmp_path / "x.npy"),
            str(tmp_path / "y.npy"),
            *ExecuTorchTensorRTConfig.runtime_imports,
        ],
        check=True,
    )
    return np.load(tmp_path / "y.npy")


class ReachedCompileError(Exception):
    pass


@pytest.mark.parametrize(("size", "expected"), [(None, {}), (1 << 20, {"workspace_size": 1 << 20})])
def test_compile_gets_the_workspace_size_only_when_it_is_set(monkeypatch, tmp_path, size, expected):
    from lerobot.export.backends.executorch_tensorrt import ExecuTorchTensorRTConfig

    seen = {}

    def compile(program, arg_inputs, *, fallback_pytorch_plugin_ops, **options):
        seen.update({key: value for key, value in options.items() if key == "workspace_size"})
        raise ReachedCompileError

    use_fake_tensorrt(monkeypatch, compile)
    with pytest.raises(ReachedCompileError):
        ExecuTorchTensorRTConfig(workspace_size=size).compile(linear_program(), tmp_path / "model.pte")
    assert seen == expected


def use_fake_tensorrt(monkeypatch, compile):
    """Replace Torch-TensorRT by a module whose `dynamo.compile` is `compile`."""
    module = types.ModuleType("torch_tensorrt")
    module.__version__ = "0.0"
    module.dynamo = types.SimpleNamespace(compile=compile)
    monkeypatch.setitem(sys.modules, "torch_tensorrt", module)


def linear_program():
    return torch.export.export(nn.Linear(2, 2), (torch.randn(1, 2),))


def test_operators_whose_plugins_need_pytorch_stay_out_of_tensorrt(monkeypatch, tmp_path):
    from lerobot.export.backends.executorch_tensorrt import ExecuTorchTensorRTConfig

    seen = {}

    def compile(program, arg_inputs, **options):
        seen.update(options)
        raise ReachedCompileError

    use_fake_tensorrt(monkeypatch, compile)
    with pytest.raises(ReachedCompileError):
        ExecuTorchTensorRTConfig().compile(linear_program(), tmp_path / "model.pte")
    assert seen["fallback_pytorch_plugin_ops"] is True
