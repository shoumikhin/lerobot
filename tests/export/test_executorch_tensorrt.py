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

"""The executorch_tensorrt backend: TensorRT takes what it can, and ExecuTorch's CUDA backend the rest."""

import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from lerobot.utils.import_utils import is_package_available

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or not is_package_available("torch_tensorrt")
        or not is_package_available("executorch"),
        reason="needs a CUDA GPU, Torch-TensorRT and ExecuTorch",
    ),
]

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


class ErfinvModel(nn.Module):
    """TensorRT has no converter for `erfinv`, so another backend has to run it."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.linspace(0.1, 0.9, 64))

    def forward(self, x):
        return torch.cos(torch.erfinv(torch.tanh(x) * self.scale))


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


def test_an_operator_tensorrt_rejects_runs_on_the_gpu(tmp_path):
    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.exir.schema import DeviceType

    from lerobot.export.backends.executorch_tensorrt import ExecuTorchTensorRTConfig

    model = ErfinvModel().cuda().eval()
    x = torch.randn(8, 64, device="cuda")
    with torch.no_grad():
        program = torch.export.export(model, (x,))
        expected = model(x).cpu().numpy()

    data = ExecuTorchTensorRTConfig().compile(program, tmp_path / "model.pte")

    (plan,) = deserialize_pte_binary((tmp_path / "model.pte").read_bytes()).program.execution_plan
    assert {delegate.id for delegate in plan.delegates} == {"TensorRTBackend", "CudaBackend"}
    # Nothing runs on the CPU, and no input or output is copied to or from the host.
    assert [operator.name for operator in plan.operators] == []
    devices = [
        plan.values[index].val.extra_tensor_info.device_type for index in [*plan.inputs, *plan.outputs]
    ]
    assert devices == [DeviceType.CUDA] * 2
    actual = run_without_torch(tmp_path / "model.pte", data, x, tmp_path)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


class Buckets(nn.Module):
    """Bucket boundaries that the CUDA backend reads, so they go into that backend's data file."""

    def __init__(self, low: float, high: float):
        super().__init__()
        self.boundaries = nn.Parameter(torch.linspace(low, high, 64))

    def forward(self, x):
        return torch.bucketize(x, self.boundaries).float()


def test_parts_in_one_folder_keep_their_own_weights(tmp_path):
    from lerobot.export.backends.executorch_tensorrt import ExecuTorchTensorRTConfig

    x = torch.randn(8, 64, device="cuda")
    backend = ExecuTorchTensorRTConfig()
    parts = {}
    for name, low, high in (("first", -1.0, 1.0), ("second", -2.0, 0.0)):
        model = Buckets(low, high).cuda().eval()
        with torch.no_grad():
            program = torch.export.export(model, (x,))
            expected = model(x).cpu().numpy()
        parts[name] = (backend.compile(program, tmp_path / f"{name}.pte"), expected)
    for name, (data, expected) in parts.items():
        np.testing.assert_array_equal(
            run_without_torch(tmp_path / f"{name}.pte", data, x, tmp_path), expected
        )
    assert sorted(data.name for data, _ in parts.values()) == ["first.ptd", "second.ptd"]


def test_float32_math_stays_float32(tmp_path):
    from lerobot.export.backends.executorch_tensorrt import ExecuTorchTensorRTConfig

    # Wide enough products that TF32's 10-bit mantissa errs by about 1e-3, a hundred times float32's.
    model = nn.Sequential(nn.Linear(1024, 1024), nn.Linear(1024, 1024)).cuda().eval()
    x = torch.randn(64, 1024, device="cuda")
    with torch.no_grad():
        program = torch.export.export(model, (x,))
        expected = model(x).cpu().numpy()

    data = ExecuTorchTensorRTConfig().compile(program, tmp_path / "model.pte")

    np.testing.assert_allclose(
        run_without_torch(tmp_path / "model.pte", data, x, tmp_path), expected, atol=2e-5
    )
