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

import json
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

from tests.export.backends import needs_executorch_tensorrt
from tests.torch_free import assert_runs_without_torch

# Loads the program the way a bundle does.
RUN = """
import json
import sys
from pathlib import Path

import numpy as np

from lerobot.bundle.runtime import prepare_runtime

path, data, inputs, outputs, imports, options = sys.argv[1:]
runtime = prepare_runtime("executorch", json.loads(imports), json.loads(options))
program = runtime.load(Path(path), Path(data) if data else None)
(result,) = program(np.load(inputs))
np.save(outputs, program.to_numpy(result))
"""


class ErfinvModel(nn.Module):
    """TensorRT has no converter for `erfinv`, so another backend has to run it."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.linspace(0.1, 0.9, 64))

    def forward(self, x):
        return torch.cos(torch.erfinv(torch.tanh(x) * self.scale))


def run_without_torch(program, data, x, tmp_path):
    """The program's output for `x`, loaded with its data file and the backend's runtime settings in a process
    without PyTorch."""
    from lerobot.export.backends.executorch_tensorrt import ExecuTorchTensorRTConfig

    np.save(tmp_path / "x.npy", x.cpu().numpy())
    assert_runs_without_torch(
        RUN,
        str(program),
        str(data or ""),
        str(tmp_path / "x.npy"),
        str(tmp_path / "y.npy"),
        json.dumps(ExecuTorchTensorRTConfig.runtime_imports),
        json.dumps(ExecuTorchTensorRTConfig.runtime_options),
    )
    return np.load(tmp_path / "y.npy")


@pytest.mark.gpu
@needs_executorch_tensorrt
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


def use_fake_tensorrt(monkeypatch, compile, save=None):
    """Replace Torch-TensorRT by a module whose `dynamo.compile` is `compile` and whose `save` is `save`, the
    CUDA partitioner of a part by its name, and the GPU memory config by a string."""
    from lerobot.export.backends import executorch_tensorrt

    module = types.ModuleType("torch_tensorrt")
    module.__version__ = "0.0"
    module.dynamo = types.SimpleNamespace(compile=compile)
    module.save = save
    monkeypatch.setitem(sys.modules, "torch_tensorrt", module)
    monkeypatch.setattr(executorch_tensorrt, "cuda_partitioner", lambda name: f"partitioner of {name}")
    monkeypatch.setattr(executorch_tensorrt, "gpu_memory_config", lambda: "GPU memory")


def linear_program():
    return torch.export.export(nn.Linear(2, 2), (torch.randn(1, 2),))


@pytest.mark.parametrize("changed", [{}, {"deduplicate_weights": False}])
def test_torch_tensorrt_gets_the_settings_an_executorch_program_needs(monkeypatch, tmp_path, changed):
    from lerobot.export.backends.executorch_tensorrt import ExecuTorchTensorRTConfig

    seen = {}

    def compile(program, arg_inputs, **options):
        seen["compile"] = options
        return "engine"

    def save(engine, path, **options):
        seen["save"] = engine, path, options

    use_fake_tensorrt(monkeypatch, compile, save)
    assert ExecuTorchTensorRTConfig(**changed).compile(linear_program(), tmp_path / "model.pte") is None
    assert seen["compile"] == {
        "min_block_size": 1,
        "fallback_pytorch_plugin_ops": True,
        "truncate_double": True,
        "offload_module_to_cpu": True,
        "disable_tf32": True,
        "deduplicate_weights": True,
        **changed,
    }
    assert seen["save"] == (
        "engine",
        str(tmp_path / "model.pte"),
        {
            "output_format": "executorch",
            "retrace": False,
            "use_cuda_graphs": True,
            "partitioners": ["partitioner of model"],
            "backend_config": "GPU memory",
        },
    )


def test_an_executorch_whose_cuda_backend_does_not_import_is_refused_before_tensorrt_builds(
    monkeypatch, tmp_path
):
    from lerobot.export.backends import executorch_tensorrt

    def compile(program, arg_inputs, *, fallback_pytorch_plugin_ops, **options):
        raise ReachedCompileError

    def refuse(name):
        raise ImportError("no CUDA backend")

    use_fake_tensorrt(monkeypatch, compile)
    monkeypatch.setattr(executorch_tensorrt, "cuda_partitioner", refuse)
    with pytest.raises(ImportError, match="no CUDA backend"):
        executorch_tensorrt.ExecuTorchTensorRTConfig().compile(linear_program(), tmp_path / "model.pte")
