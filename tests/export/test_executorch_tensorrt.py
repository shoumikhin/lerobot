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

import importlib.util
import json
import os
import re
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


# A workspace size goes to Torch-TensorRT only when it is set.
@pytest.mark.parametrize("changed", [{}, {"deduplicate_weights": False}, {"workspace_size": 1 << 20}])
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
        "release_weights_after_build": True,
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


def use_fake_modelopt(monkeypatch, calls):
    """Replace Model Optimizer by modules whose `quantize` records its arguments in `calls`."""
    quantization = types.ModuleType("modelopt.torch.quantization")
    quantization.FP8_DEFAULT_CFG = {
        "quant_cfg": [{"quantizer_name": "*", "enable": True}],
        "algorithm": "max",
    }

    def quantize(module, config, forward_loop):
        # Model Optimizer 0.48 refuses a model in which the config leaves no weight to quantize.
        if os.environ.get("MODELOPT_SKIP_WEIGHT_QUANT_CHECK") != "1":
            raise RuntimeError("no weight quantizer is enabled")
        calls.append({"config": config, "training": [m.training for m in module.modules()]})
        forward_loop(module)

    quantization.quantize = quantize
    utils = types.ModuleType("modelopt.torch.quantization.utils")
    utils.export_torch_mode = lambda: "export mode"
    for name, module in {
        "modelopt": types.ModuleType("modelopt"),
        "modelopt.torch": types.ModuleType("modelopt.torch"),
        "modelopt.torch.quantization": quantization,
        "modelopt.torch.quantization.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return quantization


class Recorded(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x)
        return self.linear(x)


def fp8_gpu(monkeypatch, capability=(9, 0)):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "a GPU")


def test_fp8_calibrates_each_part_on_its_own_inputs_in_eval_mode(monkeypatch):
    from lerobot.export.backends.executorch_tensorrt import FP8_KEPT, ExecuTorchTensorRTConfig

    calls = []
    quantization = use_fake_modelopt(monkeypatch, calls)
    fp8_gpu(monkeypatch)
    part, x = Recorded().train(), torch.randn(1, 2)
    monkeypatch.delenv("MODELOPT_SKIP_WEIGHT_QUANT_CHECK", raising=False)
    # The part has no weight the config quantizes, as pi0.5's vision part, and is calibrated all the same.
    assert ExecuTorchTensorRTConfig().quantized(part, (x,), "fp8") == "export mode"
    assert "MODELOPT_SKIP_WEIGHT_QUANT_CHECK" not in os.environ
    (call,) = calls
    assert not any(call["training"])
    assert part.inputs == [x]
    assert call["config"]["quant_cfg"] == [*quantization.FP8_DEFAULT_CFG["quant_cfg"], *FP8_KEPT]
    assert call["config"]["algorithm"] == "max"
    # Model Optimizer's own config is left as it was.
    assert len(quantization.FP8_DEFAULT_CFG["quant_cfg"]) == 1


@pytest.mark.parametrize("reason", ["gpu without fp8", "no modelopt"])
def test_fp8_is_refused_before_it_calibrates(monkeypatch, reason):
    from lerobot.export.backends.executorch_tensorrt import ExecuTorchTensorRTConfig

    calls = []
    use_fake_modelopt(monkeypatch, calls)
    fp8_gpu(monkeypatch, (8, 7) if reason == "gpu without fp8" else (9, 0))
    if reason == "no modelopt":
        monkeypatch.setitem(sys.modules, "modelopt.torch.quantization", None)
    error, match = {
        "gpu without fp8": (ValueError, "compute capability 8.9 or newer"),
        "no modelopt": (
            ImportError,
            r"Optimizer \(.*None in sys\.modules\)\. .*pip install .nvidia-modelopt>=0.48.",
        ),
    }[reason]
    with pytest.raises(error, match=match):
        ExecuTorchTensorRTConfig().quantized(Recorded(), (torch.randn(1, 2),), "fp8")
    assert calls == []


FP8_GPU = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


@pytest.mark.gpu
@needs_executorch_tensorrt
@pytest.mark.skipif(
    not (FP8_GPU and importlib.util.find_spec("modelopt")),
    reason="needs a GPU of compute capability 8.9 or newer and nvidia-modelopt",
)
def test_an_fp8_bundle_exports_and_passes_its_check(tmp_path, capfd):
    from lerobot.scripts.lerobot_export import export
    from tests.export.tiny_act import IMAGE, save_checkpoint

    save_checkpoint(tmp_path / "checkpoint", IMAGE)
    export(
        [
            "--backend.type=executorch_tensorrt",
            f"--policy.path={tmp_path / 'checkpoint'}",
            f"--output_dir={tmp_path / 'bundle'}",
            "--quantize=fp8",
            "--tolerance=1",
        ]
    )
    quantized = [
        int(n) for n in re.findall(r"Quantized the weights of (\d+) layers in fp8", capfd.readouterr().err)
    ]
    assert quantized and min(quantized) > 0


def test_fp8_leaves_unquantized_a_layer_whose_input_was_all_zeros(monkeypatch):
    """Its scale would be 0, which TensorRT divides by. ACT feeds its transformer such a latent."""
    pytest.importorskip("modelopt.torch.quantization")
    from lerobot.export.backends.executorch_tensorrt import ExecuTorchTensorRTConfig

    class TwoInputs(nn.Module):
        def __init__(self):
            super().__init__()
            self.latent, self.other = nn.Linear(4, 4), nn.Linear(4, 4)

        def forward(self, x, latent):
            return self.other(x) + self.latent(latent)

    fp8_gpu(monkeypatch)
    part = TwoInputs()
    ExecuTorchTensorRTConfig().quantized(part, (torch.randn(2, 4), torch.zeros(2, 4)), "fp8")
    assert not part.latent.input_quantizer.is_enabled
    assert part.other.input_quantizer.is_enabled and part.other.weight_quantizer.is_enabled
