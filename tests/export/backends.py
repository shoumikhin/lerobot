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

"""What the export tests that compile with a real backend need, as one skip mark per backend.

Each mark's reason names what its tests need and the installed versions. It always names ExecuTorch, and
a CUDA GPU for a CUDA backend, so the test is marked `executorch` and `gpu` whether it skips or not (see
`tests/conftest.py`).
"""

import ctypes
import importlib
import importlib.util
import os
import platform
import subprocess
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version

import pytest
import torch
from packaging.version import Version

from lerobot.bundle.executorch import TENSOR_OUTPUT
from lerobot.utils.import_utils import is_package_available


def _installed(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "none"


def _backend_runs(name: str) -> bool:
    """Whether the installed ExecuTorch says its backend `name` is available on this machine.

    Asked in a new process: this module loads before conftest.py calls `prepare()`, and ExecuTorch's bindings
    read their result type once, when they load. The new process asks for ExecuTorch's own tensors, so it does
    not load PyTorch.
    """
    code = (
        "from executorch.runtime import Runtime; "
        f"print(Runtime.get().backend_registry.is_available({name!r}))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-P", "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, TENSOR_OUTPUT: "executorch"},
        )
    except subprocess.TimeoutExpired:
        warnings.warn(f"ExecuTorch did not say within 60 s whether {name} is available.", stacklevel=2)
        return False
    if result.returncode:
        warnings.warn(f"ExecuTorch could not say whether {name} is available:\n{result.stderr}", stacklevel=2)
        return False
    return result.stdout.strip() == "True"


# ExecuTorch's CUDA compiler imports Triton, which PyTorch's builds for Windows and for the CPU lack.
def _cuda_compiler_imports() -> bool:
    try:
        importlib.import_module("executorch.backends.cuda.cuda_partitioner")
    except ImportError:
        return False
    return True


# ExecuTorch 1.6 reads programs traced by PyTorch 2.15 or newer.
EXECUTORCH = is_package_available("executorch") and Version(torch.__version__).release >= (2, 15)
# ExecuTorch's CPU build, from PyTorch's CPU index, has the CUDA backend's compiler but not its runtime.
EXECUTORCH_CUDA = (
    EXECUTORCH and torch.cuda.is_available() and _cuda_compiler_imports() and _backend_runs("CudaBackend")
)
EXECUTORCH_TENSORRT = (
    EXECUTORCH_CUDA
    and is_package_available("torch_tensorrt")
    and is_package_available("torch_tensorrt_executorch_runtime")
)
INSTALLED = (
    f"installed: PyTorch {torch.__version__}, ExecuTorch {_installed('executorch')}, "
    f"Torch-TensorRT {_installed('torch-tensorrt')}, its runtime {_installed('torch-tensorrt-executorch-runtime')}"
)

needs_executorch = pytest.mark.skipif(
    not EXECUTORCH, reason=f"needs ExecuTorch 1.6 and PyTorch 2.15 ({INSTALLED})"
)
needs_executorch_cuda = pytest.mark.skipif(
    not EXECUTORCH_CUDA, reason=f"needs a CUDA GPU, ExecuTorch 1.6 built with CUDA and Triton ({INSTALLED})"
)
needs_executorch_tensorrt = pytest.mark.skipif(
    not EXECUTORCH_TENSORRT,
    reason="needs a CUDA GPU, ExecuTorch 1.6 built with CUDA, Triton, Torch-TensorRT and "
    f"torch-tensorrt-executorch-runtime ({INSTALLED})",
)
# ExecuTorch installs coremltools only on the Pythons coremltools supports.
needs_executorch_coreml = pytest.mark.skipif(
    not (EXECUTORCH and sys.platform == "darwin" and is_package_available("coremltools")),
    reason=f"needs macOS, ExecuTorch 1.6, PyTorch 2.15 and coremltools ({INSTALLED})",
)
# ExecuTorch for macOS has no OpenVINO backend.
needs_executorch_openvino = pytest.mark.skipif(
    not (
        EXECUTORCH
        and sys.platform != "darwin"
        and is_package_available("openvino")
        and is_package_available("nncf")
    ),
    reason="needs ExecuTorch 1.6 and PyTorch 2.15 off a Mac, and OpenVINO: pip install openvino nncf "
    f"({INSTALLED})",
)


def _executorch_has_qnn() -> bool:
    """Whether ExecuTorch was built with QNN, as its Linux x86_64 package is: its QNN modules import this."""
    try:
        return importlib.util.find_spec("executorch.backends.qualcomm.python.PyQnnManagerAdaptor") is not None
    except ImportError:
        return False


def _qnn_libraries_load() -> bool:
    try:
        ctypes.CDLL("libQnnHtp.so")
    except OSError:
        return False
    return True


# ExecuTorch's QNN compiler comes with its Linux x86_64 package, and with a build that turns QNN on.
EXECUTORCH_QNN = (
    EXECUTORCH and sys.platform == "linux" and platform.machine() == "x86_64" and _executorch_has_qnn()
)

needs_executorch_qnn_compiler = pytest.mark.skipif(
    not EXECUTORCH_QNN,
    reason=f"needs ExecuTorch 1.6 with QNN, as its Linux x86_64 package has, and PyTorch 2.15 ({INSTALLED})",
)


# ExecuTorch runs QNN programs on Qualcomm's simulator of the NPU, from the SDK's libraries.
needs_executorch_qnn = pytest.mark.skipif(
    not (EXECUTORCH_QNN and _qnn_libraries_load()),
    reason="needs ExecuTorch 1.6 with QNN, as its Linux x86_64 package has, and PyTorch 2.15, with the "
    f"libraries of Qualcomm's QNN SDK on LD_LIBRARY_PATH ({INSTALLED})",
)


# ExecuTorch asks the Vulkan driver for Arm's ML extensions, which their emulation layer adds on any Vulkan
# device.
EXECUTORCH_VGF = (
    EXECUTORCH
    and sys.platform == "linux"
    and is_package_available("ai_ml_sdk_model_converter", "model_converter")
    and _backend_runs("VgfBackend")
)

needs_executorch_vgf = pytest.mark.skipif(
    not EXECUTORCH_VGF,
    reason="needs ExecuTorch 1.6 and PyTorch 2.15 on Linux, with executorch[vgf] and a Vulkan device that "
    f"has Arm's ML extensions or their emulation layer ({INSTALLED})",
)


# ExecuTorch's packages compile Vulkan programs but cannot run them: that needs ExecuTorch built with Vulkan.
EXECUTORCH_VULKAN = EXECUTORCH and _backend_runs("VulkanBackend")

needs_executorch_vulkan = pytest.mark.skipif(
    not EXECUTORCH_VULKAN,
    reason=f"needs ExecuTorch 1.6 built with Vulkan, PyTorch 2.15 and a Vulkan device ({INSTALLED})",
)

# The mark of each backend the end-to-end tests export with. Core ML and MLX export only on a Mac, and
# OpenVINO does not run these policies yet (pytorch/executorch#24325). QNN and Vulkan fail some of them (see
# the docs), and VGF, through Arm's emulation layer when no Arm GPU is present, runs some too slowly for the
# rollout test, so test_export.py tests the three on a small model.
BACKENDS = {
    "executorch_xnnpack": needs_executorch,
    "executorch_cuda": needs_executorch_cuda,
    "executorch_tensorrt": needs_executorch_tensorrt,
}
GPU_BACKENDS = ("executorch_cuda", "executorch_tensorrt")
