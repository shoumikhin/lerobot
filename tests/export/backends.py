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

A released ExecuTorch can lack a fix that a backend needs, so each mark names the fix and the installed
versions. Its reason always names ExecuTorch, and a CUDA GPU for a GPU backend, so the test is marked
`executorch` and `gpu` whether it skips or not (see `tests/conftest.py`).
"""

import ctypes
import functools
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import pytest
import torch
from packaging.version import Version

from lerobot.utils.import_utils import is_package_available


def _installed(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "none"


# A package that cannot be imported, as an ExecuTorch built without CUDA, skips the tests that need it.
def _cuda_backend_names_its_weights() -> bool:
    from lerobot.export.backends.executorch import cuda_backend_names_its_weights

    try:
        return cuda_backend_names_its_weights()
    except ImportError:
        return False


# ExecuTorch 1.6 reads programs traced by PyTorch 2.15 or newer.
EXECUTORCH = is_package_available("executorch") and Version(torch.__version__).release >= (2, 15)
EXECUTORCH_CUDA = EXECUTORCH and torch.cuda.is_available() and _cuda_backend_names_its_weights()
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
    not EXECUTORCH_CUDA,
    reason="needs a CUDA GPU and an ExecuTorch whose CUDA backend names each part's weights file "
    f"(pytorch/executorch#24314) ({INSTALLED})",
)
needs_executorch_tensorrt = pytest.mark.skipif(
    not EXECUTORCH_TENSORRT,
    reason="needs a CUDA GPU, an ExecuTorch with pytorch/executorch#24314, Torch-TensorRT and "
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


@functools.cache
def _backends_that_run() -> tuple[str, ...]:
    """The backends the installed ExecuTorch runs on this machine.

    Asked in a new process: ExecuTorch's bindings read once, when they load, which result type they return, and
    a test that runs a bundle with them sets it before they load.
    """
    code = (
        "from executorch.runtime import Runtime; registry = Runtime.get().backend_registry; "
        "print(*(name for name in registry.registered_backend_names if registry.is_available(name)))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return tuple(result.stdout.split()) if result.returncode == 0 else ()


def _qnn_libraries_load() -> bool:
    try:
        ctypes.CDLL("libQnnHtp.so")
    except OSError:
        return False
    return True


# ExecuTorch's Linux x86_64 package runs QNN programs on Qualcomm's simulator of the NPU, from the SDK's libraries.
needs_executorch_qnn = pytest.mark.skipif(
    not (
        EXECUTORCH
        and sys.platform == "linux"
        and platform.machine() == "x86_64"
        and os.environ.get("QNN_SDK_ROOT")
        and _qnn_libraries_load()
        and "QnnBackend" in _backends_that_run()
    ),
    reason="needs ExecuTorch 1.6 and PyTorch 2.15 on Linux x86_64, with Qualcomm's QNN SDK in QNN_SDK_ROOT and "
    f"its libraries on LD_LIBRARY_PATH ({INSTALLED})",
)


# ExecuTorch asks the Vulkan driver for Arm's ML extensions, which their emulation layer adds on any Vulkan device.
needs_executorch_vgf = pytest.mark.skipif(
    not (
        EXECUTORCH
        and sys.platform == "linux"
        and is_package_available("ai_ml_sdk_model_converter", "model_converter")
        and "VgfBackend" in _backends_that_run()
    ),
    reason="needs ExecuTorch 1.6 and PyTorch 2.15 on Linux, with executorch[vgf] and a Vulkan device that has "
    f"Arm's ML extensions or their emulation layer ({INSTALLED})",
)


# ExecuTorch's packages compile Vulkan programs but cannot run them: that needs ExecuTorch built with Vulkan.
needs_executorch_vulkan = pytest.mark.skipif(
    not (EXECUTORCH and "VulkanBackend" in _backends_that_run()),
    reason=f"needs ExecuTorch 1.6 built with Vulkan, PyTorch 2.15 and a Vulkan device ({INSTALLED})",
)

# The mark of each backend the end-to-end tests export with. Core ML and MLX export only on a Mac, and
# OpenVINO does not run these policies yet (pytorch/executorch#24325).
BACKENDS = {
    "executorch_xnnpack": needs_executorch,
    "executorch_cuda": needs_executorch_cuda,
    "executorch_tensorrt": needs_executorch_tensorrt,
}
GPU_BACKENDS = ("executorch_cuda", "executorch_tensorrt")
