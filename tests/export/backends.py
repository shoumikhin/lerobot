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
a CUDA GPU for a GPU backend, so the test is marked `executorch` and `gpu` whether it skips or not (see
`tests/conftest.py`).
"""

import importlib
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


# ExecuTorch's CUDA compiler imports Triton, which PyTorch's builds for Windows and for the CPU lack.
def _cuda_compiler_imports() -> bool:
    try:
        importlib.import_module("executorch.backends.cuda.cuda_partitioner")
    except ImportError:
        return False
    return True


# ExecuTorch 1.6 reads programs traced by PyTorch 2.15 or newer.
EXECUTORCH = is_package_available("executorch") and Version(torch.__version__).release >= (2, 15)
EXECUTORCH_CUDA = EXECUTORCH and torch.cuda.is_available() and _cuda_compiler_imports()
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
    not EXECUTORCH_CUDA, reason=f"needs a CUDA GPU, ExecuTorch 1.6 and Triton ({INSTALLED})"
)
needs_executorch_tensorrt = pytest.mark.skipif(
    not EXECUTORCH_TENSORRT,
    reason="needs a CUDA GPU, ExecuTorch 1.6, Triton, Torch-TensorRT and torch-tensorrt-executorch-runtime "
    f"({INSTALLED})",
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

# The mark of each backend the end-to-end tests export with. Core ML and MLX export only on a Mac, and
# OpenVINO does not run these policies yet (pytorch/executorch#24325).
BACKENDS = {
    "executorch_xnnpack": needs_executorch,
    "executorch_cuda": needs_executorch_cuda,
    "executorch_tensorrt": needs_executorch_tensorrt,
}
GPU_BACKENDS = ("executorch_cuda", "executorch_tensorrt")
