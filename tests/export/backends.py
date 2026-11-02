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

A released ExecuTorch or Torch-TensorRT can lack a fix that a backend needs, so each mark names the fixes and the
installed versions. Its reason always names ExecuTorch, and a CUDA GPU for a GPU backend, so the test is marked
`executorch` and `gpu` whether it skips or not (see `tests/conftest.py`).
"""

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


def _cuda_backend_names_its_weights() -> bool:
    from lerobot.export.backends.executorch import cuda_backend_names_its_weights

    return cuda_backend_names_its_weights()


def _tensorrt_takes_the_plugin_setting() -> bool:
    from lerobot.export.backends.executorch_tensorrt import tensorrt_takes_the_plugin_setting

    return tensorrt_takes_the_plugin_setting()


# ExecuTorch 1.6 reads programs traced by PyTorch 2.15 or newer.
EXECUTORCH = is_package_available("executorch") and Version(torch.__version__).release >= (2, 15)
EXECUTORCH_CUDA = EXECUTORCH and torch.cuda.is_available() and _cuda_backend_names_its_weights()
EXECUTORCH_TENSORRT = (
    EXECUTORCH_CUDA
    and is_package_available("torch_tensorrt")
    and _tensorrt_takes_the_plugin_setting()
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
    reason="needs a CUDA GPU, an ExecuTorch with pytorch/executorch#24314, a Torch-TensorRT with "
    f"pytorch/TensorRT#4953 and torch-tensorrt-executorch-runtime ({INSTALLED})",
)

# The mark of each backend that ExecuTorch runs.
BACKENDS = {
    "executorch_xnnpack": needs_executorch,
    "executorch_cuda": needs_executorch_cuda,
    "executorch_tensorrt": needs_executorch_tensorrt,
}
GPU_BACKENDS = ("executorch_cuda", "executorch_tensorrt")
