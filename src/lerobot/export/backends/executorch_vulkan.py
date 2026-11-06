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

"""ExecuTorch programs that run on a GPU through Vulkan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..configs import ExportBackendConfig
from .executorch import COMPUTE_PRECISIONS, ExecuTorchConfig, lower

if TYPE_CHECKING:
    from torch.export import ExportedProgram


@ExportBackendConfig.register_subclass("executorch_vulkan")
@dataclass
class ExecuTorchVulkanConfig(ExecuTorchConfig):
    """ExecuTorch programs whose work runs on a GPU through Vulkan, such as a board's Mali or Adreno GPU.

    The `executorch` package on PyPI lowers to Vulkan but cannot run the programs: the export's own check, and
    the robot, need ExecuTorch built with its Vulkan backend (CMake option `EXECUTORCH_BUILD_VULKAN=ON`).
    """

    device = "cpu"
    # ExecuTorch's package on PyPI cannot run Vulkan programs, so the install hints name no package for it.
    packages = ()

    compute_precision: str = "float32"
    """The float type of the computation on the GPU, `float32` or `float16`. `float16` can be faster and needs
    a larger `--tolerance` than a float32 policy's default."""

    def __post_init__(self) -> None:
        """Refuse an unknown option, or an ExecuTorch that cannot run Vulkan programs here, before tracing."""
        if self.compute_precision not in COMPUTE_PRECISIONS:
            raise ValueError(
                f"--backend.compute_precision must be one of {COMPUTE_PRECISIONS}, not "
                f"{self.compute_precision!r}."
            )
        # Before importing ExecuTorch, which an older PyTorch may not load.
        self.check_torch()
        try:
            from executorch.runtime import Runtime
        except ImportError as error:
            raise ValueError(
                f"executorch_vulkan cannot import ExecuTorch's runtime ({error}). It needs ExecuTorch built "
                "with its Vulkan backend (CMake option EXECUTORCH_BUILD_VULKAN=ON). "
                "See https://huggingface.co/docs/lerobot/export."
            ) from error
        if not Runtime.get().backend_registry.is_available("VulkanBackend"):
            raise ValueError(
                "executorch_vulkan needs ExecuTorch built with its Vulkan backend (CMake option "
                "EXECUTORCH_BUILD_VULKAN=ON) and a Vulkan device, since the export ends by running the "
                "bundle. See https://huggingface.co/docs/lerobot/export."
            )

    def compile(self, program: ExportedProgram, path: Path) -> None:
        """Lower `program` to Vulkan and save it as an ExecuTorch program."""
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

        edge = lower(program, VulkanPartitioner({"force_fp16": self.compute_precision == "float16"}))
        compiled = edge.to_executorch()
        with path.open("wb") as file:
            compiled.write_to_file(file)
