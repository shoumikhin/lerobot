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
from .executorch import ExecuTorchConfig, lower

if TYPE_CHECKING:
    from torch.export import ExportedProgram


@ExportBackendConfig.register_subclass("executorch_vulkan")
@dataclass
class ExecuTorchVulkanConfig(ExecuTorchConfig):
    """ExecuTorch programs whose work runs on a GPU through Vulkan, such as a phone's or a board's Mali or Adreno GPU.

    The `executorch` package on PyPI lowers to Vulkan but cannot run the programs: the export's own check, and
    the robot, need ExecuTorch built with its Vulkan backend (`EXECUTORCH_BUILD_VULKAN=1`).
    """

    device = "cpu"

    force_fp16: bool = False
    """Compute in float16 on the GPU, which is faster and moves the actions more than float32, so a float32
    policy needs a larger `--tolerance` than its default."""

    def __post_init__(self) -> None:
        """Refuse an ExecuTorch that cannot run Vulkan programs on this machine, before anything is traced."""
        try:
            from executorch.runtime import Runtime
        except ImportError:
            # The compile step then says what to install.
            return
        # The export ends by running the bundle.
        if not Runtime.get().backend_registry.is_available("VulkanBackend"):
            raise ValueError(
                "executorch_vulkan needs ExecuTorch built with its Vulkan backend (EXECUTORCH_BUILD_VULKAN=1) and "
                "a Vulkan device, since the export ends by running the bundle. The installed ExecuTorch cannot run "
                "Vulkan programs here. See https://huggingface.co/docs/lerobot/export."
            )

    def compile(self, program: ExportedProgram, path: Path) -> None:
        """Lower `program` to Vulkan and save it as an ExecuTorch program."""
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

        edge = lower(program, VulkanPartitioner({"force_fp16": True} if self.force_fp16 else {}))
        compiled = edge.to_executorch()
        with path.open("wb") as file:
            compiled.write_to_file(file)
