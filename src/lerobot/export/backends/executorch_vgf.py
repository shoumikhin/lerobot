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

"""ExecuTorch programs that run on Arm GPUs as VGF files, through Arm's ML extensions for Vulkan."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..configs import ExportBackendConfig
from .executorch import EXPORT_EXTRA, ExecuTorchConfig, lower

if TYPE_CHECKING:
    from torch.export import ExportedProgram


@ExportBackendConfig.register_subclass("executorch_vgf")
@dataclass
class ExecuTorchVGFConfig(ExecuTorchConfig):
    """ExecuTorch programs whose work runs on an Arm GPU through Arm's machine learning extensions for Vulkan.

    ExecuTorch's backend for Arm lowers each part to TOSA, Arm's standard operator set, and its model
    converter turns that into a VGF file, the format those extensions run. A machine without an Arm GPU runs
    the programs on any Vulkan device through Arm's emulation layer, which `executorch[vgf]` installs and
    which Vulkan loads when `VK_INSTANCE_LAYERS=VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation` and
    `VK_ADD_LAYER_PATH` name it and its libraries are on `LD_LIBRARY_PATH`.
    """

    device = "cpu"
    packages = (EXPORT_EXTRA, "executorch[vgf]")

    bfloat16: bool = False
    """Also lower a policy's bfloat16 parts with TOSA's bf16 extension, which is optional in TOSA, so a GPU
    may not run it. Without it they stay on the CPU."""

    def __post_init__(self) -> None:
        """Refuse to export where ExecuTorch cannot compile or run VGF programs, before anything is traced."""
        # The export ends by running the bundle.
        if sys.platform != "linux":
            raise ValueError(
                "executorch_vgf exports only on Linux: ExecuTorch's packages for other systems leave VGF "
                "out. Export on a Linux machine, which checks the bundle through Arm's emulation layer."
            )
        # Before importing ExecuTorch, which an older PyTorch may not load.
        self.check_torch()
        try:
            from executorch.runtime import Runtime
        except ImportError as error:
            raise ValueError(
                f"executorch_vgf cannot import ExecuTorch's runtime ({error}): pip install 'lerobot[export]' "
                "'executorch[vgf]'. See https://huggingface.co/docs/lerobot/export."
            ) from error
        try:
            from executorch.backends.arm.vgf import VgfCompileSpec
        except ImportError as error:
            raise ValueError(
                f"executorch_vgf cannot import ExecuTorch's VGF compiler ({error}): pip install "
                "'executorch[vgf]'. See https://huggingface.co/docs/lerobot/export."
            ) from error
        # ExecuTorch's own check of what its compiler runs, such as the model converter in this Python's
        # scripts.
        try:
            VgfCompileSpec().validate_environment()
        except RuntimeError as error:
            raise ValueError(
                f"executorch_vgf cannot compile here. {error}\nSee "
                "https://huggingface.co/docs/lerobot/export."
            ) from error
        if not Runtime.get().backend_registry.is_available("VgfBackend"):
            raise ValueError(
                "executorch_vgf needs an ExecuTorch that runs VGF programs, as its Linux packages do, and a "
                "Vulkan device with Arm's machine learning extensions, or Arm's emulation layer from "
                "executorch[vgf] (VK_ADD_LAYER_PATH, VK_INSTANCE_LAYERS, and its libraries on "
                "LD_LIBRARY_PATH), since the export ends by running the bundle. See "
                "https://huggingface.co/docs/lerobot/export."
            )

    def compile(self, program: ExportedProgram, path: Path) -> None:
        """Lower `program` to VGF and save it as an ExecuTorch program."""
        from executorch.backends.arm.vgf import VgfCompileSpec, VgfPartitioner

        spec = VgfCompileSpec("TOSA-1.0+FP+bf16" if self.bfloat16 else "TOSA-1.0+FP")
        edge = lower(program, VgfPartitioner(spec))
        compiled = edge.to_executorch()
        with path.open("wb") as file:
            compiled.write_to_file(file)
