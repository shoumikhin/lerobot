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

"""ExecuTorch programs that run on Apple Silicon through MLX."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..configs import ExportBackendConfig
from .executorch import ExecuTorchConfig, lower

if TYPE_CHECKING:
    from torch.export import ExportedProgram


@ExportBackendConfig.register_subclass("executorch_mlx")
@dataclass
class ExecuTorchMLXConfig(ExecuTorchConfig):
    """ExecuTorch programs whose work runs on the GPU of an Apple Silicon Mac, through Apple's MLX."""

    device = "cpu"

    def __post_init__(self) -> None:
        """Refuse to export where ExecuTorch has no MLX, before anything is traced."""
        if sys.platform != "darwin":
            raise ValueError(
                "executorch_mlx compiles only on a Mac: ExecuTorch's packages for other systems leave MLX out. "
                "Export on the Apple Silicon Mac that runs the policy."
            )

    def compile(self, program: ExportedProgram, path: Path) -> None:
        """Lower `program` to MLX and save it as an ExecuTorch program."""
        from executorch.backends.mlx import MLXPartitioner

        edge = lower(program, MLXPartitioner())
        compiled = edge.to_executorch()
        with path.open("wb") as file:
            compiled.write_to_file(file)
