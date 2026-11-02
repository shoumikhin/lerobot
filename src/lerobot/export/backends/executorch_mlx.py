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

    def compile(self, program: ExportedProgram, path: Path) -> None:
        """Lower `program` to MLX and save it as an ExecuTorch program."""
        from executorch.backends.mlx import MLXPartitioner

        edge = lower(program, MLXPartitioner())
        path.write_bytes(edge.to_executorch().buffer)
