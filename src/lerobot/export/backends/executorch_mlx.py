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

if TYPE_CHECKING:
    from torch.export import ExportedProgram


@ExportBackendConfig.register_subclass("executorch_mlx")
@dataclass
class ExecuTorchMLXConfig(ExportBackendConfig):
    """ExecuTorch programs whose work runs on the GPU of an Apple Silicon Mac, through Apple's MLX.

    Needs `pip install "executorch>=1.6.0,<1.7.0"` on macOS on Apple Silicon.
    """

    runtime = "executorch"
    file_suffix = ".pte"
    device = "cpu"

    def compile(self, program: ExportedProgram, path: Path) -> None:
        """Lower `program` to MLX and save it as an ExecuTorch program."""
        try:
            from executorch.backends.mlx import MLXPartitioner
            from executorch.exir import to_edge_transform_and_lower
            from executorch.exir.passes.constant_prop_pass import constant_prop_pass
        except ImportError as error:
            raise ImportError(
                'The executorch_mlx backend needs ExecuTorch: pip install "executorch>=1.6.0,<1.7.0"'
            ) from error

        # Values computed from constants alone, such as position embeddings, are folded now: MLX lacks some ops
        # that build them (a number raised to a tensor power), and one left on the CPU splits the program in two.
        edge = to_edge_transform_and_lower(constant_prop_pass(program), partitioner=[MLXPartitioner()])
        path.write_bytes(edge.to_executorch().buffer)
