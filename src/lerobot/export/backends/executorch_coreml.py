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

"""ExecuTorch programs that run on Apple devices through Core ML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..configs import ExportBackendConfig
from .executorch import ExecuTorchConfig, lower

if TYPE_CHECKING:
    from torch.export import ExportedProgram

COMPUTE_UNITS = ("all", "cpu_only", "cpu_and_gpu", "cpu_and_ne")
COMPUTE_PRECISIONS = ("float32", "float16")


@ExportBackendConfig.register_subclass("executorch_coreml")
@dataclass
class ExecuTorchCoreMLConfig(ExecuTorchConfig):
    """ExecuTorch programs whose work runs through Core ML, Apple's runtime, on a Mac, iPhone or iPad.

    Core ML picks the CPU, the GPU or the Neural Engine for each part of the program when it loads it. The
    ExecuTorch package for macOS brings `coremltools`.

    Args:
        compute_units (`str`, *optional*, defaults to `"all"`):
            The hardware Core ML may use: `all`, `cpu_only`, `cpu_and_gpu` or `cpu_and_ne` (the CPU and the
            Neural Engine).
        compute_precision (`str`, *optional*, defaults to `"float32"`):
            The float type of the computation, `float32` or `float16`. The Neural Engine runs only `float16`,
            which moves the actions further from PyTorch's.
    """

    device = "cpu"

    compute_units: str = "all"
    compute_precision: str = "float32"

    def __post_init__(self) -> None:
        """Refuse an unknown compute unit or precision before anything is traced."""
        if self.compute_units not in COMPUTE_UNITS:
            raise ValueError(
                f"--backend.compute_units must be one of {COMPUTE_UNITS}, not {self.compute_units!r}."
            )
        if self.compute_precision not in COMPUTE_PRECISIONS:
            raise ValueError(
                f"--backend.compute_precision must be one of {COMPUTE_PRECISIONS}, not {self.compute_precision!r}."
            )

    def compile(self, program: ExportedProgram, path: Path) -> None:
        """Lower `program` to Core ML and save it as an ExecuTorch program."""
        import coremltools as ct
        from executorch.backends.apple.coreml.compiler import CoreMLBackend
        from executorch.backends.apple.coreml.partition import CoreMLPartitioner

        compile_specs = CoreMLBackend.generate_compile_specs(
            compute_unit=ct.ComputeUnit[self.compute_units.upper()],
            compute_precision=ct.precision[self.compute_precision.upper()],
        )
        edge = lower(program, CoreMLPartitioner(compile_specs=compile_specs))
        path.write_bytes(edge.to_executorch().buffer)
