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

"""ExecuTorch programs that run on Intel hardware through OpenVINO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..configs import ExportBackendConfig
from .executorch import EXECUTORCH, ExecuTorchConfig, lower

if TYPE_CHECKING:
    from torch.export import ExportedProgram


@ExportBackendConfig.register_subclass("executorch_openvino")
@dataclass
class ExecuTorchOpenVINOConfig(ExecuTorchConfig):
    """ExecuTorch programs whose work runs through OpenVINO, Intel's compiler, on the CPU."""

    device = "cpu"
    packages = (EXECUTORCH, "openvino", "nncf")
    # ExecuTorch's OpenVINO backend loads the OpenVINO runtime that this package installs.
    runtime_imports = ("openvino",)
    runtime_modules = ("executorch", "openvino")

    def compile(self, program: ExportedProgram, path: Path) -> None:
        """Lower `program` to OpenVINO and save it as an ExecuTorch program."""
        from executorch.backends.openvino.partitioner import OpenvinoPartitioner
        from executorch.exir.backend.compile_spec_schema import CompileSpec

        edge = lower(program, OpenvinoPartitioner([CompileSpec("device", b"CPU")]))
        path.write_bytes(edge.to_executorch().buffer)
