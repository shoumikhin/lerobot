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

if TYPE_CHECKING:
    from torch.export import ExportedProgram


@ExportBackendConfig.register_subclass("executorch_openvino")
@dataclass
class ExecuTorchOpenVINOConfig(ExportBackendConfig):
    """ExecuTorch programs whose work runs through OpenVINO, Intel's compiler, on the CPU.

    Needs `pip install "executorch>=1.6.0,<1.7.0" openvino nncf`.
    """

    runtime = "executorch"
    file_suffix = ".pte"
    device = "cpu"
    # ExecuTorch's OpenVINO backend loads the OpenVINO runtime that this package installs.
    runtime_imports = ("openvino",)

    def compile(self, program: ExportedProgram, path: Path) -> None:
        """Lower `program` to OpenVINO and save it as an ExecuTorch program."""
        try:
            from executorch.backends.openvino.partitioner import OpenvinoPartitioner
            from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
            from executorch.exir.backend.compile_spec_schema import CompileSpec
        except ImportError as error:
            raise ImportError(
                "The executorch_openvino backend needs ExecuTorch, OpenVINO and NNCF: "
                'pip install "executorch>=1.6.0,<1.7.0" openvino nncf'
            ) from error

        partitioner = OpenvinoPartitioner([CompileSpec("device", b"CPU")])
        edge = to_edge_transform_and_lower(program, partitioner=[partitioner])
        # With OpenVINO 2026, a program whose delegate data sits in a separate segment crashes when loaded from a
        # file, so the data stays inside the program.
        path.write_bytes(edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False)).buffer)
