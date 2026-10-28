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
    from ..recipe import Part


@ExportBackendConfig.register_subclass("executorch_openvino")
@dataclass
class ExecuTorchOpenVINOConfig(ExportBackendConfig):
    """ExecuTorch programs whose work runs through OpenVINO, Intel's compiler, on the CPU.

    Needs an ExecuTorch build with its OpenVINO backend, and `pip install openvino nncf`.
    """

    runtime = "executorch"
    file_suffix = ".pte"
    device = "cpu"

    def compile(self, part: Part, path: Path) -> None:
        """Lower `part` to OpenVINO and save it as an ExecuTorch program."""
        import torch

        try:
            from executorch.backends.openvino.partitioner import OpenvinoPartitioner
            from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
            from executorch.exir.backend.compile_spec_schema import CompileSpec
        except ImportError as error:
            raise ImportError(
                "The executorch_openvino backend needs ExecuTorch built with its OpenVINO backend, "
                "and pip install openvino nncf"
            ) from error

        with torch.no_grad():
            program = torch.export.export(part.module, part.example_inputs)
        partitioner = OpenvinoPartitioner([CompileSpec("device", b"CPU")])
        edge = to_edge_transform_and_lower(program, partitioner=[partitioner])
        # ExecuTorch's OpenVINO backend frees a separate delegate segment while OpenVINO still reads it, so a
        # program loaded from a file crashes unless the segment stays inside the program.
        path.write_bytes(edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False)).buffer)
