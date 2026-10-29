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

"""ExecuTorch programs that run on NVIDIA GPUs through ExecuTorch's own CUDA backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..configs import ExportBackendConfig
from .executorch import ExecuTorchConfig, cuda_partitioner, gpu_memory_config, lower

if TYPE_CHECKING:
    from torch.export import ExportedProgram


@ExportBackendConfig.register_subclass("executorch_cuda")
@dataclass
class ExecuTorchCUDAConfig(ExecuTorchConfig):
    """ExecuTorch programs whose work runs on the GPU as CUDA kernels that ExecuTorch compiles ahead of time.

    The kernels are built for the GPU model that exports, so export on the kind of board that will run the
    policy. Needs the CUDA build of ExecuTorch, and the CUDA compiler (`nvcc`) at export.
    """

    device = "cuda"

    def compile(self, program: ExportedProgram, path: Path) -> Path:
        """Lower `program` with ExecuTorch's CUDA backend and save it as an ExecuTorch program.

        Returns the data file that holds the compiled kernels and the weights.
        """
        from executorch.exir import EdgeCompileConfig

        # The CUDA backend keeps every operator whole for its compiler, which the edge dialect check refuses.
        edge = lower(program, cuda_partitioner(path.stem), EdgeCompileConfig(_check_ir_validity=False))
        compiled = edge.to_executorch(gpu_memory_config())
        path.write_bytes(compiled.buffer)
        compiled.write_tensor_data_to_file(str(path.parent))
        return path.with_suffix(".ptd")
