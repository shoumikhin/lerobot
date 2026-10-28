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

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..configs import ExportBackendConfig

if TYPE_CHECKING:
    from ..recipe import Part


@ExportBackendConfig.register_subclass("executorch_cuda")
@dataclass
class ExecuTorchCUDAConfig(ExportBackendConfig):
    """ExecuTorch programs whose work runs on the GPU as kernels that AOTInductor compiles ahead of time.

    Needs `pip install "executorch>=1.6.0,<1.7.0"` and a CUDA toolkit with `nvcc` at export.
    """

    runtime = "executorch"
    file_suffix = ".pte"
    device = "cuda"

    def compile(self, part: Part, path: Path) -> None:
        """Lower `part` with ExecuTorch's CUDA backend and save it as an ExecuTorch program."""
        import torch

        try:
            from executorch.backends.cuda.cuda_backend import CudaBackend
            from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
            from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
        except ImportError as error:
            raise ImportError(
                'The executorch_cuda backend needs ExecuTorch: pip install "executorch>=1.6.0,<1.7.0"'
            ) from error

        with torch.no_grad():
            program = torch.export.export(part.module, part.example_inputs)
        partitioner = CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
        edge = to_edge_transform_and_lower(
            program, partitioner=[partitioner], compile_config=EdgeCompileConfig(_check_ir_validity=False)
        )
        program = edge.to_executorch(ExecutorchBackendConfig())
        path.write_bytes(program.buffer)
        # The CUDA backend keeps the compiled kernels and the weights in a separate data file, named by the
        # backend, which the runtime finds beside the program.
        with tempfile.TemporaryDirectory() as folder:
            program.write_tensor_data_to_file(folder)
            (data,) = Path(folder).iterdir()
            shutil.move(data, path.with_suffix(".ptd"))
