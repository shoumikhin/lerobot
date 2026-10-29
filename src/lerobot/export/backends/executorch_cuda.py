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

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..configs import ExportBackendConfig

if TYPE_CHECKING:
    from torch.export import ExportedProgram


@ExportBackendConfig.register_subclass("executorch_cuda")
@dataclass
class ExecuTorchCUDAConfig(ExportBackendConfig):
    """ExecuTorch programs whose work runs on the GPU as CUDA kernels that ExecuTorch compiles ahead of time.

    The kernels are built for the GPU model that exports, so export on the kind of board that will run the
    policy. Needs the CUDA build of `pip install "executorch>=1.6.0,<1.7.0"`, and the CUDA compiler (`nvcc`)
    at export.
    """

    runtime = "executorch"
    file_suffix = ".pte"
    device = "cuda"

    def compile(self, program: ExportedProgram, path: Path) -> Path:
        """Lower `program` with ExecuTorch's CUDA backend and save it as an ExecuTorch program.

        Returns the data file that holds the compiled kernels and the weights.
        """
        import torch

        try:
            from executorch.backends.cuda.cuda_backend import CudaBackend
            from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
            from executorch.exir import (
                EdgeCompileConfig,
                ExecutorchBackendConfig,
                to_edge_transform_and_lower,
            )
            from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
            from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig
        except ImportError as error:
            raise ImportError(
                'The executorch_cuda backend needs the CUDA build of ExecuTorch: pip install "executorch>=1.6.0,<1.7.0"'
            ) from error

        partitioner = CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
        # Inductor's convolution kernels follow this flag, and in TF32 they drift several times further from
        # PyTorch than PyTorch's own GPU kernels do, so the kernels are compiled in full float32.
        allow_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        try:
            # The CUDA backend keeps view operators such as split whole, which the edge dialect check refuses.
            edge = to_edge_transform_and_lower(
                program, partitioner=[partitioner], compile_config=EdgeCompileConfig(_check_ir_validity=False)
            )
        finally:
            torch.backends.cudnn.allow_tf32 = allow_tf32
        compiled = edge.to_executorch(
            ExecutorchBackendConfig(
                # Inputs and outputs stay in GPU memory, so a program reads an earlier one's outputs in place.
                propagate_device_config=PropagateDeviceConfig(
                    skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
                ),
                enable_non_cpu_memory_planning=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False),
            )
        )
        path.write_bytes(compiled.buffer)
        # The CUDA backend gives its data file the same name for every program, so it is renamed after this one.
        with tempfile.TemporaryDirectory(dir=path.parent) as folder:
            compiled.write_tensor_data_to_file(folder)
            (written,) = Path(folder).iterdir()
            return written.replace(path.with_suffix(".ptd"))
