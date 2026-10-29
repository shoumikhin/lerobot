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

"""What the backends whose programs ExecuTorch runs have in common."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..configs import ExportBackendConfig

if TYPE_CHECKING:
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.exir import EdgeCompileConfig, EdgeProgramManager, ExecutorchBackendConfig
    from executorch.exir.backend.partitioner import Partitioner
    from torch.export import ExportedProgram

EXECUTORCH = "executorch>=1.6.0,<1.7.0"


@dataclass
class ExecuTorchConfig(ExportBackendConfig):
    """The base of the backends that compile into ExecuTorch programs. Not a `--backend.type` itself."""

    runtime = "executorch"
    file_suffix = ".pte"
    runtime_modules = ("executorch",)
    packages = (EXECUTORCH,)
    min_torch = "2.15"
    torch_overrides = ("torch>=2.15,<2.16", "torchvision>=0.29.2,<0.30")


def lower(
    program: ExportedProgram, partitioner: Partitioner, compile_config: EdgeCompileConfig | None = None
) -> EdgeProgramManager:
    """`program` lowered with `partitioner`, the values it computes from constants alone folded first.

    An op that just builds a constant, such as position ids, may be one the backend lacks, so it stays on the
    CPU and splits the program in two, or one whose result size the GPU reads back to the host on every run.
    A backend may also take a linear layer only when its weights are constants, while attention layers slice
    theirs at run time.
    """
    from executorch.exir import to_edge_transform_and_lower
    from executorch.exir.passes.constant_prop_pass import constant_prop_pass

    # Folding skips in-place ops, so a value written in place from constants, as `ids[mask] = ...`, needs the
    # functional form first. No op is decomposed here.
    return to_edge_transform_and_lower(
        constant_prop_pass(program.run_decompositions({})),
        partitioner=[partitioner],
        compile_config=compile_config,
    )


def gpu_memory_config() -> ExecutorchBackendConfig:
    """Keep a program's inputs and outputs in GPU memory, so a program reads an earlier one's outputs in place."""
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
    from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig

    return ExecutorchBackendConfig(
        propagate_device_config=PropagateDeviceConfig(
            skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
        ),
        enable_non_cpu_memory_planning=True,
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False),
    )


def cuda_partitioner(name: str) -> CudaPartitioner:
    """Hand ExecuTorch's CUDA backend what it takes of the part `name`, its weights file named after the part."""
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

    return CudaPartitioner(
        [
            CudaBackend.generate_method_name_compile_spec("forward"),
            CudaBackend.generate_data_name_compile_spec(name),
        ]
    )
