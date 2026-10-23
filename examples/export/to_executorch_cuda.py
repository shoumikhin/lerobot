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

"""Export a trained policy as ExecuTorch programs that run on ExecuTorch's CUDA backend, with no PyTorch at
run time.

The CUDA backend compiles each program with AOTInductor into CUDA kernels tuned on this GPU, so run it on the
board that will run the policy:

    python examples/export/to_executorch_cuda.py \\
        --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model

Then run the folder it writes with `lerobot-rollout --policy.path=<folder>`.
"""

import tempfile
from pathlib import Path

import torch
from common import Part, parse_args, run_route
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig


def compile_part(part: Part, path: Path) -> None:
    with torch.no_grad():
        program = torch.export.export(part.module, part.inputs)
    partitioner = CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])
    # Float32 matrix multiplications and convolutions in TensorFloat-32, as TensorRT builds them.
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.fp32_precision = "tf32"
    # As ExecuTorch's CUDA backend guide lowers: the backend takes operators, such as split, that the default
    # edge check refuses.
    edge = to_edge_transform_and_lower(
        program,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    )
    del program
    executorch_program = edge.to_executorch(
        ExecutorchBackendConfig(
            # The rollout hands each program its inputs, and takes its outputs, on the GPU.
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
            ),
            enable_non_cpu_memory_planning=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False),
        )
    )
    path.write_bytes(executorch_program.buffer)
    # The weights and the compiled kernels go to a data file the backend names the same for every program.
    with tempfile.TemporaryDirectory(dir=path.parent) as folder:
        executorch_program.write_tensor_data_to_file(folder)
        (data,) = Path(folder).glob("*.ptd")
        data.rename(path.with_suffix(".ptd"))


if __name__ == "__main__":
    args, recipe = parse_args(__doc__, "executorch_cuda")
    run_route(__file__, args, recipe, compile_part, ".pte")
