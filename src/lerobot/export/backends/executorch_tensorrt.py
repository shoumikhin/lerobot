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

"""ExecuTorch programs that run on NVIDIA GPUs through TensorRT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..configs import ExportBackendConfig

if TYPE_CHECKING:
    from torch.export import ExportedProgram


@ExportBackendConfig.register_subclass("executorch_tensorrt")
@dataclass
class ExecuTorchTensorRTConfig(ExportBackendConfig):
    """ExecuTorch programs whose work runs on the GPU through TensorRT, NVIDIA's compiler.

    A TensorRT engine runs only on the GPU model that built it, so export on the kind of board that will run
    the policy. Needs `pip install "executorch>=1.6.0,<1.7.0" "torch-tensorrt[executorch]>=2.15,<2.16"`.

    Args:
        workspace_size (`int`, *optional*):
            The most GPU memory, in bytes, TensorRT may use for scratch while it builds an engine, for
            example to fit a build into a small board. `None` lets TensorRT decide.
    """

    runtime = "executorch"
    file_suffix = ".pte"
    device = "cuda"
    # Registers the TensorRT backend with the ExecuTorch runtime.
    runtime_imports = ("torch_tensorrt_executorch_runtime",)

    workspace_size: int | None = None

    def compile(self, program: ExportedProgram, path: Path) -> None:
        """Compile `program` with Torch-TensorRT and save it as an ExecuTorch program."""
        try:
            import torch_tensorrt
            from executorch.exir import ExecutorchBackendConfig
            from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
            from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig
        except ImportError as error:
            raise ImportError(
                "The executorch_tensorrt backend needs ExecuTorch and Torch-TensorRT: "
                'pip install "executorch>=1.6.0,<1.7.0" "torch-tensorrt[executorch]>=2.15,<2.16"'
            ) from error

        options = {
            "min_block_size": 1,
            # Flow-matching time embeddings are computed in float64, which TensorRT does not support.
            "truncate_double": True,
        }
        if self.workspace_size is not None:
            options["workspace_size"] = self.workspace_size
        args, _ = program.example_inputs
        engine = torch_tensorrt.dynamo.compile(program, arg_inputs=args, **options)
        torch_tensorrt.save(
            engine,
            str(path),
            output_format="executorch",
            # Tracing the compiled module again would drop the engines it holds.
            retrace=False,
            use_cuda_graphs=True,
            backend_config=ExecutorchBackendConfig(
                # Inputs and outputs stay in GPU memory, so a program reads an earlier one's outputs in place.
                propagate_device_config=PropagateDeviceConfig(
                    skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
                ),
                enable_non_cpu_memory_planning=True,
                memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False),
            ),
        )
