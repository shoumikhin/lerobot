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

import tempfile
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

    An operator TensorRT does not take runs on the GPU too, compiled by ExecuTorch's CUDA backend, which needs
    the CUDA compiler (`nvcc`) at export. A TensorRT engine runs only on the GPU model that built it, so export
    on the kind of board that will run the policy. Needs `pip install "executorch>=1.6.0,<1.7.0" "torch-tensorrt[executorch]>=2.15,<2.16"`.

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

    def compile(self, program: ExportedProgram, path: Path) -> Path | None:
        """Compile `program` with Torch-TensorRT and save it as an ExecuTorch program.

        Returns the weights file of the CUDA backend's part, if it has one.
        """
        try:
            import torch_tensorrt
            from executorch.backends.cuda.cuda_backend import CudaBackend
            from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
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
        # The CUDA backend writes its weights under a fixed file name beside the program, so each part saves
        # into a folder of its own and the weights take the part's name.
        with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
            staged = Path(scratch) / path.name
            torch_tensorrt.save(
                engine,
                str(staged),
                output_format="executorch",
                # Tracing the compiled module again would drop the engines it holds.
                retrace=False,
                use_cuda_graphs=True,
                # Takes what TensorRT left, so no operator falls back to a CPU kernel between GPU parts.
                partitioners=[CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])],
                backend_config=ExecutorchBackendConfig(
                    # Inputs and outputs stay in GPU memory, so a program reads an earlier one's outputs in place.
                    propagate_device_config=PropagateDeviceConfig(
                        skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
                    ),
                    enable_non_cpu_memory_planning=True,
                    memory_planning_pass=MemoryPlanningPass(
                        alloc_graph_input=False, alloc_graph_output=False
                    ),
                ),
            )
            staged.replace(path)
            data_files = list(Path(scratch).glob("*.ptd"))
            if not data_files:
                return None
            if len(data_files) > 1:
                raise RuntimeError(f"Expected one data file beside {path.name}, got {sorted(data_files)}.")
            data = path.with_suffix(".ptd")
            data_files[0].replace(data)
            return data
