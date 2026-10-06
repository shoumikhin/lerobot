#!/usr/bin/env python

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

"""Export a trained ACT policy as an ExecuTorch program that runs on ExecuTorch's CUDA backend.

The CUDA backend compiles the policy with AOTInductor into CUDA and Triton kernels, without
TensorRT, and saves the weights in a `.ptd` file beside the program. Run it on the device the
policy will run on, because the kernels are tuned for the GPU that compiles them:

    python examples/export/act_executorch_cuda.py \
        --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model

Then run the exported folder with `lerobot-rollout --policy.path=<folder>`.
"""

import time

import torch
from act_recipe import ACTExport, parse_args
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower


def lower_to_cuda(program: torch.export.ExportedProgram):
    """Compile the whole program into one CUDA delegate, as ExecuTorch's CUDA example does."""
    return to_edge_transform_and_lower(
        program,
        partitioner=[CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    )


def main() -> None:
    args = parse_args(__doc__, "executorch_cuda")
    if args.export_only:
        raise SystemExit("--export_only is not supported here: build_engine.py builds TensorRT engines only.")
    export = ACTExport(args.policy_path, args.output_dir, args.job_name)
    pte_path = export.output_dir / "model.pte"

    start = time.perf_counter()
    with torch.no_grad():
        program = torch.export.export(export.module, export.inputs)
    lowered = lower_to_cuda(program)
    del program
    export.release_policy()
    executorch_program = lowered.to_executorch()
    executorch_program.save(str(pte_path))
    executorch_program.write_tensor_data_to_file(str(export.output_dir))
    print(f"Exported {pte_path} in {time.perf_counter() - start:.0f} s")

    export.write("executorch_cuda", pte_path.name, args.tolerance)


if __name__ == "__main__":
    main()
