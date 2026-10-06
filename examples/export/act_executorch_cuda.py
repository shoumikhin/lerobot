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

import torch
from act_recipe import ACTExport, parse_args
from executorch.backends.cuda.cuda_backend import CudaBackend
from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower


def main() -> None:
    args = parse_args(__doc__, "executorch_cuda")
    export = ACTExport(args)
    pte_path = export.output_dir / "model.pte"

    with torch.no_grad():
        program = torch.export.export(export.module, export.inputs)
    # Compile the whole program into one CUDA delegate, as ExecuTorch's CUDA example does.
    lowered = to_edge_transform_and_lower(
        program,
        partitioner=[CudaPartitioner([CudaBackend.generate_method_name_compile_spec("forward")])],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    )
    del program
    export.release_policy()
    executorch_program = lowered.to_executorch()
    executorch_program.save(str(pte_path))
    executorch_program.write_tensor_data_to_file(str(export.output_dir))
    export.write(pte_path)


if __name__ == "__main__":
    main()
