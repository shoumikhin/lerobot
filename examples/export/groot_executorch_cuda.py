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

"""Export a trained GR00T N1.7 policy as an ExecuTorch program that runs on ExecuTorch's CUDA backend.

The CUDA backend compiles the policy with AOTInductor into CUDA and Triton kernels, without
TensorRT, and saves the weights in a `.ptd` file beside the program. Run it on the device the
policy will run on, because the kernels are tuned for the GPU that compiles them:

    python examples/export/groot_executorch_cuda.py \
        --policy.path=outputs/train/groot_so101/checkpoints/last/pretrained_model \
        --dataset.repo_id=<user>/so101_dataset

Then run the exported folder with `lerobot-rollout --policy.path=<folder>`, with the same task.
"""

import time

import torch
from act_executorch_cuda import lower_to_cuda
from act_executorch_tensorrt import GPU_RESIDENT, check_program
from groot_recipe import GrootExport, parse_args


def main() -> None:
    args = parse_args(__doc__, "executorch_cuda")
    if args.export_only:
        raise SystemExit("--export_only is not supported here: build_engine.py builds TensorRT engines only.")
    export = GrootExport(args.policy_path, args.dataset, args.dataset_root, args.output_dir, args.job_name)
    pte_path = export.output_dir / "model.pte"

    start = time.perf_counter()
    with torch.no_grad():
        program = torch.export.export(export.module, export.inputs)
    lowered = lower_to_cuda(program)
    del program
    export.release_policy()
    executorch_program = lowered.to_executorch(GPU_RESIDENT)
    executorch_program.save(str(pte_path))
    executorch_program.write_tensor_data_to_file(str(export.output_dir))
    print(f"Exported {pte_path} in {time.perf_counter() - start:.0f} s")

    check_program(pte_path, "CudaBackend")
    export.write("executorch_cuda", pte_path.name, args.tolerance)


if __name__ == "__main__":
    main()
