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

"""Export a trained pi0.5 policy as an ExecuTorch program that runs on TensorRT.

Run it on the device the policy will run on, because a TensorRT engine only runs on the GPU
model that built it:

    python examples/export/pi05_executorch_tensorrt.py \
        --policy.path=outputs/train/pi05_so101/checkpoints/last/pretrained_model \
        --task="Pick up the block and place it in the cup"

Then run the exported folder with `lerobot-rollout --policy.path=<folder>`.
"""

import time

import torch
import torch_tensorrt
from act_executorch_tensorrt import GPU_RESIDENT, check_program
from pi05_recipe import PI05Export, parse_args


def main() -> None:
    args = parse_args(__doc__, "executorch_tensorrt")
    export = PI05Export(args.policy_path, args.task, args.output_dir, args.job_name)
    pte_path = export.output_dir / "model.pte"

    start = time.perf_counter()
    with torch.no_grad():
        program = torch.export.export(export.module, export.inputs)
    engine = torch_tensorrt.dynamo.compile(program, arg_inputs=export.inputs, min_block_size=1)
    torch_tensorrt.save(
        engine, str(pte_path), output_format="executorch", retrace=False, backend_config=GPU_RESIDENT
    )
    print(f"Exported {pte_path} in {time.perf_counter() - start:.0f} s")

    check_program(pte_path)
    export.write("executorch_tensorrt", pte_path.name, args.tolerance)


if __name__ == "__main__":
    main()
