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

"""Export a trained ACT policy as an AOTInductor package that runs on TensorRT.

Run it on the device the policy will run on, because a TensorRT engine only runs on the GPU
model that built it:

    python examples/export/act_torch_tensorrt.py \
        --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model

Then run the exported folder with `lerobot-rollout --policy.path=<folder>`.
Unlike the other two backends, the package runs only with PyTorch and Torch-TensorRT installed.
"""

import torch
import torch_tensorrt
from act_recipe import ACTExport, parse_args


def main() -> None:
    args = parse_args(__doc__, "torch_tensorrt")
    export = ACTExport(args)
    package_path = export.output_dir / "model.pt2"

    with torch.no_grad():
        program = torch.export.export(export.module, export.inputs)
    engine = torch_tensorrt.dynamo.compile(program, arg_inputs=export.inputs, min_block_size=1)
    del program
    export.release_policy()
    torch_tensorrt.save(engine, str(package_path), output_format="aot_inductor", arg_inputs=export.inputs)
    export.write(package_path)


if __name__ == "__main__":
    main()
