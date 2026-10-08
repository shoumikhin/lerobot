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

"""Export a trained SmolVLA policy to ONNX, then build a TensorRT engine from it.

Run it on the device the policy will run on, because a TensorRT engine only runs on the GPU
model that built it:

    python examples/export/smolvla_onnx_tensorrt.py \
        --policy.path=outputs/train/smolvla_so101/checkpoints/last/pretrained_model \
        --dataset.repo_id=<user>/so101_dataset

Then run the exported folder with `lerobot-rollout --policy.path=<folder>`, with the same task.
"""

import torch
from build_engine import build_engine
from smolvla_recipe import SmolVLAExport, parse_args

from lerobot.utils.constants import ACTION


def main() -> None:
    args = parse_args(__doc__, "onnx_tensorrt")
    export = SmolVLAExport(args)
    onnx_path = export.output_dir / "model.onnx"
    engine_path = export.output_dir / "model.engine"

    with torch.no_grad():
        torch.onnx.export(
            export.module,
            export.inputs,
            onnx_path,
            dynamo=True,
            input_names=export.input_names,
            output_names=[ACTION],
        )
    export.release_policy()
    if not args.export_only:
        build_engine(onnx_path, engine_path)
    export.write(engine_path)


if __name__ == "__main__":
    main()
