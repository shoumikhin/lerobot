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

"""Export a trained pi0.5 policy to ONNX, then build a TensorRT engine from it.

Run it on the device the policy will run on, because a TensorRT engine only runs on the GPU
model that built it:

    python examples/export/pi05_onnx_tensorrt.py \
        --policy.path=outputs/train/pi05_so101/checkpoints/last/pretrained_model \
        --task="Pick up the block and place it in the cup"

Then run the exported folder with `lerobot-rollout --policy.path=<folder>`.

With `--step_engine`, the chunk becomes three engines instead of one: the prefix (the cameras and
the prompt to the KV cache), one denoising step, and the actions. The runtime runs the step engine
once per Euler step, so no engine holds the whole 10-step loop.
"""

import time
from pathlib import Path

import torch
from act_onnx_tensorrt import build_engine
from pi05_recipe import PI05Export, parse_args

from lerobot.utils.constants import ACTION


def export_onnx(programs: dict, folder: Path) -> dict:
    """Export each program to `<name>.onnx`; return each engine's file, inputs and outputs."""
    files = {}
    with torch.no_grad():
        for name, (module, inputs, input_names, output_names) in programs.items():
            start = time.perf_counter()
            torch.onnx.export(
                module,
                inputs,
                folder / f"{name}.onnx",
                dynamo=True,
                input_names=input_names,
                output_names=output_names,
            )
            print(f"Exported {name}.onnx in {time.perf_counter() - start:.0f} s")
            files[name] = {"file": f"{name}.engine", "inputs": input_names, "outputs": output_names}
    return files


def main() -> None:
    args = parse_args(__doc__, "onnx_tensorrt")
    export = PI05Export(args.policy_path, args.task, args.output_dir, args.job_name, args.cameras)
    if args.step_engine:
        files = export_onnx(export.denoising_programs(), export.output_dir)
    else:
        files = export_onnx(
            {"model": (export.module, export.inputs, export.input_names, [ACTION])}, export.output_dir
        )
    export.release_policy()

    if not args.export_only:
        for name, program in files.items():
            start = time.perf_counter()
            build_engine(export.output_dir / f"{name}.onnx", export.output_dir / program["file"])
            print(f"Built {program['file']} in {time.perf_counter() - start:.0f} s")

    export.write("onnx_tensorrt", files if args.step_engine else files["model"]["file"], args.tolerance)


if __name__ == "__main__":
    main()
