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

"""Build the TensorRT engine of a folder exported with `--export_only`, for this device's GPU.

A TensorRT engine only runs on the GPU model that built it, and exporting a large policy needs more
memory than a small device has. So export once on a big machine with `--export_only`, copy the folder
to each device, and build its engine there:

    python examples/export/build_engine.py outputs/export/pi05_onnx_tensorrt

Then run the folder with `lerobot-rollout --policy.path=<folder>`.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from act_onnx_tensorrt import GIB, build_engine

from lerobot.rollout.inference.export import ExportInferenceEngine


def build_executorch(folder: Path, program_file: str, args: argparse.Namespace) -> None:
    import torch_tensorrt
    from act_executorch_tensorrt import EXPORTED_PROGRAM

    program = torch.export.load(folder / EXPORTED_PROGRAM)  # nosec B614: a folder the user exported
    inputs = program.example_inputs[0]
    options = {}
    if args.workspace_gib is not None:
        options["workspace_size"] = int(args.workspace_gib * GIB)
    if args.optimization_level is not None:
        options["optimization_level"] = args.optimization_level
    engine = torch_tensorrt.dynamo.compile(program, arg_inputs=inputs, min_block_size=1, **options)
    del program
    gc.collect()
    torch.cuda.empty_cache()
    torch_tensorrt.save(engine, str(folder / program_file), output_format="executorch", retrace=False)
    # The startup check loads the engine again; a compiled GraphModule is only freed by the cycle collector.
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", type=Path, help="A folder an export script wrote with --export_only.")
    parser.add_argument("--workspace_gib", type=float, help="Most scratch memory a TensorRT layer may use.")
    parser.add_argument(
        "--tactic_gib", type=float, help="Most memory TensorRT may use to time its kernels (ONNX only)."
    )
    parser.add_argument(
        "--optimization_level", type=int, choices=range(6), help="Lower builds faster, with less memory."
    )
    args = parser.parse_args()
    info = json.loads((args.folder / "export.json").read_text())

    if info["backend"] == "onnx_tensorrt":
        # A chunk exported with --step_engine has one ONNX file per engine, named like its program.
        for name, program in info.get("programs", {"model": info}).items():
            start = time.perf_counter()
            build_engine(
                args.folder / f"{name}.onnx",
                args.folder / program["file"],
                args.workspace_gib,
                args.tactic_gib,
                args.optimization_level,
            )
            print(f"Built {args.folder / program['file']} in {time.perf_counter() - start:.0f} s")
    else:
        start = time.perf_counter()
        build_executorch(args.folder, info["file"], args)
        print(f"Built {args.folder / info['file']} in {time.perf_counter() - start:.0f} s")

    # The rollout's own startup check: the engine must reproduce the PyTorch test case.
    ExportInferenceEngine(args.folder, task=info.get("task", ""), robot_type="")
    print(f"{args.folder} reproduces its test case")


if __name__ == "__main__":
    main()
