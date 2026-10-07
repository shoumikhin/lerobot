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

With `--step_engine`, the chunk becomes a chain of programs: the image and prompt embeddings, the
language model in groups of `--layers_per_program` layers, one denoising step, and the actions.
Each program is built in its own process, which loads only that program's weights from the
checkpoint, so a device with less memory than the whole policy can export it.
"""

import subprocess
import sys

import torch
import torch_tensorrt
from act_recipe import make_output_dir, save_for_build_engine
from pi05_recipe import PI05Export, free_memory, parse_args, part_names


def main() -> None:
    args = parse_args(__doc__, "executorch_tensorrt")
    if args.step_engine and args.export_only:
        raise SystemExit("--step_engine exports each program on the device itself; drop --export_only.")
    if args.step_engine and args.part is None:
        # A process returns all its memory when it exits; one that built a part keeps some of it.
        output_dir = make_output_dir(args.output_dir, args.job_name)
        for name in part_names(args.policy_path, args.layers_per_program):
            command = [
                sys.executable,
                __file__,
                *sys.argv[1:],
                f"--output_dir={output_dir}",
                f"--part={name}",
            ]
            subprocess.run(command, check=True)
        return
    export = PI05Export(args, split=args.step_engine)
    options = {"min_block_size": 1, "offload_module_to_cpu": args.offload_module_to_cpu}
    if args.workspace_gib is not None:
        options["workspace_size"] = int(args.workspace_gib * (1 << 30))
    if args.optimization_level is not None:
        options["optimization_level"] = args.optimization_level
    if args.step_engine:
        programs = {}
        for name, module, inputs, input_names, output_names in export.parts(args.layers_per_program):
            programs[name] = {"file": f"{name}.pte", "inputs": input_names, "outputs": output_names}
            if name != args.part:
                # The parts before this one run in PyTorch only, to make its example inputs.
                del module
                free_memory()
                continue
            print(f"{name}: exporting", flush=True)
            with torch.no_grad():
                program = torch.export.export(module, inputs)
            print(f"{name}: building the TensorRT engine", flush=True)
            engine = torch_tensorrt.dynamo.compile(program, arg_inputs=inputs, **options)
            torch_tensorrt.save(
                engine, str(export.output_dir / f"{name}.pte"), output_format="executorch", retrace=False
            )
            print(f"Wrote {name}.pte", flush=True)
            if name == "actions":
                export.write(programs)
            return

    pte_path = export.output_dir / "model.pte"
    with torch.no_grad():
        program = torch.export.export(export.module, export.inputs)
    if args.export_only:
        save_for_build_engine(export, program, pte_path)
        return
    engine = torch_tensorrt.dynamo.compile(program, arg_inputs=export.inputs, **options)
    del program
    export.release_policy()
    torch_tensorrt.save(engine, str(pte_path), output_format="executorch", retrace=False)
    export.write(pte_path)


if __name__ == "__main__":
    main()
