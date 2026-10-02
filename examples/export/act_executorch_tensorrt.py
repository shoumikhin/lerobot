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

"""Export a trained ACT policy as an ExecuTorch program that runs on TensorRT.

Run it on the device the policy will run on, because a TensorRT engine only runs on the GPU
model that built it:

    python examples/export/act_executorch_tensorrt.py \
        --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model

Then run the exported folder with `lerobot-rollout --policy.path=<folder>`.
"""

import time
from pathlib import Path

import torch
import torch_tensorrt
from act_recipe import ACTExport, parse_args
from executorch.exir import ExecutorchBackendConfig
from executorch.exir._serialize._program import _ExtendedHeader, _flatbuffer_to_program, _get_extended_header
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig
from executorch.exir.schema import DeviceType, Tensor

EXPORTED_PROGRAM = "model.pt2"

# No copies at the program's edges: it reads the caller's GPU inputs directly and returns GPU outputs.
# Outputs stay planned in the program's own GPU memory, because a Python caller cannot provide one.
GPU_RESIDENT = ExecutorchBackendConfig(
    propagate_device_config=PropagateDeviceConfig(
        skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
    ),
    enable_non_cpu_memory_planning=True,
    memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
)


def check_program(path: Path, backend: str = "TensorRTBackend") -> None:
    """Fail unless the program is one `backend` delegate and nothing else, reading and writing GPU memory."""
    # Read only the program's description, not the engine after it, which can be several GB.
    with path.open("rb") as file:
        header = _get_extended_header(file.read(_ExtendedHeader.NUM_HEAD_BYTES))
        file.seek(0)
        plan = _flatbuffer_to_program(file.read(header.program_size)).execution_plan[0]
    delegates = [delegate.id for delegate in plan.delegates]
    operators = [operator.name for operator in plan.operators]
    on_host = [
        index
        for index in (*plan.inputs, *plan.outputs)
        if isinstance(plan.values[index].val, Tensor)
        and getattr(plan.values[index].val.extra_tensor_info, "device_type", DeviceType.CPU)
        != DeviceType.CUDA
    ]
    if delegates != [backend] or operators or on_host:
        raise SystemExit(f"{path}: delegates {delegates}, operators {operators}, CPU tensors {on_host}.")
    print(f"{path} is one {backend} delegate, with its inputs and outputs on the GPU")


def main() -> None:
    args = parse_args(__doc__, "executorch_tensorrt")
    export = ACTExport(args.policy_path, args.output_dir, args.job_name)
    pte_path = export.output_dir / "model.pte"

    start = time.perf_counter()
    with torch.no_grad():
        program = torch.export.export(export.module, export.inputs)
    if args.export_only:
        torch.export.save(program, export.output_dir / EXPORTED_PROGRAM)
        export.release_policy()
        export.write("executorch_tensorrt", pte_path.name, args.tolerance)
        return
    engine = torch_tensorrt.dynamo.compile(program, arg_inputs=export.inputs, min_block_size=1)
    del program
    export.release_policy()
    torch_tensorrt.save(
        engine, str(pte_path), output_format="executorch", retrace=False, backend_config=GPU_RESIDENT
    )
    print(f"Exported {pte_path} in {time.perf_counter() - start:.0f} s")

    check_program(pte_path)
    export.write("executorch_tensorrt", pte_path.name, args.tolerance)


if __name__ == "__main__":
    main()
