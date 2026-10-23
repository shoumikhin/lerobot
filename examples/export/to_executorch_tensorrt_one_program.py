"""Experiment: build a chain recipe's parts in ONE process and join them into ONE ExecuTorch program.

Each part is compiled by Torch-TensorRT and lowered to ExecuTorch on its own, with only its own weights in
memory. A composite module then calls the lowered parts in the chain's order, the denoising step part once per
step, and is exported as one `forward` method. The step delegate is the same object at every call, so the
program stores and loads its engine once.

    python to_executorch_tensorrt_one_program.py <to_executorch_tensorrt.py arguments> [--lean]

--lean adds Torch-TensorRT's memory-saving settings (engine_output_dir, release_weights_after_build) when the
installed Torch-TensorRT has them.
"""
import argparse
import os
import warnings
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch  # noqa: E402
import torch_tensorrt  # noqa: E402
import torch_tensorrt.executorch  # noqa: E402,F401
from build_engine import check_folder  # noqa: E402
from common import TENSORRT_OPTIONS, free_memory, make_output_dir, parse_args  # noqa: E402
from torch import nn  # noqa: E402

from lerobot.inference.export.engine import NOISE  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

extra = argparse.ArgumentParser(add_help=False)
extra.add_argument("--layers", type=int, help="Language model layers per engine.")
extra.add_argument("--blocks", type=int, help="GR00T diffusion blocks per engine.")
extra.add_argument("--lean", action="store_true")
extra.add_argument("--no_cuda_graphs", action="store_true")
extra.add_argument("--max_parts", type=int, help="Diagnose: stop after this many parts.")
args, recipe = parse_args(__doc__, "executorch_tensorrt", parents=(extra,))
if args.layers:
    recipe.LANGUAGE_LAYERS_PER_PROGRAM = args.layers
if args.blocks:
    recipe.DIT_BLOCKS_PER_PROGRAM = args.blocks
args.output_dir = make_output_dir(args.output_dir, f"{recipe.__name__}_one_program")
options = dict(TENSORRT_OPTIONS)
if args.lean:
    options["engine_output_dir"] = str(args.output_dir / "engines")
    options["release_weights_after_build"] = True


def held():
    warnings.filterwarnings("ignore")
    """What the process holds: anonymous and file RSS, big bytes objects, CPU and CUDA tensor storages."""
    import gc

    gc.collect()
    fields = dict(line.split(":", 1) for line in open("/proc/self/status") if line.startswith("Rss"))
    mib = {k: int(v.split()[0]) >> 10 for k, v in fields.items()}
    big = [o for o in gc.get_objects() if isinstance(o, (bytes, bytearray)) and len(o) > (16 << 20)]
    storages = {}
    for o in gc.get_objects():
        if isinstance(o, torch.Tensor) and not o.is_meta:
            try:
                st = o.untyped_storage()
                storages[(st.device.type, st.data_ptr())] = st.nbytes()
            except Exception:
                continue
    cpu = sum(v for (d, _), v in storages.items() if d == "cpu") >> 20
    cuda = sum(v for (d, _), v in storages.items() if d == "cuda") >> 20
    return (f"anon {mib.get('RssAnon', 0)} file {mib.get('RssFile', 0)} | bytes>16M {len(big)} "
            f"({sum(map(len, big)) >> 20} MiB) | tensors cpu {cpu} cuda {cuda} MiB")


def log(message):
    rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") >> 20
    extra_info = f" [{held()}]" if args.max_parts else ""
    print(f"[{time.strftime('%H:%M:%S')}] rss {rss} MiB: {message}{extra_info}", flush=True)


start = time.monotonic()
exporter = recipe.Exporter(args)
log("exporter ready")
lowered, programs = {}, {}
for part in exporter.parts():
    t0 = time.monotonic()
    with torch.no_grad():
        program = torch.export.export(part.module, part.inputs)
    compiled = torch_tensorrt.dynamo.compile(program, arg_inputs=part.inputs, **options)
    del program
    edge = torch_tensorrt.executorch.export(
        compiled, arg_inputs=part.inputs, use_cuda_graphs=not args.no_cuda_graphs
    )
    lowered[part.name] = edge.exported_program().module()
    programs[part.name] = {"inputs": part.input_names, "outputs": part.output_names}
    example = part.inputs
    del edge, compiled, part
    free_memory()
    log(f"lowered {list(programs)[-1]} in {time.monotonic() - t0:.0f} s")
    if args.max_parts and len(programs) >= args.max_parts:
        raise SystemExit("diagnose: stopping")

names = [*exporter.frame, NOISE]
steps = [name for name in programs if name == "step" or name.startswith("step_")]
timesteps = list(exporter.timesteps)


class OneProgram(nn.Module):
    """The chain's loop, as `ProgramChain` runs it, over the lowered parts."""

    def __init__(self):
        super().__init__()
        self.parts = nn.ModuleDict(lowered)
        for i, t in enumerate(timesteps):
            self.register_buffer(f"timestep_{i}", torch.tensor([t], dtype=torch.float32), persistent=False)

    def forward(self, *inputs):
        values = dict(zip(names, inputs, strict=True))
        *order, last = programs
        for name in order:
            if name not in steps:
                self._run(name, values)
            elif name == steps[0]:
                values["x_t"] = values[NOISE]
                for i in range(len(timesteps)):
                    values["timestep"] = getattr(self, f"timestep_{i}")
                    for step in steps:
                        self._run(step, values)
                    values["x_t"] = values["x_next"]
        return self.parts[last](*(values[n] for n in programs[last]["inputs"]))

    def _run(self, name, values):
        outputs = self.parts[name](*(values[n] for n in programs[name]["inputs"]))
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        values.update(zip(programs[name]["outputs"], outputs, strict=True))


from executorch.exir import ExecutorchBackendConfig, to_edge  # noqa: E402
from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass  # noqa: E402
from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig  # noqa: E402
from torch_tensorrt.executorch import get_edge_compile_config  # noqa: E402

device = "cuda"
inputs = (*(torch.from_numpy(x).to(device) for x in exporter.frame.values()), exporter.noise.to(device))
if getattr(exporter, "image_resize", None) is not None:
    from lerobot.inference.export.images import resize_images

    resized = resize_images(exporter.frame, exporter.image_resize)
    inputs = (*(torch.from_numpy(x).to(device) for x in resized.values()), exporter.noise.to(device))
one = OneProgram()
t0 = time.monotonic()
program = torch.export.export(one, inputs, strict=False)
calls = sum(1 for n in program.graph.nodes if "executorch_call_delegate" in str(n.target))
log(f"composite exported: {calls} delegate calls in {time.monotonic() - t0:.0f} s")
edge = to_edge(program, compile_config=get_edge_compile_config())
del program
et = edge.to_executorch(
    ExecutorchBackendConfig(
        propagate_device_config=PropagateDeviceConfig(
            skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
        ),
        enable_non_cpu_memory_planning=True,
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False),
    )
)
plan = et.executorch_program.execution_plan[0]
log(f"to_executorch: {len(plan.delegates)} delegates, "
    f"{sum(1 for c in plan.chains for i in c.instructions if type(i.instr_args).__name__ == 'DelegateCall')} calls")
with open(args.output_dir / "policy.pte", "wb") as f:
    et.write_to_file(f)
del et, edge, one, lowered
free_memory()
log(f"wrote policy.pte ({os.path.getsize(args.output_dir / 'policy.pte') >> 20} MiB)")
args.one_piece = True
exporter.write({"policy": {"file": "policy.pte", "inputs": names, "outputs": [ACTION]}})
if args.lean:
    import shutil

    shutil.rmtree(args.output_dir / "engines", ignore_errors=True)
log(f"export done in {time.monotonic() - start:.0f} s")
check_folder(args.output_dir)
