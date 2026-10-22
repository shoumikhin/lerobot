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

"""Measure the time per chunk and the accuracy of a compiled policy, and of the same policy in PyTorch.

Time per chunk is the median of three runs of 100 chunks, after 20 warm-up chunks, through the inference
engine `lerobot-rollout` uses. Accuracy compares the actions with the original LeRobot policy's, on 50
dataset frames, with the same random noise: the largest difference in each frame, then the mean, the 95th
percentile and the largest over the frames, in robot units (degrees, and percent for the gripper). The
reference runs the original policy in full float32.

The PyTorch and torch.compile runs execute the export recipe's own parts through the same engine, so only
what runs the parts differs. Run each step on the board that will run the policy, except the reference: it
is the same on every board, so run it once wherever the policy fits in float32, and share its file:

    # 50 frames from the training dataset, named like the robot's (add --rename_map as for lerobot-rollout)
    python examples/export/benchmark.py frames --dataset.repo_id=<user>/so101_dataset \\
        --output=outputs/benchmark/frames.npz
    # the reference, given an export folder of the policy for the inputs, task and noise it takes
    python examples/export/benchmark.py reference outputs/export/act --policy.path=<checkpoint> \\
        --frames=outputs/benchmark/frames.npz --output=outputs/benchmark/reference_act.npz
    # PyTorch, then torch.compile; with the export script's arguments
    python examples/export/benchmark.py pytorch --policy.path=<checkpoint> \\
        --frames=outputs/benchmark/frames.npz --output=outputs/benchmark/act_pytorch.npz
    python examples/export/benchmark.py torch.compile --policy.path=<checkpoint> \\
        --frames=outputs/benchmark/frames.npz --output=outputs/benchmark/act_compile.npz
    # each compiled folder
    python examples/export/benchmark.py outputs/export/act \\
        --frames=outputs/benchmark/frames.npz --output=outputs/benchmark/act_executorch.npz
    # the results, compared with the reference, which comes first
    python examples/export/benchmark.py report outputs/benchmark/reference_act.npz outputs/benchmark/act_*.npz

With `--duration=120` instead of `--output`, an executor runs chunks for 120 s after the warm-up and saves
nothing, for `benchmark_memory.py` to measure. It prints a line when the warm-up is done and when it stops.
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

FRAMES = 50
WARMUP_CHUNKS = 20
RUNS = 3
CHUNKS_PER_RUN = 100
# What `--duration` prints when the warm-up chunks are done, and when it stops running chunks.
WARM = "benchmark: warm"
STOPPED = "benchmark: stopped"
TASK_HELP = "For a folder that takes any task, the task to run. Default: the task it was exported with."


def time_per_chunk(chunk_ms: np.ndarray) -> float:
    """The median over runs of each run's median time per chunk, in milliseconds."""
    return float(np.median(np.median(chunk_ms, axis=1)))


def action_error(actions: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    """The mean, the 95th percentile and the largest, over frames, of each frame's largest action difference."""
    if actions.shape != reference.shape:
        raise ValueError(f"Actions shaped {actions.shape} cannot be compared with {reference.shape}.")
    per_frame = np.abs(actions - reference).max(axis=(1, 2))
    return float(per_frame.mean()), float(np.percentile(per_frame, 95)), float(per_frame.max())


def load_frames(path: Path) -> list[dict[str, np.ndarray]]:
    """The frames `benchmark.py frames` wrote, one dict per frame, as the robot gives them."""
    # Each read of an .npz entry loads all its frames, and a frame is a view that keeps them alive.
    data = dict(np.load(path))
    return [{name: np.ascontiguousarray(data[name][i]) for name in data} for i in range(FRAMES)]


def write_frames(argv: list[str]) -> None:
    """Save evenly spaced dataset frames as the robot gives them: uint8 HWC cameras, float32 state."""
    parser = argparse.ArgumentParser(description="Write the benchmark's dataset frames.")
    parser.add_argument("--dataset.repo_id", dest="repo_id", required=True)
    parser.add_argument("--dataset.root", dest="root", type=Path)
    parser.add_argument("--rename_map", type=json.loads, default={}, help="As for lerobot-rollout.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    frames: dict[str, list[np.ndarray]] = {}
    for index in np.linspace(0, len(dataset) - 1, FRAMES).round().astype(int):
        for name, value in dataset[int(index)].items():
            if name in dataset.meta.camera_keys:
                value = (value * 255).round().byte().permute(1, 2, 0)
            if name.startswith("observation."):
                frames.setdefault(args.rename_map.get(name, name), []).append(value.numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **{name: np.stack(values) for name, values in frames.items()})


class TorchProgram:
    """A recipe part run by PyTorch, behind the engine's Program protocol like a compiled program.

    The chain's first part, `starts_chunk`, starts a new CUDA graph generation each chunk: torch.compile's CUDA
    graphs keep each part's outputs until then, for the later parts of the same chunk to read.
    """

    def __init__(self, module, starts_chunk: bool = False):
        self.module = module
        self.starts_chunk = starts_chunk

    def run_device(self, *inputs):
        import torch

        if self.starts_chunk:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.inference_mode():
            outputs = self.module(*(self.upload(x) if isinstance(x, np.ndarray) else x for x in inputs))
        return outputs if isinstance(outputs, tuple) else (outputs,)

    def __call__(self, *inputs) -> np.ndarray:
        (output,) = self.run_device(*inputs)
        return output.float().cpu().numpy()

    def upload(self, value: np.ndarray):
        import torch
        from common import device

        return torch.from_numpy(value).to(device())


def torch_engine(executor: str, argv: list[str], folder: Path):
    """The engine around the recipe's parts in PyTorch, or compiled by torch.compile, and its arguments.

    `folder` receives export.json and the test case, which the engine checks on load as for an export.
    """
    import torch
    from common import parse_args

    from lerobot.inference.export import ExportInferenceEngine

    args, recipe = parse_args(__doc__, executor, (benchmark_arguments(),), argv)
    # The compiled routes run float32 matrix multiplications in TensorFloat-32 too.
    torch.set_float32_matmul_precision("high")
    if executor == "torch.compile":
        # Inductor's compile worker processes outlive the compile, and the memory benchmark counts every process.
        torch._inductor.config.compile_threads = 1
    exporter = recipe.Exporter(args)
    programs, modules = {}, {}
    for part in exporter.parts():
        programs[part.name] = {"file": "", "inputs": part.input_names, "outputs": part.output_names}
        module = part.module
        if executor == "torch.compile":
            # The graph torch.export captures, which the compiled routes build from too, replayed as CUDA
            # graphs, as the compiled routes replay theirs.
            with torch.no_grad():
                graph = torch.export.export(part.module, part.inputs).module()
                module = torch.compile(graph, mode="reduce-overhead")
        modules[part.name] = module
    args.output_dir = folder
    exporter.write(programs)
    del exporter
    # Building the parts and the folder leaves freed GPU memory in PyTorch's cache, which the parts do not use.
    torch.cuda.empty_cache()
    compiled = executor == "torch.compile"
    loaded = {
        name: TorchProgram(module, compiled and i == 0) for i, (name, module) in enumerate(modules.items())
    }
    return ExportInferenceEngine(args.output_dir, task="", programs=loaded), args


def benchmark_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--frames", type=Path, required=True, help="The file `benchmark.py frames` wrote.")
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", type=Path, help="The .npz file to write.")
    output.add_argument(
        "--duration", type=float, help="Instead, run chunks for this many seconds after the warm-up."
    )
    return parser


def measure(executor: str, argv: list[str]) -> None:
    """Run one executor on the frames and save its times per chunk and its actions."""
    from lerobot.inference.export import ExportInferenceEngine

    if executor in ("pytorch", "torch.compile"):
        with tempfile.TemporaryDirectory() as folder:
            program, args = torch_engine(executor, argv, Path(folder))
            measure_engine(executor, program, args, Path(folder))
    else:
        parser = argparse.ArgumentParser(parents=[benchmark_arguments()])
        parser.add_argument("--task", default="", help=TASK_HELP)
        args = parser.parse_args(argv)
        measure_engine(executor, ExportInferenceEngine(Path(executor), task=args.task), args, Path(executor))


def measure_engine(executor: str, program, args: argparse.Namespace, folder: Path) -> None:
    """Time `program` on the first frame, then save its actions for every frame, or with `--duration`, keep
    running it."""
    from safetensors.numpy import load_file

    info = json.loads((folder / "export.json").read_text())
    chunk_size = len(load_file(folder / info["test_case"])["expected_actions"])
    frames = load_frames(args.frames)
    if args.duration is not None:
        # A frame is a view of every frame, so only a copy lets the others go before memory is measured.
        frames = [{name: value.copy() for name, value in frames[0].items()}]

    def run_chunk(frame: dict, seed: int) -> np.ndarray:
        program.reset()
        # The engine draws each chunk's noise from NumPy, so every executor gets the same noise per frame.
        np.random.seed(seed)
        return np.stack([program.get_action(frame) for _ in range(chunk_size)])

    def chunk_ms() -> float:
        program.reset()
        start = time.perf_counter()
        program.get_action(frames[0])
        return (time.perf_counter() - start) * 1000

    for _ in range(WARMUP_CHUNKS):
        chunk_ms()
    if args.duration is not None:
        print(WARM, flush=True)
        end = time.monotonic() + args.duration
        while time.monotonic() < end:
            chunk_ms()
        print(STOPPED, flush=True)
        return
    times = np.array([[chunk_ms() for _ in range(CHUNKS_PER_RUN)] for _ in range(RUNS)])
    actions = np.stack([run_chunk(frame, seed) for seed, frame in enumerate(frames)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, chunk_ms=times, actions=actions)
    print(f"{executor}: {time_per_chunk(times):.1f} ms per chunk, wrote {args.output}")
    if executor == "torch.compile":
        from torch._dynamo.utils import counters

        # More graphs than parts means torch.compile broke a part into pieces and runs Python between them.
        print(
            f"torch.compile built {counters['stats']['unique_graphs']} graphs for {len(info['programs'])} parts, "
            f"{counters['inductor']['cudagraph_skips']} of them not replayed as CUDA graphs"
        )


def write_reference(argv: list[str]) -> None:
    """Save the original LeRobot policy's actions for every frame, as the reference.

    The policy runs as `lerobot-rollout` runs it, but entirely in float32 with no TensorFloat-32, on the
    frames, with the export folder's task and the engine's noise, so the reference has neither the recipe's
    nor a compiler's rounding.
    """
    parser = argparse.ArgumentParser(description="Write the reference actions.")
    parser.add_argument("folder", type=Path, help="An export folder of the policy.")
    parser.add_argument("--policy.path", dest="policy_path", required=True, help="Trained policy checkpoint.")
    parser.add_argument("--frames", type=Path, required=True, help="The file `benchmark.py frames` wrote.")
    parser.add_argument("--output", type=Path, required=True, help="The .npz file to write.")
    parser.add_argument("--task", default="", help=TASK_HELP)
    args = parser.parse_args(argv)
    import torch
    from common import CAMERAS_PRESENT, NOISE, device, processors
    from safetensors.numpy import load_file

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class
    from lerobot.policies.utils import prepare_observation_for_inference

    info = json.loads((args.folder / "export.json").read_text())
    # The engine gives the programs only these, so a camera the export left out stays out of the reference too.
    names = [name for name in info["inputs"] if name not in (NOISE, CAMERAS_PRESENT)]
    chunk_size = len(load_file(args.folder / info["test_case"])["expected_actions"])
    config = PreTrainedConfig.from_pretrained(args.policy_path)
    config.device = str(device())
    config.use_amp = False
    config.dtype = torch.float32
    if hasattr(config, "use_bf16"):  # GR00T
        config.use_bf16 = False
    if hasattr(config, "compile_model"):  # SmolVLA and pi0.5
        config.compile_model = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    policy = get_policy_class(config.type).from_pretrained(args.policy_path, config=config)
    # SmolVLA loads its vision-language model in bfloat16 whatever its config says.
    policy = policy.to(device(), torch.float32).eval()
    preprocessor, postprocessor = processors(config, args.policy_path)
    reference = []
    for seed, frame in enumerate(load_frames(args.frames)):
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        # The noise the engine draws for this frame, which the policy takes in place of its own.
        np.random.seed(seed)
        noise = {}
        if "noise_shape" in info:
            drawn = np.random.standard_normal(info["noise_shape"]).astype(np.float32)
            noise["noise"] = torch.from_numpy(drawn)[None].to(device())
        # A camera the frames lack is one the robot did not send, which a policy may fill in itself.
        observation = {name: frame[name] for name in names if name in frame}
        with torch.inference_mode():
            task = args.task or info.get("task") or info.get("prompt", {}).get("task", "")
            observation = prepare_observation_for_inference(observation, device(), task)
            # The whole chunk at once, as the engine runs it: GR00T with relative actions refuses one at a time.
            chunk = policy.predict_action_chunk(preprocessor(observation), **noise)[:, :chunk_size]
            reference.append(postprocessor(chunk).squeeze(0).float().cpu().numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, reference=np.stack(reference))


def report(paths: list[str]) -> None:
    """Print each file's time per chunk and its action error against the reference file, which comes first."""
    first = np.load(paths[0])
    if "reference" not in first.files:
        raise SystemExit(
            f"{paths[0]} holds no reference actions: give the file `benchmark.py reference` wrote first."
        )
    reference = first["reference"]
    print(f"{'':40} {'ms per chunk':>12} {'mean error':>11} {'p95 error':>10} {'max error':>10}")
    for path in paths[1:]:
        result = np.load(path)
        mean, p95, largest = action_error(result["actions"], reference)
        print(
            f"{Path(path).stem:40} {time_per_chunk(result['chunk_ms']):12.1f} {mean:11.3f} {p95:10.3f} {largest:10.3f}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
    elif sys.argv[1] == "frames":
        write_frames(sys.argv[2:])
    elif sys.argv[1] == "reference":
        write_reference(sys.argv[2:])
    elif sys.argv[1] == "report":
        report(sys.argv[2:])
    else:
        measure(sys.argv[1], sys.argv[2:])
