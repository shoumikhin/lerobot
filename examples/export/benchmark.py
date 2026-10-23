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
engine `lerobot-rollout` uses. Accuracy compares the actions with PyTorch's on 50 dataset frames, with the
same random noise: the largest difference in each frame, then the mean and the 95th percentile over the
frames, in robot units (degrees, and percent for the gripper).

The PyTorch and torch.compile runs execute the export recipe's own parts through the same engine, so only
what runs the parts differs. Run each step on the board that will run the policy:

    # 50 frames from the training dataset, named like the robot's (add --rename_map as for lerobot-rollout)
    python examples/export/benchmark.py frames --dataset.repo_id=<user>/so101_dataset \\
        --output=outputs/benchmark/frames.npz
    # the reference, in PyTorch, then torch.compile; with the export script's arguments
    python examples/export/benchmark.py pytorch --policy.path=<checkpoint> \\
        --frames=outputs/benchmark/frames.npz --output=outputs/benchmark/act_pytorch.npz
    python examples/export/benchmark.py torch.compile --policy.path=<checkpoint> \\
        --frames=outputs/benchmark/frames.npz --output=outputs/benchmark/act_compile.npz
    # each compiled folder
    python examples/export/benchmark.py outputs/export/act \\
        --frames=outputs/benchmark/frames.npz --output=outputs/benchmark/act_executorch.npz
    # the results, compared with the first file
    python examples/export/benchmark.py report outputs/benchmark/act_pytorch.npz outputs/benchmark/act_*.npz
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np

FRAMES = 50
WARMUP_CHUNKS = 20
RUNS = 3
CHUNKS_PER_RUN = 100


def time_per_chunk(chunk_ms: np.ndarray) -> float:
    """The median over runs of each run's median time per chunk, in milliseconds."""
    return float(np.median(np.median(chunk_ms, axis=1)))


def action_error(actions: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """The mean and the 95th percentile, over frames, of each frame's largest action difference."""
    per_frame = np.abs(actions - reference).max(axis=(1, 2))
    return float(per_frame.mean()), float(np.percentile(per_frame, 95))


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
    """A recipe part run by PyTorch, behind the engine's Program protocol like a compiled program."""

    def __init__(self, module):
        self.module = module

    def run_device(self, *inputs):
        import torch

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

    from lerobot.inference.export import ExportInferenceEngine, engine

    args, recipe = parse_args(__doc__, executor, (benchmark_arguments(),), argv)
    # The compiled routes run float32 matrix multiplications in TensorFloat-32 too.
    torch.set_float32_matmul_precision("high")
    exporter = recipe.Exporter(args)
    programs, modules = {}, {}
    for part in exporter.parts():
        programs[part.name] = {"file": "", "inputs": part.input_names, "outputs": part.output_names}
        module = part.module
        if executor == "torch.compile":
            # The graph torch.export captures, which the compiled routes build from too.
            with torch.no_grad():
                module = torch.compile(torch.export.export(part.module, part.inputs).module())
        modules[part.name] = module
    args.output_dir = folder
    exporter.write(programs)
    loaded = {name: TorchProgram(module) for name, module in modules.items()}
    with mock.patch.object(engine, "load_programs", lambda folder, info: loaded):
        return ExportInferenceEngine(args.output_dir, task=""), args


def benchmark_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--frames", type=Path, required=True, help="The file `benchmark.py frames` wrote.")
    parser.add_argument("--output", type=Path, required=True, help="The .npz file to write.")
    return parser


def measure(executor: str, argv: list[str]) -> None:
    """Run one executor on the frames and save its times per chunk and its actions."""
    from lerobot.inference.export import ExportInferenceEngine

    if executor in ("pytorch", "torch.compile"):
        with tempfile.TemporaryDirectory() as folder:
            program, args = torch_engine(executor, argv, Path(folder))
            measure_engine(executor, program, args, Path(folder))
    else:
        args = benchmark_arguments().parse_args(argv)
        measure_engine(executor, ExportInferenceEngine(Path(executor), task=""), args, Path(executor))


def measure_engine(executor: str, program, args: argparse.Namespace, folder: Path) -> None:
    """Time `program` on the first frame, then save its actions for every frame."""
    from safetensors.numpy import load_file

    info = json.loads((folder / "export.json").read_text())
    chunk_size = len(load_file(folder / info["test_case"])["expected_actions"])
    data = np.load(args.frames)
    frames = [{name: np.ascontiguousarray(data[name][i]) for name in data.files} for i in range(FRAMES)]

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
    times = np.array([[chunk_ms() for _ in range(CHUNKS_PER_RUN)] for _ in range(RUNS)])
    actions = np.stack([run_chunk(frame, seed) for seed, frame in enumerate(frames)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, chunk_ms=times, actions=actions)
    print(f"{executor}: {time_per_chunk(times):.1f} ms per chunk, wrote {args.output}")
    if executor == "torch.compile":
        from torch._dynamo.utils import counters

        # More graphs than parts means torch.compile broke a part into pieces and runs Python between them.
        print(
            f"torch.compile built {counters['stats']['unique_graphs']} graphs for {len(info['programs'])} parts"
        )


def report(paths: list[str]) -> None:
    """Print each file's time per chunk and its action error against the first file's actions."""
    reference = np.load(paths[0])["actions"]
    print(f"{'':40} {'ms per chunk':>12} {'mean error':>11} {'p95 error':>10}")
    for path in paths:
        result = np.load(path)
        mean, p95 = action_error(result["actions"], reference)
        print(f"{Path(path).stem:40} {time_per_chunk(result['chunk_ms']):12.1f} {mean:11.3f} {p95:10.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
    elif sys.argv[1] == "frames":
        write_frames(sys.argv[2:])
    elif sys.argv[1] == "report":
        report(sys.argv[2:])
    else:
        measure(sys.argv[1], sys.argv[2:])
