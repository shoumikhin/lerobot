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

"""The arithmetic behind the README's results: time per chunk, action error and board memory."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from tests.test_export_recipes import (
    act_checkpoint,
    dataset_metadata,
    pi05_checkpoint,
    smolvla_checkpoint,
    tiny_smolvlm,
)
from tests.utils import require_hf_token

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "export"


@pytest.fixture
def examples(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    return importlib.import_module


def test_time_per_chunk_is_the_median_of_the_run_medians(examples):
    benchmark = examples("benchmark")
    # A slow outlier chunk moves no median; the slow run is outvoted by the other two.
    chunk_ms = np.array([[1.0, 2.0, 100.0], [3.0, 3.0, 3.0], [50.0, 60.0, 70.0]])
    assert benchmark.time_per_chunk(chunk_ms) == 3.0


def test_action_error_takes_each_frames_largest_difference(examples):
    benchmark = examples("benchmark")
    reference = np.zeros((20, 2, 3))
    actions = reference.copy()
    actions[:, 1, 2] = 0.5  # every frame is off by 0.5 on one joint of one action
    actions[7, 0, 0] = 10.0  # and one frame by 10
    mean, p95, largest = benchmark.action_error(actions, reference)
    assert mean == pytest.approx((19 * 0.5 + 10.0) / 20)
    assert p95 == pytest.approx(0.5 + 0.05 * (10.0 - 0.5))
    assert largest == 10.0


def test_action_error_refuses_actions_of_another_shape(examples):
    benchmark = examples("benchmark")
    # NumPy would broadcast one action against a chunk of 100 and find no error at all.
    with pytest.raises(ValueError, match="cannot be compared"):
        benchmark.action_error(np.zeros((50, 1, 6)), np.zeros((50, 100, 6)))


def test_reference_runs_the_original_policy_with_the_engine_noise(examples, tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    benchmark = examples("benchmark")
    for name, value in (("FRAMES", 3), ("WARMUP_CHUNKS", 1), ("RUNS", 1), ("CHUNKS_PER_RUN", 1)):
        monkeypatch.setattr(benchmark, name, value)
    import json

    import torch
    from transformers import AutoConfig, SmolVLMForConditionalGeneration

    vlm = tiny_smolvlm(tmp_path / "vlm")
    checkpoint = smolvla_checkpoint(tmp_path / "checkpoint", vlm)
    # As a real checkpoint does: the policy loads its vision-language model's weights, in bfloat16.
    SmolVLMForConditionalGeneration(AutoConfig.from_pretrained(vlm)).save_pretrained(vlm)
    config = json.loads((checkpoint / "config.json").read_text())
    (checkpoint / "config.json").write_text(json.dumps({**config, "load_vlm_weights": True}))
    features = {
        "observation.images.front": {
            "dtype": "video",
            "shape": (32, 32, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (3,), "names": None},
        "action": {"dtype": "float32", "shape": (3,), "names": None},
    }
    metadata = dataset_metadata(features, ["pick the cup"])
    monkeypatch.setattr(examples("common"), "LeRobotDatasetMetadata", lambda repo_id, root=None: metadata)
    rng = np.random.default_rng(0)
    frames = {
        "observation.images.front": rng.integers(0, 256, (3, 32, 32, 3), dtype=np.uint8),
        "observation.state": rng.uniform(-1, 1, (3, 3)).astype(np.float32),
        # A camera the dataset has and the export leaves out.
        "observation.images.wrist": rng.integers(0, 256, (3, 32, 32, 3), dtype=np.uint8),
    }
    np.savez(tmp_path / "frames.npz", **frames)
    folder = tmp_path / "folder"
    folder.mkdir()
    frames, output = f"--frames={tmp_path / 'frames.npz'}", tmp_path / "pytorch.npz"
    argv = [f"--policy.path={checkpoint}", "--dataset.repo_id=user/dataset", frames, f"--output={output}"]
    engine, args = benchmark.torch_engine("pytorch", argv, folder)
    benchmark.measure_engine("pytorch", engine, args, folder)
    reference_file = tmp_path / "reference.npz"
    from lerobot.policies import utils
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    def one_action_at_a_time(self, batch, **kwargs):
        raise NotImplementedError

    # As GR00T with relative actions refuses it: the reference asks for whole chunks.
    monkeypatch.setattr(SmolVLAPolicy, "select_action", one_action_at_a_time)

    observed, ran = [], []
    prepare = utils.prepare_observation_for_inference
    monkeypatch.setattr(
        utils,
        "prepare_observation_for_inference",
        lambda frame, device, task: observed.append((set(frame), task)) or prepare(frame, device, task),
    )
    predict = SmolVLAPolicy.predict_action_chunk

    def predict_in_float32(self, batch, **kwargs):
        dtypes = {parameter.dtype for parameter in self.parameters()}
        ran.append((dtypes, torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32))
        return predict(self, batch, **kwargs)

    monkeypatch.setattr(SmolVLAPolicy, "predict_action_chunk", predict_in_float32)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    benchmark.write_reference(
        [str(folder), f"--policy.path={checkpoint}", frames, f"--output={reference_file}"]
    )
    actions, reference = np.load(output)["actions"], np.load(reference_file)["reference"]
    assert reference.shape == actions.shape == (3, 2, 3)
    # The recipe keeps the bfloat16 weights SmolVLA loads, the reference runs in float32: a small difference.
    # Other noise or frames would move the actions as much as another frame does, by whole units here.
    assert np.abs(actions - reference).max() < 0.05
    assert np.abs(np.diff(reference, axis=0)).max() > 1
    assert observed == [({"observation.images.front", "observation.state"}, "pick the cup")] * 3
    # Neither bfloat16 weights nor TensorFloat-32.
    assert ran == [({torch.float32}, "highest", False)] * 3


@require_hf_token
def test_reference_loads_the_checkpoint_in_float32_whatever_dtype_it_names(examples, tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    import json

    import torch
    from safetensors.numpy import save_file

    benchmark = examples("benchmark")
    monkeypatch.setattr(benchmark, "FRAMES", 1)
    checkpoint, config = pi05_checkpoint(tmp_path / "checkpoint", monkeypatch)
    folder = tmp_path / "folder"
    folder.mkdir()
    camera = "observation.images.base_0_rgb"
    info = {
        "inputs": [camera, "observation.state", "noise"],
        "noise_shape": [config.chunk_size, config.max_action_dim],
        "task": "pick up the cube",
        "test_case": "test_case.safetensors",
    }
    (folder / "export.json").write_text(json.dumps(info))
    save_file(
        {"expected_actions": np.zeros((config.n_action_steps, 3), np.float32)}, folder / info["test_case"]
    )
    frames = {
        camera: np.full((1, 28, 28, 3), 127, np.uint8),
        "observation.state": np.zeros((1, 3), np.float32),
    }
    np.savez(tmp_path / "frames.npz", **frames)
    references = []
    for dtype in (torch.float32, torch.bfloat16):
        config.dtype = dtype
        config.save_pretrained(checkpoint)
        output = tmp_path / f"reference_{dtype}.npz"
        argv = [str(folder), f"--policy.path={checkpoint}", f"--frames={tmp_path / 'frames.npz'}"]
        benchmark.write_reference([*argv, f"--output={output}"])
        references.append(np.load(output)["reference"])
    # Float32 weights loaded into a bfloat16 policy are rounded, and casting it to float32 later cannot undo that.
    np.testing.assert_array_equal(*references)


def act_pytorch_run(benchmark, tmp_path, monkeypatch) -> tuple[Path, Path, str]:
    """`benchmark.py pytorch` on a tiny ACT and 50 frames; its checkpoint, export folder and frames argument."""
    for name, value in (("WARMUP_CHUNKS", 1), ("RUNS", 1), ("CHUNKS_PER_RUN", 1)):
        monkeypatch.setattr(benchmark, name, value)
    checkpoint = act_checkpoint(tmp_path / "checkpoint")
    rng = np.random.default_rng(0)
    np.savez(
        tmp_path / "frames.npz",
        **{
            "observation.images.front": rng.integers(0, 256, (benchmark.FRAMES, 32, 32, 3), dtype=np.uint8),
            "observation.state": rng.uniform(-1, 1, (benchmark.FRAMES, 3)).astype(np.float32),
        },
    )
    folder, frames = tmp_path / "folder", f"--frames={tmp_path / 'frames.npz'}"
    folder.mkdir()
    argv = [f"--policy.path={checkpoint}", frames, f"--output={tmp_path / 'pytorch.npz'}"]
    engine, args = benchmark.torch_engine("pytorch", argv, folder)
    benchmark.measure_engine("pytorch", engine, args, folder)
    return checkpoint, folder, frames


def test_reference_and_report_commands_compare_a_result_with_the_policy(examples, tmp_path, monkeypatch):
    import subprocess
    import sys

    benchmark = examples("benchmark")
    checkpoint, folder, frames = act_pytorch_run(benchmark, tmp_path, monkeypatch)
    script = [sys.executable, str(EXAMPLES / "benchmark.py")]
    reference = tmp_path / "reference.npz"
    subprocess.run(
        [*script, "reference", str(folder), f"--policy.path={checkpoint}", frames, f"--output={reference}"],
        check=True,
    )
    report = subprocess.run(
        [*script, "report", str(reference), str(tmp_path / "pytorch.npz")],
        check=True,
        capture_output=True,
        text=True,
    )
    # On a GPU the PyTorch run uses TensorFloat-32, which moves ACT's actions by about 0.002 here.
    assert max(float(error) for error in report.stdout.split()[-3:]) < 0.01


def test_report_compares_each_result_with_the_reference_file(examples, tmp_path, capsys):
    benchmark = examples("benchmark")
    reference = np.zeros((4, 2, 3))
    actions = reference.copy()
    actions[:, 0, 0] = 0.5
    np.savez(tmp_path / "reference.npz", reference=reference)
    np.savez(tmp_path / "pytorch.npz", chunk_ms=np.full((3, 5), 7.0), actions=actions)
    benchmark.report([str(tmp_path / "reference.npz"), str(tmp_path / "pytorch.npz")])
    assert capsys.readouterr().out.splitlines()[1:] == [
        f"{'pytorch':40} {7.0:12.1f} {0.5:11.3f} {0.5:10.3f} {0.5:10.3f}"
    ]
    with pytest.raises(SystemExit, match="holds no reference actions"):
        benchmark.report([str(tmp_path / "pytorch.npz"), str(tmp_path / "pytorch.npz")])


def test_memory_counts_a_process_own_memory_and_swap_but_not_its_files(examples):
    memory = examples("benchmark_memory")
    rollup = "\n".join(
        [
            "00400000-ffffd0a1c000 ---p 00000000 00:00 0                  [rollup]",
            "Rss:              457620 kB",
            "Pss:              445068 kB",
            "Pss_Anon:         323340 kB",
            "Pss_File:         121728 kB",
            "Pss_Shmem:          1024 kB",
            "Swap:               4096 kB",
            "SwapPss:            2048 kB",
        ]
    )
    assert memory.own_mib(rollup) == (323340 + 1024 + 2048) / 1024


def test_memory_reads_gpu_memory_per_process_from_nvidia_smi(examples, monkeypatch):
    from types import SimpleNamespace

    memory = examples("benchmark_memory")
    table = "4242, 1140\n4343, 12\n"
    monkeypatch.setattr(memory.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=table))
    assert memory.nvidia_smi_gpu_mib() == {4242: 1140.0, 4343: 12.0}


def test_memory_sums_each_process_gpu_memory_over_its_jtop_rows(examples):
    memory = examples("benchmark_memory")
    # jtop rows: process ID, user, GPU, type, priority, state, CPU %, memory and GPU memory in KiB, name.
    rows = [
        [4242, "user", "I", "G", 0, "S", 10.0, 900000, 102400, "python3"],
        [4242, "user", "I", "G", 0, "S", 10.0, 900000, 51200, "python3"],
        [4343, "user", "I", "G", 0, "S", 1.0, 20000, 1024, "nvargus"],
    ]
    assert memory.jtop_gpu_mib(rows) == {4242: 150.0, 4343: 1.0}


def test_memory_times_a_sample_when_its_read_ends(examples, monkeypatch):
    from types import SimpleNamespace

    memory = examples("benchmark_memory")
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        memory, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=lambda seconds: None)
    )

    def slow_read(pid, gpu):
        clock.now += 1.0
        return 3000.0

    monkeypatch.setattr(memory, "process_mib", slow_read)
    process = SimpleNamespace(pid=1, poll=iter([None, None, 0]).__next__)
    # Timed when it starts, a read that ends after the stopped line would be counted as before it.
    assert memory.sample(process, lambda: {}) == [(1.0, 3000.0), (2.0, 3000.0)]


def test_memory_counts_only_the_warm_chunks(examples):
    memory = examples("benchmark_memory")
    # Loading peaks, compiling holds a long flat plateau, the warm chunks still give back some memory from
    # loading, then hold steady, and the teardown frees everything.
    use = [9000.0] * 10 + [7000.0] * 120 + [3100.0] * 80 + [3000.0] * 100 + [500.0] * 6
    run = [(i * memory.SAMPLE_SECONDS, u) for i, u in enumerate(use)]
    marks = {memory.WARM: 130 * memory.SAMPLE_SECONDS, memory.STOPPED: 310 * memory.SAMPLE_SECONDS}
    steady = [mib for _, mib in memory.steady_window(run, marks)]
    assert steady == [3000.0] * int(memory.STEADY_SECONDS / memory.SAMPLE_SECONDS)


def run_benchmark_memory(memory, monkeypatch, process_mib, command: str) -> None:
    """Run benchmark_memory.py on a Python `command`, with a 0.3 s window after 0.1 s to settle."""
    import sys

    monkeypatch.setattr(memory, "SAMPLE_SECONDS", 0.02)
    monkeypatch.setattr(memory, "STEADY_SECONDS", 0.3)
    monkeypatch.setattr(memory, "SETTLE_SECONDS", 0.1)
    monkeypatch.setattr(memory, "NVGPU", Path("/missing"))
    monkeypatch.setattr(memory, "nvidia_smi_gpu_mib", lambda: {})
    monkeypatch.setattr(memory, "process_mib", process_mib)
    monkeypatch.setattr(sys, "argv", ["benchmark_memory.py", "--", sys.executable, "-c", command])
    memory.main()


def test_memory_measures_a_command_between_its_warm_and_stopped_lines(
    examples, monkeypatch, capsys, tmp_path
):
    memory = examples("benchmark_memory")
    warm = tmp_path / "warm"
    # 9000 MiB while the command loads, then 1500 MiB once it runs warm, however long each step takes.
    command = (
        f"import pathlib, time; time.sleep(0.2); pathlib.Path({str(warm)!r}).touch();"
        f"print({memory.WARM!r}, flush=True); time.sleep(1.0); print({memory.STOPPED!r}, flush=True)"
    )
    run_benchmark_memory(memory, monkeypatch, lambda pid, gpu: 1500 if warm.exists() else 9000, command)
    assert capsys.readouterr().out.splitlines()[-1] == "Memory: 1500 MiB"


def test_memory_refuses_a_run_whose_memory_still_changes(examples, monkeypatch):
    import itertools

    memory = examples("benchmark_memory")
    reads = itertools.count()
    command = f"import time; print({memory.WARM!r}, flush=True); time.sleep(1.0); print({memory.STOPPED!r})"
    with pytest.raises(SystemExit, match="Invalid"):
        run_benchmark_memory(memory, monkeypatch, lambda pid, gpu: 1000 + 50 * next(reads), command)


def test_memory_refuses_a_run_too_short_to_settle(examples, monkeypatch):
    import sys

    memory = examples("benchmark_memory")
    monkeypatch.setattr(memory, "process_mib", lambda pid, gpu: 1500)
    command = f"print({memory.WARM!r}); print({memory.STOPPED!r})"
    with pytest.raises(SystemExit, match="Run for at least 90 s"):
        memory.run([sys.executable, "-c", command], lambda: {})


@pytest.mark.skipif(not Path("/proc/self/smaps_rollup").exists(), reason="needs Linux's /proc")
def test_memory_counts_what_a_child_process_allocates(examples):
    import subprocess
    import sys

    memory = examples("benchmark_memory")
    sizes = []
    for mib in (10, 210):
        # A parent that starts a child, which touches every page of `mib` MiB and waits for its input to close.
        child = f"data = bytearray({mib << 20}); data[::4096] = b'1' * {(mib << 20) // 4096}; print(1, flush=True); input()"
        parent = f"import subprocess, sys; subprocess.run([sys.executable, '-c', {child!r}])"
        process = subprocess.Popen(
            [sys.executable, "-c", parent], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        process.stdout.readline()
        sizes.append(memory.process_mib(process.pid, {}))
        process.stdin.close()
        process.wait()
    assert sizes[1] - sizes[0] == pytest.approx(200, abs=10)


@pytest.mark.skipif(not Path("/proc/self/smaps_rollup").exists(), reason="needs Linux's /proc")
def test_memory_adds_the_gpu_memory_of_every_process_it_counts(examples):
    import os
    import subprocess
    import sys

    memory = examples("benchmark_memory")
    child = "print(1, flush=True); input()"
    parent = f"import subprocess, sys; subprocess.run([sys.executable, '-c', {child!r}])"
    process = subprocess.Popen(
        [sys.executable, "-c", parent], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    process.stdout.readline()
    root, child_pid = memory.process_tree(process.pid)
    # This test's own process is not one the command started.
    gpu = {root: 100.0, child_pid: 200.0, os.getpid(): 5000.0}
    added = memory.process_mib(process.pid, gpu) - memory.process_mib(process.pid, {})
    process.stdin.close()
    process.wait()
    assert added == pytest.approx(300, abs=1)


def test_frames_share_one_copy_of_the_file(examples, tmp_path):
    benchmark = examples("benchmark")
    np.savez(tmp_path / "frames.npz", image=np.zeros((benchmark.FRAMES, 4, 4, 3), np.uint8))
    frames = benchmark.load_frames(tmp_path / "frames.npz")
    # A frame per copy of all the frames would hold the file 50 times over in memory.
    assert len({id(frame["image"].base) for frame in frames}) == 1


def test_memory_is_flat_only_when_it_stops_changing(examples):
    memory = examples("benchmark_memory")
    times = [i * memory.SAMPLE_SECONDS for i in range(int(memory.STEADY_SECONDS / memory.SAMPLE_SECONDS))]
    # A settled process whose allocator moves its use up and down by 30 MiB.
    jitter = np.random.default_rng(0).uniform(-30, 30, len(times))
    assert memory.is_flat(list(zip(times, 4000.0 + jitter, strict=True)))
    # The board still returning a previous job's memory: use falling 7 MiB a second.
    assert not memory.is_flat([(t, 3000.0 - 7 * t) for t in times])


def test_memory_is_flat_measures_its_slope_on_the_samples_real_times(examples):
    memory = examples("benchmark_memory")
    # Each read takes time, so the samples are further apart than the time slept between them.
    reads = np.random.default_rng(0).uniform(0.1, 0.5, 60)
    times = np.cumsum(memory.SAMPLE_SECONDS + reads).tolist()
    assert memory.is_flat([(t, 4000.0 - 1.5 * t) for t in times])
    assert not memory.is_flat([(t, 4000.0 - 3 * t) for t in times])


def test_torch_compile_replays_cuda_graphs_one_generation_per_chunk(examples, tmp_path, monkeypatch):
    import torch

    benchmark = examples("benchmark")
    compiled, generations = [], []
    monkeypatch.setattr(torch, "compile", lambda module, **options: compiled.append(options) or module)
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: generations.append(1))
    argv = [f"--policy.path={act_checkpoint(tmp_path / 'act')}", "--frames=unused", "--output=unused"]
    engine, _ = benchmark.torch_engine("torch.compile", argv, tmp_path)
    assert compiled == [{"mode": "reduce-overhead"}]
    generations.clear()
    frame = {
        "observation.images.front": np.zeros((32, 32, 3), np.uint8),
        "observation.state": np.zeros(3, np.float32),
    }
    for _ in range(4):  # two chunks of two actions
        engine.get_action(frame)
    assert len(generations) == 2


def test_torch_compile_starts_one_cuda_graph_generation_per_chunk_of_a_chain(examples, tmp_path, monkeypatch):
    import json
    from argparse import Namespace
    from types import SimpleNamespace

    import torch
    from safetensors.numpy import save_file

    benchmark = examples("benchmark")
    state = np.arange(3, dtype=np.float32)

    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class Chunk(torch.nn.Module):
        def forward(self, hidden):
            return torch.stack([hidden, 2 * hidden])

    class Exporter:
        """A recipe of two parts, the second reading the first's output."""

        def __init__(self, args):
            self.args = args

        def parts(self):
            example = (torch.from_numpy(state),)
            yield SimpleNamespace(
                name="first", module=AddOne(), inputs=example, input_names=["state"], output_names=["hidden"]
            )
            yield SimpleNamespace(
                name="second", module=Chunk(), inputs=example, input_names=["hidden"], output_names=["action"]
            )

        def write(self, programs):
            case = {"state": state, "expected_actions": np.stack([state + 1, 2 * state + 2])}
            save_file(case, self.args.output_dir / "case.safetensors")
            info = {
                "route": "fake",
                "programs": programs,
                "inputs": ["state"],
                "test_case": "case.safetensors",
            }
            (self.args.output_dir / "export.json").write_text(json.dumps({**info, "tolerance": 1e-6}))

    recipe = SimpleNamespace(Exporter=Exporter)
    monkeypatch.setattr(examples("common"), "parse_args", lambda *args: (Namespace(), recipe))
    monkeypatch.setattr(torch, "compile", lambda module, **options: module)
    generations = []
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: generations.append(1))
    engine, _ = benchmark.torch_engine("torch.compile", [], tmp_path)
    generations.clear()
    for _ in range(4):  # two chunks of two actions
        engine.get_action({"state": state})
    # A generation per part would let the first part's output, which the second part reads, be overwritten.
    assert len(generations) == 2


def test_torch_compile_compiles_in_the_benchmark_process(examples, tmp_path, monkeypatch):
    import torch

    benchmark = examples("benchmark")
    monkeypatch.setattr(torch._inductor.config, "compile_threads", 8)
    threads = []
    monkeypatch.setattr(
        torch,
        "compile",
        lambda module, **options: threads.append(torch._inductor.config.compile_threads) or module,
    )
    argv = [f"--policy.path={act_checkpoint(tmp_path / 'act')}", "--frames=unused", "--output=unused"]
    benchmark.torch_engine("torch.compile", argv, tmp_path)
    # More than one thread starts worker processes that stay alive while the memory is measured.
    assert threads == [1]


def test_duration_runs_warm_chunks_between_its_two_lines(examples, tmp_path, capsys):
    from argparse import Namespace

    from tests.export_helpers import write_export_folder

    benchmark = examples("benchmark")
    write_export_folder(tmp_path, np.zeros((2, 3)))
    np.savez(tmp_path / "frames.npz", **{"observation.state": np.zeros((benchmark.FRAMES, 3), np.float32)})
    chunks = []

    class Program:
        def reset(self):
            pass

        def get_action(self, frame):
            chunks.append(capsys.readouterr().out)

    args = Namespace(frames=tmp_path / "frames.npz", output=None, duration=0.2)
    benchmark.measure_engine("folder", Program(), args, tmp_path)
    # The warm line comes only after every warm-up chunk, the stopped line after the last chunk.
    assert chunks[: benchmark.WARMUP_CHUNKS] == [""] * benchmark.WARMUP_CHUNKS
    assert chunks[benchmark.WARMUP_CHUNKS] == benchmark.WARM + "\n"
    assert len(chunks) > benchmark.WARMUP_CHUNKS + 1
    assert capsys.readouterr().out == benchmark.STOPPED + "\n"


def test_duration_keeps_only_a_copy_of_the_first_frame(examples, tmp_path):
    from argparse import Namespace

    from tests.export_helpers import write_export_folder

    benchmark = examples("benchmark")
    write_export_folder(tmp_path, np.zeros((2, 3)))
    states = np.arange(benchmark.FRAMES * 3, dtype=np.float32).reshape(benchmark.FRAMES, 3)
    np.savez(tmp_path / "frames.npz", **{"observation.state": states})
    seen = []

    class Program:
        def reset(self):
            pass

        def get_action(self, frame):
            seen.append(frame["observation.state"])

    args = Namespace(frames=tmp_path / "frames.npz", output=None, duration=0.0)
    benchmark.measure_engine("folder", Program(), args, tmp_path)
    # A view of the first frame would keep every frame in memory while the run is measured.
    assert all(state.base is None for state in seen)
    np.testing.assert_array_equal(seen[-1], states[0])
