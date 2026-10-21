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

"""Measure the memory a policy takes on a Jetson while `benchmark.py --duration` runs it.

The run's memory is what its processes hold that the board cannot take back without swapping: their own
memory and what they have swapped out, from /proc/<pid>/smaps_rollup, plus their GPU memory. Each process
counts its proportional share of a page it shares with others. Library and model files they read stay in the
page cache, which the kernel can drop, so they are not counted. The result is the median over the window,
in MiB.

A Jetson's CPU and GPU share one memory, but a process's own counters miss its GPU memory, so that comes
from the GPU driver: from `nvidia-smi` on Thor, and on an Orin from jtop (`pip install jetson-stats`), whose
service reads the driver's per-process counts, which only root can read. The board's free memory cannot stand
in for it on an Orin: the driver keeps the memory a finished GPU job freed, up to an eighth of the board, for
the next one, so a run that reuses it would look smaller by an amount that depends on what ran before.

The window is the last 30 s before the benchmark says it stopped, before its teardown frees everything, and
at least 60 s after it says its warm-up chunks are done, so loading and compiling stay out of it: a process
can still give back memory from loading well after its first warm chunks. A run whose window is still
changing is reported as invalid, not as a number.

    python examples/export/benchmark_memory.py -- python examples/export/benchmark.py outputs/export/act \\
        --frames=outputs/benchmark/frames.npz --duration=90
"""

import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from benchmark import STOPPED, WARM

SAMPLE_SECONDS = 0.5
STEADY_SECONDS = 30
# ACT in PyTorch gave back about 90 MiB from loading 40 s after its warm-up chunks, then stayed flat.
SETTLE_SECONDS = 60
FLAT_MIB_PER_SECOND = 2
# Orin and older Jetsons run their GPU on the nvgpu driver, whose memory `nvidia-smi` does not report.
NVGPU = Path("/sys/module/nvgpu")


def own_mib(smaps_rollup: str) -> float:
    """A process's anonymous and shared memory plus what it swapped out, from its smaps_rollup, in MiB."""
    fields = {line.split(":")[0]: int(line.split()[1]) for line in smaps_rollup.splitlines()[1:]}
    return (fields["Pss_Anon"] + fields["Pss_Shmem"] + fields["SwapPss"]) / 1024


def nvidia_smi_gpu_mib() -> dict[int, float]:
    """GPU memory per process ID as `nvidia-smi` reports it, in MiB."""
    table = subprocess.run(  # nosec B607
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {int(pid): float(mib) for pid, mib in (line.split(",") for line in table.splitlines())}


def process_tree(root: int) -> list[int]:
    """`root` and every process it started, directly or through others."""
    parents = {}
    for entry in Path("/proc").iterdir():
        try:
            parents[int(entry.name)] = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (ValueError, OSError):
            continue
    tree = [root]
    for pid in tree:
        tree += [child for child, parent in parents.items() if parent == pid]
    return tree


def process_mib(root: int, gpu: dict[int, float]) -> float:
    """The memory of `root` and the processes it started, with their GPU memory from `gpu`, in MiB."""
    total = 0.0
    for pid in process_tree(root):
        try:
            total += own_mib(Path(f"/proc/{pid}/smaps_rollup").read_text()) + gpu.get(pid, 0.0)
        except OSError:
            continue
    return total


def steady_window(run: list[tuple[float, float]], marks: dict[str, float]) -> list[float]:
    """The memory sampled in the last 30 s before the benchmark's stopped line, from (time, memory) samples."""
    return [mib for at, mib in run if marks[STOPPED] - STEADY_SECONDS <= at < marks[STOPPED]]


def is_flat(samples: list[float]) -> bool:
    """Whether memory use, sampled every half second, changes by less than 2 MiB/s on a fitted line.

    Allocator jitter of tens of MiB moves the fitted slope by a fraction of 1 MiB/s, while memory still being
    returned after loading falls several MiB a second.
    """
    times = [i * SAMPLE_SECONDS for i in range(len(samples))]
    return abs(statistics.linear_regression(times, samples).slope) < FLAT_MIB_PER_SECOND


def sample(process: subprocess.Popen, gpu_mib: Callable[[], dict[int, float]]) -> list[tuple[float, float]]:
    """(time, memory of `process` and the processes it started) every half second until it exits."""
    samples = []
    while process.poll() is None:
        samples.append((time.monotonic(), process_mib(process.pid, gpu_mib())))
        time.sleep(SAMPLE_SECONDS)
    return samples


def watch(process: subprocess.Popen, marks: dict[str, float]) -> None:
    """Pass the command's output through, and note when it prints the benchmark's warm and stopped lines."""
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.strip() in (WARM, STOPPED):
            marks[line.strip()] = time.monotonic()


def run(command: list[str], gpu_mib: Callable[[], dict[int, float]]) -> list[float]:
    """Run `command` and return its memory over the window."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    marks: dict[str, float] = {}
    watcher = threading.Thread(target=watch, args=(process, marks))
    watcher.start()
    samples = sample(process, gpu_mib)
    watcher.join()
    if process.returncode != 0:
        raise SystemExit(f"{command[0]} exited with {process.returncode}")
    if WARM not in marks or STOPPED not in marks:
        raise SystemExit(
            f"The command printed no {WARM!r} and {STOPPED!r} lines: run benchmark.py --duration."
        )
    if marks[STOPPED] - marks[WARM] < SETTLE_SECONDS + STEADY_SECONDS:
        raise SystemExit(
            f"Run for at least {SETTLE_SECONDS + STEADY_SECONDS} s after the warm-up, with --duration."
        )
    return steady_window(samples, marks)


def main() -> None:
    if "--" not in sys.argv:
        print(__doc__)
        return
    command = sys.argv[sys.argv.index("--") + 1 :]
    if NVGPU.is_dir():
        from jtop import jtop

        with jtop(interval=SAMPLE_SECONDS) as jetson:
            # A process row lists its ID first and its GPU memory, in KiB, ninth.
            steady = run(command, lambda: {int(row[0]): row[8] / 1024 for row in jetson.processes})
    else:
        steady = run(command, nvidia_smi_gpu_mib)
    if not is_flat(steady):
        raise SystemExit("Invalid: memory use was still changing while the policy ran warm.")
    print(f"Memory: {statistics.median(steady):.0f} MiB")


if __name__ == "__main__":
    main()
