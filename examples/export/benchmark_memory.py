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

A Jetson's CPU and GPU share one memory, and a process's own counters miss some GPU allocations, so this
reads the whole board from /proc/meminfo: the drop in available memory plus the growth in swap, from the
10 s before the command starts to the time the policy runs warm, in MiB. Keep the board otherwise idle.

The board keeps returning memory for a while after a GPU job ends, so the baseline waits until memory
use is flat. The run's window opens when the benchmark says its warm-up chunks are done, so loading and
compiling stay out of it, and closes when it says it stopped, before its teardown frees everything. A run
whose window is not flat by the same test is reported as invalid, not as a number.

    python examples/export/benchmark_memory.py -- python examples/export/benchmark.py outputs/export/act \\
        --frames=outputs/benchmark/frames.npz --duration=60
"""

import statistics
import subprocess
import sys
import threading
import time

from benchmark import STOPPED, WARM

SAMPLE_SECONDS = 0.5
BASELINE_SECONDS = 10
STEADY_SECONDS = 30
FLAT_MIB_PER_SECOND = 2
SETTLE_LIMIT_SECONDS = 300


def used_mib(meminfo: dict[str, int]) -> float:
    """Memory in use as the measure counts it: what is not available, plus what is swapped out, in MiB."""
    return (meminfo["MemTotal"] - meminfo["MemAvailable"] + meminfo["SwapTotal"] - meminfo["SwapFree"]) / 1024


def steady_window(run: list[tuple[float, float]], marks: dict[str, float]) -> list[float]:
    """The use sampled between the benchmark's warm and stopped lines, from (time, use) samples."""
    return [use for at, use in run if marks[WARM] <= at <= marks[STOPPED]]


def extra_mib(baseline: list[float], window: list[float]) -> float:
    """The median use over the run's steady window minus the median use before it started."""
    return statistics.median(window) - statistics.median(baseline)


def is_flat(samples: list[float]) -> bool:
    """Whether memory use, sampled every half second, changes by less than 2 MiB/s on a fitted line.

    Allocator jitter of tens of MiB moves the fitted slope by a fraction of 1 MiB/s, while memory still being
    returned after loading falls several MiB a second.
    """
    times = [i * SAMPLE_SECONDS for i in range(len(samples))]
    return abs(statistics.linear_regression(times, samples).slope) < FLAT_MIB_PER_SECOND


def read_meminfo() -> dict[str, int]:
    """/proc/meminfo's values, in KiB."""
    with open("/proc/meminfo") as file:
        return {line.split(":")[0]: int(line.split()[1]) for line in file}


def sample(
    seconds: float | None = None, process: subprocess.Popen | None = None
) -> list[tuple[float, float]]:
    """(time, memory use) every half second, for `seconds` or until `process` exits."""
    samples = []
    end = time.monotonic() + (seconds or 0)
    while (process.poll() is None) if process else time.monotonic() < end:
        samples.append((time.monotonic(), used_mib(read_meminfo())))
        time.sleep(SAMPLE_SECONDS)
    return samples


def watch(process: subprocess.Popen, marks: dict[str, float]) -> None:
    """Pass the command's output through, and note when it prints the benchmark's warm and stopped lines."""
    for line in process.stdout:
        print(line, end="", flush=True)
        if line.strip() in (WARM, STOPPED):
            marks[line.strip()] = time.monotonic()


def main() -> None:
    if "--" not in sys.argv:
        print(__doc__)
        return
    command = sys.argv[sys.argv.index("--") + 1 :]
    window = int(BASELINE_SECONDS / SAMPLE_SECONDS)
    baseline = [use for _, use in sample(BASELINE_SECONDS)]
    settle_end = time.monotonic() + SETTLE_LIMIT_SECONDS
    while not is_flat(baseline[-window:]):
        if time.monotonic() > settle_end:
            raise SystemExit(f"Invalid: memory use was still changing after {SETTLE_LIMIT_SECONDS} s.")
        baseline += [use for _, use in sample(SAMPLE_SECONDS)]
    baseline = baseline[-window:]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    marks: dict[str, float] = {}
    watcher = threading.Thread(target=watch, args=(process, marks))
    watcher.start()
    run = sample(process=process)
    watcher.join()
    if process.returncode != 0:
        raise SystemExit(f"{command[0]} exited with {process.returncode}")
    if WARM not in marks or STOPPED not in marks:
        raise SystemExit(
            f"The command printed no {WARM!r} and {STOPPED!r} lines: run benchmark.py --duration."
        )
    if marks[STOPPED] - marks[WARM] < STEADY_SECONDS:
        raise SystemExit(f"Run for at least {STEADY_SECONDS} s after the warm-up, with --duration.")
    steady = steady_window(run, marks)
    if not is_flat(steady):
        raise SystemExit("Invalid: memory use was still changing while the policy ran warm.")
    print(f"Extra memory: {extra_mib(baseline, steady):.0f} MiB")


if __name__ == "__main__":
    main()
