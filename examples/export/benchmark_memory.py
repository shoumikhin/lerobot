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

"""Measure the memory a `lerobot-rollout` run takes on a Jetson.

A Jetson's CPU and GPU share one memory, and a process's own counters miss some GPU allocations, so this
reads the whole board from /proc/meminfo: the drop in available memory plus the growth in swap, from the
10 s before the command starts to the last 30 s of its run, in MiB. Keep the board otherwise idle.

The board keeps returning memory for a while after a GPU job ends, so the baseline waits until memory
use is flat. The run's window ends 3 s before the command exits, since its teardown frees everything,
and a run whose window is not steady is reported as invalid, not as a number.

    python examples/export/benchmark_memory.py -- lerobot-rollout --policy.path=outputs/export/act \\
        --robot.type=replay --robot.frames_dir=outputs/replay_frames \\
        --duration=60 --return_to_initial_position=false --play_sounds=false
"""

import statistics
import subprocess
import sys
import time

SAMPLE_SECONDS = 0.5
BASELINE_SECONDS = 10
STEADY_SECONDS = 30
FLAT_MIB_PER_SECOND = 2
STEADY_MIB = 8
TEARDOWN_SECONDS = 3
SETTLE_LIMIT_SECONDS = 300


def used_mib(meminfo: dict[str, int]) -> float:
    """Memory in use as the measure counts it: what is not available, plus what is swapped out, in MiB."""
    return (meminfo["MemTotal"] - meminfo["MemAvailable"] + meminfo["SwapTotal"] - meminfo["SwapFree"]) / 1024


def steady_window(run: list[float]) -> list[float]:
    """The last 30 s of the run before the command's teardown."""
    return run[: -int(TEARDOWN_SECONDS / SAMPLE_SECONDS)][-int(STEADY_SECONDS / SAMPLE_SECONDS) :]


def extra_mib(baseline: list[float], run: list[float]) -> float:
    """The median use over the run's steady window minus the median use before it started."""
    return statistics.median(steady_window(run)) - statistics.median(baseline)


def long_enough(run: list[float]) -> bool:
    """Whether the command ran for 60 s, as `--duration=60` does, so the window before teardown is loop."""
    return len(run) * SAMPLE_SECONDS >= 2 * STEADY_SECONDS


def is_steady(samples: list[float]) -> bool:
    """Whether the medians of the first and second halves differ by less than 8 MiB, which one outlier cannot move."""
    half = len(samples) // 2
    return abs(statistics.median(samples[half:]) - statistics.median(samples[:half])) < STEADY_MIB


def is_flat(samples: list[float]) -> bool:
    """Whether memory use, sampled every half second, changes by less than 2 MiB/s on a fitted line."""
    times = [i * SAMPLE_SECONDS for i in range(len(samples))]
    return abs(statistics.linear_regression(times, samples).slope) < FLAT_MIB_PER_SECOND


def read_meminfo() -> dict[str, int]:
    """/proc/meminfo's values, in KiB."""
    with open("/proc/meminfo") as file:
        return {line.split(":")[0]: int(line.split()[1]) for line in file}


def sample(seconds: float | None = None, process: subprocess.Popen | None = None) -> list[float]:
    """Memory use every half second, for `seconds` or until `process` exits."""
    samples = []
    end = time.monotonic() + (seconds or 0)
    while (process.poll() is None) if process else time.monotonic() < end:
        samples.append(used_mib(read_meminfo()))
        time.sleep(SAMPLE_SECONDS)
    return samples


def main() -> None:
    if "--" not in sys.argv:
        print(__doc__)
        return
    command = sys.argv[sys.argv.index("--") + 1 :]
    window = int(BASELINE_SECONDS / SAMPLE_SECONDS)
    baseline = sample(BASELINE_SECONDS)
    settle_end = time.monotonic() + SETTLE_LIMIT_SECONDS
    while not is_flat(baseline[-window:]):
        if time.monotonic() > settle_end:
            raise SystemExit(f"Invalid: memory use was still changing after {SETTLE_LIMIT_SECONDS} s.")
        baseline += sample(SAMPLE_SECONDS)
    baseline = baseline[-window:]
    process = subprocess.Popen(command)
    run = sample(process=process)
    if process.returncode != 0:
        raise SystemExit(f"{command[0]} exited with {process.returncode}")
    if not long_enough(run):
        raise SystemExit(
            f"Run for at least {2 * STEADY_SECONDS} s, so the last {STEADY_SECONDS} s are steady."
        )
    if not is_steady(steady_window(run)):
        raise SystemExit(f"Invalid: memory use was still changing in the last {STEADY_SECONDS} s of the run.")
    print(f"Extra memory: {extra_mib(baseline, run):.0f} MiB")


if __name__ == "__main__":
    main()
