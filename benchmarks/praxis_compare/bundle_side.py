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

"""The exported bundle through LeRobot's torch-free runtime (`BundleEngine`, what `lerobot-rollout --bundle.path`
runs), on the recorded sequence. PyTorch is blocked: importing it raises, and the run fails if it was loaded.

usage: python bundle_side.py <bundle dir> <seq.npz> <out.npz> [--warmup 3] [--count 30] [--chunks N]

Every control tick hands the engine one frame: the two raw 288x512 uint8 views and the DROID 7-value state, as
NumPy arrays on the host. On the tick whose action queue is empty, `get_action` runs the bundle's programs for a
new chunk. TIMER (chunk ticks only): `time.perf_counter()` right before `get_action(frame)` and right after it
returns. Inside: the history copy of the frame, the stack of the kept frames, their host-to-device copies, the
noise draw, encode, start, sample, actions, and the copy of the actions back into NumPy, which waits for the GPU.
The GPU is idle when the timer starts: the ticks between chunks run nothing on it.
"""

import importlib.abc
import importlib.machinery
import json
import sys


class _BlockTorch(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    attempts: list[str] = []

    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            return importlib.machinery.ModuleSpec(name, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.attempts.append(module.__name__)
        raise ImportError("torch is blocked in this run")


sys.meta_path.insert(0, _BlockTorch())

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from lerobot.bundle.engine import BundleEngine  # noqa: E402
from lerobot.bundle.manifest import Bundle  # noqa: E402

# The sampler settings every side records, so that summarize.py can refuse passes that ran different ones.
SAMPLER = ("num_steps", "t_shift", "cfg_scale", "cfg_video_scale")


def torch_report() -> dict:
    maps = Path("/proc/self/maps").read_text() if Path("/proc/self/maps").exists() else ""
    mapped = sorted({ln.split()[-1] for ln in maps.splitlines() if "libtorch" in ln or "libc10" in ln})
    modules = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
    return {"torch_modules": modules, "libtorch_mapped": mapped, "import_attempts": _BlockTorch.attempts}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bundle")
    p.add_argument("seq")
    p.add_argument("out")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--count", type=int, default=30)
    p.add_argument(
        "--chunks", type=int, default=0, help="stop after this many chunks (0: the whole sequence)"
    )
    a = p.parse_args()
    seq = dict(np.load(a.seq))
    chunk_ticks = [int(t) for t in seq["chunk_ticks"]]
    if a.chunks:
        chunk_ticks = chunk_ticks[: a.chunks]
    config = json.loads((Path(a.bundle) / "policy_config.json").read_text())
    sampler = {key: float(config[key]) for key in SAMPLER}
    t0 = time.perf_counter()
    engine = BundleEngine(Bundle.read(Path(a.bundle)), task=str(seq["task"]))
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.1f} s, task {engine.task!r}, chunk_length {engine.chunk_length}", flush=True)
    engine.reset()
    times, chunks = [], []
    for tick in range(chunk_ticks[-1] + 1):
        i = int(seq["index"][tick])
        frame = {
            "observation.images.exterior": seq["exterior"][i],
            "observation.images.wrist": seq["wrist"][i],
            "observation.state": seq["state7"][tick],
        }
        if tick not in chunk_ticks:
            assert engine._actions, f"tick {tick}: the action queue ran out off schedule"
            engine.get_action(frame)
            continue
        assert not engine._actions, f"tick {tick}: the action queue is not empty at a chunk tick"
        s = time.perf_counter()
        first = engine.get_action(frame)
        e = time.perf_counter()
        chunk = np.stack([first, *engine._actions])
        times.append((e - s) * 1e3)
        chunks.append(chunk)
        print(json.dumps({"chunk": len(times) - 1, "tick": tick, "ms": round(times[-1], 2)}), flush=True)
    timed = times[a.warmup : a.warmup + a.count]
    report = torch_report()
    summary = {
        "side": "bundle",
        "load_s": round(load_s, 1),
        "warmup_ms": [round(x, 1) for x in times[: a.warmup]],
        "median_ms": round(float(np.median(timed)), 2) if timed else None,
        "p90_ms": round(float(np.percentile(timed, 90)), 2) if timed else None,
        "n_timed": len(timed),
        "sampler": sampler,
        **report,
    }
    width = max(len(c) for c in chunks)
    padded = np.full((len(chunks), width, chunks[0].shape[1]), np.nan, np.float32)
    for k, c in enumerate(chunks):
        padded[k, : len(c)] = c
    np.savez(a.out, actions=padded, rows=np.array([len(c) for c in chunks]), chunk_ms=np.array(times))
    print("SUMMARY " + json.dumps(summary), flush=True)
    if any(report.values()):
        sys.exit("PyTorch was loaded or imported in a torch-free run")
    if len(timed) != a.count:
        sys.exit(f"timed {len(timed)} chunks, not --count {a.count}")


if __name__ == "__main__":
    main()
