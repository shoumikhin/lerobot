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

"""Runway's own code (github.com/runwayml/praxis-1-oss) on the recorded sequence: `PraxisPipeline` with the
settings of `configs/serve_droid.yaml` and one `Session.step` per chunk, the call its policy server and
`scripts/benchmark_inference.py` make. Run it in Runway's environment; it does not import LeRobot.

usage: python runway_side.py <praxis-1-oss dir> <seq.npz> <out.npz> [--weights-dir DIR] [--checkpoint DIR]
       [--warmup 3] [--count 30] [--chunks N] [--client-resize] [--rotation PATH]

The script keeps what a robot client keeps: the frames and states of the last 25 ticks, as LeRobot's policy holds
them (1 on the cold call, 22 on the second, 25 after), and sends that stack on each chunk tick. TIMER (chunk ticks
only): `torch.cuda.synchronize()`, `time.perf_counter()`, then the stack of the kept frames (raw 288x512 uint8),
the DROID states to praxis pose rows (LeRobot's rule), `Session.step(obs)`, its commands to DROID 7-value rows
(LeRobot's rule), `torch.cuda.synchronize()`, `time.perf_counter()`. So both sides time the same thing: raw
observation in, DROID action chunk out.
--client-resize: each frame is fit-resized to Runway's tile once, at its own tick, outside the timer, as Runway's
client does before it sends a view. This is a lower bound for Runway: the bundle's timer still includes the resize.
--rotation: LeRobot's `lerobot/utils/rotation.py` (NumPy only), loaded by path so the DROID conversions are
LeRobot's own. Default: the one in this repository.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_rotation(path: str):
    spec = importlib.util.spec_from_file_location("lerobot_rotation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Rotation


def droid7_to_pose8(rotation, rows: np.ndarray) -> np.ndarray:
    """A copy of `droid7_to_pose8` in `lerobot/policies/praxis/processor_praxis.py`."""
    a = np.asarray(rows, dtype=np.float64)
    flat = a.reshape(-1, a.shape[-1])
    quat = rotation.matrix_to_quat_xyzw(rotation.euler_xyz_to_matrix(flat[:, 3:6]))[:, [3, 0, 1, 2]]
    grip = np.clip(flat[:, 6:7], 0.0, 1.0)
    out = np.concatenate([flat[:, :3], quat, grip], axis=1).astype(np.float32)
    return out.reshape(*a.shape[:-1], 8)


def pose12_grip_to_droid7(rotation, pose12: np.ndarray, grip: float) -> np.ndarray:
    """A copy of `pose12_grip_to_droid7` in `lerobot/policies/praxis/processor_praxis.py`."""
    p = np.asarray(pose12, dtype=np.float64).reshape(12)
    rpy = rotation.matrix_to_euler_xyz(p[3:].reshape(1, 3, 3))[0]
    return np.concatenate([p[:3], rpy, [float(grip)]]).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("seq")
    p.add_argument("out")
    p.add_argument(
        "--weights-dir", default="", help="folder the config's relative paths start from (default: repo)"
    )
    p.add_argument("--checkpoint", default="checkpoints/praxis-1-droid")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--chunks", type=int, default=0)
    p.add_argument("--client-resize", action="store_true")
    p.add_argument("--rotation", default=str(HERE.parents[1] / "src/lerobot/utils/rotation.py"))
    a = p.parse_args()
    repo = str(Path(a.repo).resolve())
    out = Path(a.out).resolve()  # before the chdir below
    rotation = load_rotation(a.rotation)
    seq = dict(np.load(a.seq))
    sys.path.insert(0, repo)
    os.chdir(a.weights_dir or repo)
    import praxis
    import torch
    from praxis.layers import Attention
    from scripts.run_inference import build_pipeline, load_config

    assert praxis.__file__.startswith(repo), praxis.__file__
    print("praxis from", praxis.__file__, "torch", torch.__version__, flush=True)
    cfg = load_config(f"{repo}/configs/serve_droid.yaml", [f"model.checkpoint={a.checkpoint}"])
    chunk_ticks = [int(t) for t in seq["chunk_ticks"]]
    if a.chunks:
        chunk_ticks = chunk_ticks[: a.chunks]
    t0 = time.perf_counter()
    pipe = build_pipeline(cfg)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    sc = cfg["session"]
    skip = str(sc["skip_video_decoder"]).lower() in ("on", "true", "1", "auto")
    sess = pipe.session(
        prompt=str(seq["task"]),
        seed=int(seq["seed"]),
        max_context_chunks=int(sc["max_context_chunks"]),
        control_mode=str(sc["control_mode"]),
        exec_horizon=sc["exec_horizon"],
        skip_last_video=skip,
    )
    backend = next(
        m.effective_backend(pipe.device) for m in pipe.policy.modules() if isinstance(m, Attention)
    )
    print(f"loaded in {load_s:.1f} s, attention backend {backend}", flush=True)
    resize = None
    if a.client_resize:
        from praxis import single_arm

        rh, rw = single_arm.fit_resize_size(pipe.variant.tile_size[1])

        def resize(img: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float()
            y = torch.nn.functional.interpolate(
                x, size=(rh, rw), mode="bilinear", align_corners=False, antialias=True
            )
            return y.round_().clamp_(0, 255)[0].permute(1, 2, 0).to(torch.uint8).numpy()

    kept: deque = deque(maxlen=25)
    times, step_times, chunks = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for tick in range(chunk_ticks[-1] + 1):
        i = int(seq["index"][tick])
        ext, wr = seq["exterior"][i], seq["wrist"][i]
        if resize is not None:
            ext, wr = resize(ext), resize(wr)
        kept.append((ext, wr, seq["state7"][tick]))
        if tick not in chunk_ticks:
            continue
        torch.cuda.synchronize()
        s = time.perf_counter()
        pose8 = droid7_to_pose8(rotation, np.stack([row[2] for row in kept]))
        obs = {
            "descriptor": "flange",
            "image.exterior": np.stack([row[0] for row in kept]),
            "image.wrist": np.stack([row[1] for row in kept]),
            "robot_state.ee_pose": pose8[:, :7],
            "grip": pose8[:, 7:8],
            "task": str(seq["task"]),
        }
        s_step = time.perf_counter()
        res = sess.step(obs)
        e_step = time.perf_counter()
        chunk = np.stack(
            [
                pose12_grip_to_droid7(rotation, c["robot_command"]["pose"], c["target_grip"])
                for c in res.commands
            ]
        )
        torch.cuda.synchronize()
        e = time.perf_counter()
        times.append((e - s) * 1e3)
        step_times.append((e_step - s_step) * 1e3)
        chunks.append(chunk)
        print(json.dumps({"chunk": len(times) - 1, "tick": tick, "ms": round(times[-1], 2)}), flush=True)
    timed = times[a.warmup : a.warmup + a.count]
    summary = {
        "side": "runway" + ("_client_resize" if a.client_resize else ""),
        "attention_backend": backend,
        "load_s": round(load_s, 1),
        "warmup_ms": [round(x, 1) for x in times[: a.warmup]],
        "median_ms": round(float(np.median(timed)), 2) if timed else None,
        "p90_ms": round(float(np.percentile(timed, 90)), 2) if timed else None,
        "n_timed": len(timed),
        "sampler": {
            key: float(getattr(sess.sampler, key))
            for key in ("num_steps", "t_shift", "cfg_scale", "cfg_video_scale")
        },
        "step_only_median_ms": round(float(np.median(step_times[a.warmup : a.warmup + a.count])), 2)
        if timed
        else None,
        "peak_alloc_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    width = max(len(c) for c in chunks)
    padded = np.full((len(chunks), width, 7), np.nan, np.float32)
    for k, c in enumerate(chunks):
        padded[k, : len(c)] = c
    np.savez(out, actions=padded, rows=np.array([len(c) for c in chunks]), chunk_ms=np.array(times))
    print("SUMMARY " + json.dumps(summary), flush=True)
    if len(timed) != a.count:
        sys.exit(f"timed {len(timed)} chunks, not --count {a.count}")


if __name__ == "__main__":
    main()
