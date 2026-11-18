"""Time a LeRobot pi0.5 bundle on real LIBERO frames, without PyTorch, and save its actions.

    python bench_bundle.py BUNDLE --frames frames224.npz --out OUT.json [--actions OUT.npz] [--chunks 30] [--warm 3]

Run it in the environment that ran lerobot-export. Chunk i uses frame i % N and that frame's task. The timer is
`BundleEngine.predict_chunk`: upload the frame, run every program, download the actions. The task is set outside
the timer, as openpi's tokenization is outside its model timer.

The accuracy pass runs each frame once with the same noise as bench_openpi.py (seed 0, one draw per frame). A
bundle exported from a policy with a third camera input (`observation.images.image3`) gets an all-zero image
there, as openpi fills its missing camera.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np


def percentile(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def summary(values):
    return {
        "p50": round(statistics.median(values), 3),
        "p99": round(percentile(values, 0.99), 3),
        "min": round(min(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "n": len(values),
    }


def card_memory_mib():
    """Memory in use on the card, from nvidia-smi. BENCH_GPU names the physical card (default 0)."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            os.environ.get("BENCH_GPU", "0"),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return int(out) if out.isdigit() else None


def version_of(name):
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle")
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--actions", default=None)
    p.add_argument("--chunks", type=int, default=30)
    p.add_argument("--warm", type=int, default=3)
    p.add_argument("--noise-seed", type=int, default=0)
    a = p.parse_args()

    t0 = time.perf_counter()
    from lerobot.bundle.engine import BundleEngine
    from lerobot.bundle.manifest import Bundle

    bundle = Bundle.read(a.bundle)
    engine = BundleEngine(bundle)
    load_s = time.perf_counter() - t0

    z = np.load(a.frames)
    n = len(z["state"])
    shapes = engine.frame_shapes
    cameras = [k for k in shapes if len(shapes[k]) == 3]
    state_key = next(k for k in shapes if k not in cameras)
    zeros = np.zeros_like(z["image"])

    def camera(key, i):
        if key.endswith("image3"):
            return zeros[i]
        return z["image2" if key.endswith("image2") else "image"][i]

    frames = [{**{k: camera(k, i) for k in cameras}, state_key: z["state"][i]} for i in range(n)]
    tasks = [str(t) for t in z["task"]]
    if bundle.manifest.task_tokens is None:
        # A bundle without task tokens runs only the task it was exported with.
        tasks = [bundle.manifest.task] * n
    rng = np.random.default_rng(a.noise_seed)
    noises = [rng.standard_normal(tuple(bundle.manifest.noise_shape)).astype(np.float32) for _ in range(n)]
    for task in dict.fromkeys(tasks):
        engine.set_task(task)

    warm_ms = []
    for i in range(a.warm):
        engine.set_task(tasks[i % n])
        t = time.perf_counter()
        engine.predict_chunk(frames[i % n])
        warm_ms.append(round((time.perf_counter() - t) * 1e3, 3))

    model_ms, e2e_ms = [], []
    for i in range(a.chunks):
        t = time.perf_counter()
        engine.set_task(tasks[i % n])
        t1 = time.perf_counter()
        engine.predict_chunk(frames[i % n])
        t2 = time.perf_counter()
        model_ms.append((t2 - t1) * 1e3)
        e2e_ms.append((t2 - t) * 1e3)
    steady_mib = card_memory_mib()

    actions = []
    for i in range(n):
        engine.reset()
        engine.set_task(tasks[i])
        engine._observe(frames[i])
        actions.append(engine._run_chunk(frames[i], noises[i]).astype(np.float64))
    engine.reset()
    if a.actions:
        np.savez(a.actions, actions=np.stack(actions))

    row = {
        "path": "executorch_tensorrt_bundle",
        "bundle": os.path.basename(os.path.normpath(a.bundle)),
        "programs": sorted(f.stem for f in bundle.path.glob("*.pte")),
        "frame_inputs": {k: list(shapes[k]) for k in shapes},
        "frames": n,
        "chunks": a.chunks,
        "model_ms": summary(model_ms),
        "model_ms_all": [round(x, 3) for x in model_ms],
        "e2e_ms": summary(e2e_ms),
        "steady_card_mib": steady_mib,
        "load_s": round(load_s, 1),
        "warmup_ms": warm_ms,
        "torch_loaded": "torch" in sys.modules,
        "versions": {
            k: version_of(k)
            for k in (
                "executorch",
                "torch_tensorrt",
                "torch_tensorrt_executorch_runtime",
                "tensorrt_cu13_libs",
                "lerobot",
                "numpy",
            )
        },
    }
    print("RESULT " + json.dumps(row), flush=True)
    with open(a.out, "w") as f:
        json.dump(row, f, indent=1)


if __name__ == "__main__":
    main()
