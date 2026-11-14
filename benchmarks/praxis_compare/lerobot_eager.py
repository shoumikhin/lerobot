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

"""LeRobot's eager PyTorch policy on the recorded sequence: `PraxisPolicy` in its checkpoint dtype (bfloat16),
tick by tick, with the checkpoint's own preprocessor and postprocessor, as `lerobot-rollout --policy.path` runs it.

usage: python lerobot_eager.py <seq.npz> <out.npz> [--warmup 3] [--count 30] [--chunks N] [--policy PATH]

TIMER (the ticks where the policy samples a new chunk, which must be the sequence's chunk ticks, as for the bundle):
`torch.cuda.synchronize()`, `time.perf_counter()`, then the preprocessor on the raw NumPy frame, `select_action`,
the postprocessor, the copy of the action back into NumPy (which waits for the GPU), `time.perf_counter()`.
"""

import argparse
import json
import sys
import time

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import load_pretrained_policy, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("seq")
    p.add_argument("out")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--count", type=int, default=30)
    p.add_argument(
        "--chunks", type=int, default=0, help="stop after this many chunks (0: the whole sequence)"
    )
    p.add_argument("--policy", default="runwayml/praxis-1-droid-lerobot")
    a = p.parse_args()
    seq = dict(np.load(a.seq))
    chunk_ticks = [int(t) for t in seq["chunk_ticks"]]
    if a.chunks:
        chunk_ticks = chunk_ticks[: a.chunks]
    device = torch.device("cuda")
    config = PreTrainedConfig.from_pretrained(a.policy)
    config.pretrained_path = a.policy
    config.device = device.type
    t0 = time.perf_counter()
    with device:
        policy = load_pretrained_policy(config, strict=True)
    policy.eval()
    overrides = {"device_processor": {"device": device.type}}
    pre, post = make_pre_post_processors(
        config, pretrained_path=a.policy, preprocessor_overrides=overrides, postprocessor_overrides=overrides
    )
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.1f} s, dtype {policy.model.dtype}", flush=True)
    policy.reset()
    pre.reset()
    post.reset()
    times, other_ms, rows, sample_ticks = [], [], [], []
    with torch.no_grad():
        for tick in range(chunk_ticks[-1] + 1):
            i = int(seq["index"][tick])
            frame = {
                "observation.images.exterior": seq["exterior"][i],
                "observation.images.wrist": seq["wrist"][i],
                "observation.state": seq["state7"][tick],
            }
            samples = policy.count_queued_actions() == 0
            torch.cuda.synchronize()
            s = time.perf_counter()
            batch = pre(prepare_observation_for_inference(dict(frame), device, str(seq["task"])))
            out = post(policy.select_action(batch))
            row = (out[ACTION] if isinstance(out, dict) else out)[0].float().cpu().numpy()
            ms = (time.perf_counter() - s) * 1e3
            rows.append(row)
            if samples:
                sample_ticks.append(tick)
                times.append(ms)
                print(json.dumps({"chunk": len(times) - 1, "tick": tick, "ms": round(ms, 2)}), flush=True)
            else:
                other_ms.append(ms)
    if sample_ticks != chunk_ticks:
        sys.exit(
            f"the policy sampled at ticks {sample_ticks}, not at the sequence's chunk ticks {chunk_ticks}"
        )
    timed = times[a.warmup : a.warmup + a.count]
    summary = {
        "side": "lerobot_eager",
        "dtype": str(policy.model.dtype),
        "load_s": round(load_s, 1),
        "warmup_ms": [round(x, 1) for x in times[: a.warmup]],
        "median_ms": round(float(np.median(timed)), 2) if timed else None,
        "p90_ms": round(float(np.percentile(timed, 90)), 2) if timed else None,
        "n_timed": len(timed),
        "sampler": {
            key: float(getattr(config, key))
            for key in ("num_steps", "t_shift", "cfg_scale", "cfg_video_scale")
        },
        # a tick between chunks only pops a queued action
        "max_other_tick_ms": round(max(other_ms), 2) if other_ms else None,
        "peak_alloc_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    np.savez(a.out, actions=np.stack(rows), chunk_ms=np.array(times))
    print("SUMMARY " + json.dumps(summary), flush=True)
    if len(timed) != a.count:
        sys.exit(f"timed {len(timed)} chunks, not --count {a.count}")


if __name__ == "__main__":
    main()
