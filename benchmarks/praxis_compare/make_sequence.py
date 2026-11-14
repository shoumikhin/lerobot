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

"""Write the recorded sequence every side replays: real DROID frames from `lerobot/droid_100`.

usage: python make_sequence.py seq.npz [--root <local copy of lerobot/droid_100>]

One pass plays the control ticks of 33 chunks from a reset (3 warm-up + 30 timed): a chunk at tick 0 (21
actions), then one every 24 ticks from tick 21, so 766 ticks. Each tick is a real frame of the dataset (15 fps,
the policy's own rate): the `exterior_image_1_left` and `wrist_image_left` views and the 7-value DROID state
`[x, y, z, roll, pitch, yaw, gripper]`, episodes 0, 1, 2, ... back to back. The 180x320 frames are resized to
288x512 (PIL bilinear) once, here, outside every timer, so every side reads the same bytes.
"""

import argparse
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image

REPO_ID = "lerobot/droid_100"
REVISION = "87301a2d2e99340e2010c9ef0f1d8e780b08aaf9"
VIEWS = ("exterior_image_1_left", "wrist_image_left")
HEIGHT, WIDTH = 288, 512
CHUNKS, FIRST, LATER = 33, 21, 24
TICKS = FIRST + LATER * (CHUNKS - 2) + 1  # the last chunk runs at tick FIRST + LATER * (CHUNKS - 2)
TASK = "pick up the red cube"


def decode(path: Path, count: int) -> np.ndarray:
    out = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            img = Image.fromarray(frame.to_ndarray(format="rgb24")).resize((WIDTH, HEIGHT), Image.BILINEAR)
            out.append(np.asarray(img))
            if len(out) == count:
                break
    return np.stack(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--root", default="", help="a local copy of lerobot/droid_100 (default: download it)")
    a = p.parse_args()
    if a.root:
        root = Path(a.root)
    else:
        from huggingface_hub import snapshot_download

        patterns = ["meta/**", "data/**", *(f"videos/observation.images.{v}/**" for v in VIEWS)]
        root = Path(
            snapshot_download(REPO_ID, repo_type="dataset", revision=REVISION, allow_patterns=patterns)
        )
    data = pd.concat([pd.read_parquet(f) for f in sorted((root / "data").rglob("*.parquet"))])
    data = data.sort_values("index").iloc[:TICKS]
    assert len(data) == TICKS, len(data)
    # one video file per camera holds every episode back to back, in index order
    ext, wrist = (
        decode(next((root / f"videos/observation.images.{v}").rglob("*.mp4")), TICKS) for v in VIEWS
    )
    state = np.stack(data["observation.state"].to_numpy()).astype(np.float32)
    chunk_ticks = np.array([0, *range(FIRST, TICKS, LATER)])
    assert len(chunk_ticks) == CHUNKS
    np.savez(
        a.out,
        exterior=ext,
        wrist=wrist,
        index=np.arange(TICKS, dtype=np.int64),
        state7=state,
        task=np.array(TASK),
        chunk_ticks=chunk_ticks,
        seed=np.array(0),
        episodes=data["episode_index"].to_numpy(),
    )
    print("wrote", a.out, ext.shape, wrist.shape, "episodes", sorted(set(data["episode_index"].tolist())))


if __name__ == "__main__":
    main()
