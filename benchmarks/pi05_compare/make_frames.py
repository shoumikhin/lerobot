"""Pick 16 real LIBERO frames and resize them the way openpi's LIBERO client does.

    python make_frames.py frames224.npz

Run it in openpi's environment (it needs `openpi_client`, `huggingface_hub`, `pyarrow` and Pillow). It reads
the `physical-intelligence/libero` dataset at a pinned revision and downloads only the 16 episodes it needs.

Which frames: the dataset is cut into 32 evenly spaced samples (every `total_frames // 32`-th frame). Those 32
episodes are read one after another, and the frame halfway between each of the first 16 samples and the next one is
taken. The result is 16 frames with 6 different tasks.

The images are resized to 224 x 224 with `openpi_client.image_tools.resize_with_pad`, as openpi's LIBERO client does
before it calls the policy, so no path under test resizes again. The output holds `image`, `image2` (the wrist
camera), `state`, `task` and `index` (the frame's index in the dataset).
"""

import bisect
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from openpi_client import image_tools
from PIL import Image

REPO = "physical-intelligence/libero"
REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
SAMPLES = 32
FRAMES = 16


def fetch(path):
    return hf_hub_download(REPO, path, repo_type="dataset", revision=REVISION)


def read_jsonl(path):
    return [json.loads(line) for line in Path(fetch(path)).read_text().splitlines() if line.strip()]


def main():
    info = json.loads(Path(fetch("meta/info.json")).read_text())
    lengths = [
        e["length"] for e in sorted(read_jsonl("meta/episodes.jsonl"), key=lambda e: e["episode_index"])
    ]
    tasks = {t["task_index"]: t["task"] for t in read_jsonl("meta/tasks.jsonl")}
    total = info["total_frames"]
    assert sum(lengths) == total, (sum(lengths), total)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()

    step = total // SAMPLES
    samples = list(range(0, SAMPLES * step, step))
    episode_of = [bisect.bisect_right(starts, g) - 1 for g in samples]
    episodes = sorted(set(episode_of))
    # The sampled episodes, read one after another: position in that sequence -> (episode, frame in episode).
    offsets = np.concatenate([[0], np.cumsum([lengths[e] for e in episodes])]).tolist()
    position = [offsets[episodes.index(e)] + g - starts[e] for g, e in zip(samples, episode_of, strict=True)]

    def frame_at(pos):
        k = bisect.bisect_right(offsets, pos) - 1
        return episodes[k], pos - offsets[k]

    picks = [frame_at((a + b) // 2) for a, b in zip(position[:FRAMES], position[1 : FRAMES + 1], strict=True)]

    rows = {name: [] for name in ("image", "image2", "state", "task", "index")}
    for episode, frame in picks:
        path = info["data_path"].format(episode_chunk=episode // info["chunks_size"], episode_index=episode)
        table = pq.read_table(fetch(path)).to_pylist()
        row = table[frame]
        assert row["episode_index"] == episode and row["frame_index"] == frame, (row["episode_index"], frame)
        for key, name in (("image", "image"), ("wrist_image", "image2")):
            image = np.asarray(Image.open(io.BytesIO(row[key]["bytes"])).convert("RGB"))
            rows[name].append(image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224)))
        rows["state"].append(np.asarray(row["state"], np.float32))
        rows["task"].append(tasks[row["task_index"]])
        rows["index"].append(row["index"])

    out = {k: np.stack(v) if k != "task" else np.array(v) for k, v in rows.items()}
    np.savez(sys.argv[1], **out)
    print(sys.argv[1], out["image"].shape, "indices", out["index"].tolist(), "tasks", len(set(rows["task"])))


if __name__ == "__main__":
    main()
