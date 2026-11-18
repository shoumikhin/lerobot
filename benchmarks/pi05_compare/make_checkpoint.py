"""A LeRobot pi0.5 policy folder that holds openpi's pi05_libero weights and openpi's own normalization.

    python make_checkpoint.py OPENPI_PYTORCH_CKPT NORM_STATS_JSON OUT [--third-camera]

Run it in the environment that runs lerobot-export. OPENPI_PYTORCH_CKPT is openpi's PyTorch conversion of
`pi05_libero` (a folder with `model.safetensors`); LeRobot renames openpi's weight names when it loads them.
NORM_STATS_JSON is `assets/physical-intelligence/libero/norm_stats.json` from openpi's JAX checkpoint.

The configuration and processor files start from `lerobot/pi05_libero_finetuned_v044` (at a pinned revision) and
are changed to match openpi's pi05_libero inference:
- a 10-action chunk (openpi's `action_horizon`), still 10 denoising steps;
- state and action normalization with openpi's q01 / q99 quantiles (v044 uses MEAN_STD with other statistics);
- 224 x 224 camera inputs, since the frames arrive already resized, as openpi's LIBERO client sends them.

--third-camera turns the policy's empty camera slot into a real input, `observation.images.image3`, which the
benchmark fills with zeros. The bundle then runs the vision encoder on 3 images and its prefix has 3 x 256 image
tokens, as openpi's does. Unlike openpi, that slot is not masked, so the actions differ a little: use it only to
time the same amount of work as openpi at its default shape.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file, save_file

SOURCE = "lerobot/pi05_libero_finetuned_v044"
REVISION = "7152f13e34c6c115d808aea0e5d08aca74eac14b"
CAMERAS = ("observation.images.image", "observation.images.image2")


def rename_empty_camera(features):
    return {
        ("observation.images.image3" if k == "observation.images.empty_camera_0" else k): v
        for k, v in features.items()
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("openpi_ckpt")
    p.add_argument("norm_stats")
    p.add_argument("out")
    p.add_argument("--third-camera", action="store_true")
    a = p.parse_args()

    src = snapshot_download(SOURCE, revision=REVISION, allow_patterns=["*.json", "*processor.safetensors"])
    os.makedirs(a.out, exist_ok=True)

    cfg = json.loads(Path(f"{src}/config.json").read_text())
    cfg["chunk_size"] = cfg["n_action_steps"] = 10
    assert cfg["num_inference_steps"] == 10, cfg["num_inference_steps"]
    cfg["normalization_mapping"] = {"ACTION": "QUANTILES", "STATE": "QUANTILES", "VISUAL": "IDENTITY"}
    cameras = CAMERAS
    if a.third_camera:
        cfg["input_features"] = rename_empty_camera(cfg["input_features"])
        cfg["empty_cameras"] = 0
        cameras = (*CAMERAS, "observation.images.image3")
    for k in cameras:
        cfg["input_features"][k]["shape"] = [3, 224, 224]
    Path(f"{a.out}/config.json").write_text(json.dumps(cfg, indent=2))

    stats = json.loads(Path(a.norm_stats).read_text())["norm_stats"]
    for name in ("policy_preprocessor", "policy_postprocessor"):
        pipeline = json.loads(Path(f"{src}/{name}.json").read_text())
        for step in pipeline["steps"]:
            c = step.get("config", {})
            if "features" in c and a.third_camera:
                c["features"] = rename_empty_camera(c["features"])
            if "norm_map" in c:
                c["norm_map"] = cfg["normalization_mapping"]
                for k in cameras:
                    if k in c.get("features", {}):
                        c["features"][k]["shape"] = [3, 224, 224]
            if "state_file" in step:
                tensors = load_file(f"{src}/{step['state_file']}")
                for key, openpi_key in (("observation.state", "state"), ("action", "actions")):
                    for stat in ("q01", "q99", "mean", "std"):
                        # openpi may store statistics for its padded 32-wide vectors; it uses the leading entries.
                        n = tensors[f"{key}.{stat}"].shape[0]
                        tensors[f"{key}.{stat}"] = np.asarray(stats[openpi_key][stat], np.float32)[:n]
                save_file(tensors, f"{a.out}/{step['state_file']}")
        Path(f"{a.out}/{name}.json").write_text(json.dumps(pipeline, indent=2))

    weights = f"{a.out}/model.safetensors"
    if not os.path.exists(weights):
        os.symlink(os.path.abspath(f"{a.openpi_ckpt}/model.safetensors"), weights)
    print("wrote", a.out, sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
