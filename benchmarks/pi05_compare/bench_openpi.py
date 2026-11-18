"""Time openpi's own pi0.5 inference (JAX or PyTorch) on real LIBERO frames, and save its actions.

    python bench_openpi.py --ckpt DIR --frames frames224.npz --out OUT.json [--actions OUT.npz]
        [--chunks 30] [--warm 3] [--two-cams] [--max-token-len N] [--discrete-state] [--f32]

Run it in openpi's own environment. The policy is built with `openpi.policies.policy_config.create_trained_policy`,
as openpi's serve script does. openpi picks PyTorch when DIR holds `model.safetensors`, and JAX otherwise.

Chunk i uses frame i % N, that frame's task, and noise draw i % N (seed 0, one draw per frame).

Timers per chunk:
- model_ms: `policy._sample_actions` only, followed by a device sync (`jax.block_until_ready` or
  `torch.cuda.synchronize`). openpi's own `infer_ms` is taken around the same call without a sync, so for JAX it
  measures little more than the launch. It is recorded too, from a separate pass, for reference.
- e2e_ms: the whole `policy.infer` call (input transforms, tokenization, host to device, model, device to host,
  output transforms).

Options that change the work:
- --two-cams leaves out the third camera slot, which openpi fills with an all-zero image and masks. With
  --max-token-len 73 the prefix then has as many tokens as a LeRobot bundle's (2 x 256 + 73 = 585).
- --discrete-state writes the state into the prompt as bin numbers, as LeRobot's pi0.5 always does. Use it for the
  float32 reference that a bundle is compared with.
- --unmask-third keeps openpi's all-zero third camera but does not mask it. Use it for the float32 reference of a
  bundle made with make_checkpoint.py --third-camera, which sees that camera too.
- --f32 builds a float32 reference: every parameter in float32, no torch.compile, TF32 off. PyTorch checkpoints only.
"""

import argparse
import dataclasses
import json
import os
import statistics
import subprocess
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
    p.add_argument("--ckpt", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--actions", default=None)
    p.add_argument("--config", default="pi05_libero")
    p.add_argument("--chunks", type=int, default=30)
    p.add_argument("--warm", type=int, default=3)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--two-cams", action="store_true")
    p.add_argument("--max-token-len", type=int, default=None)
    p.add_argument("--discrete-state", action="store_true")
    p.add_argument("--unmask-third", action="store_true")
    p.add_argument("--f32", action="store_true")
    a = p.parse_args()

    import jax
    import torch
    from openpi.policies import policy_config
    from openpi.training import config as _config

    cfg = _config.get_config(a.config)
    is_pytorch = os.path.exists(os.path.join(a.ckpt, "model.safetensors"))
    model_cfg = cfg.model
    if a.discrete_state:
        model_cfg = dataclasses.replace(model_cfg, discrete_state_input=True)
    if a.max_token_len:
        model_cfg = dataclasses.replace(model_cfg, max_token_len=a.max_token_len)
    if is_pytorch and a.f32:
        model_cfg = dataclasses.replace(model_cfg, pytorch_compile_mode=None, dtype="float32")
    cfg = dataclasses.replace(cfg, model=model_cfg)
    if a.two_cams:
        # openpi has no setting for the camera set: the image keys are a default argument of its preprocessing.
        from openpi.models import model as _model
        from openpi.models_pytorch import preprocessing_pytorch as _preprocessing

        for fn in (_model.preprocess_observation, _preprocessing.preprocess_observation_pytorch):
            assert "image_keys" in (fn.__kwdefaults__ or {}), fn
            fn.__kwdefaults__["image_keys"] = ("base_0_rgb", "left_wrist_0_rgb")
    if a.unmask_third:
        from openpi.policies import libero_policy

        inputs = libero_policy.LiberoInputs.__call__

        def unmasked(self, data):
            out = inputs(self, data)
            out["image_mask"]["right_wrist_0_rgb"] = np.True_
            return out

        libero_policy.LiberoInputs.__call__ = unmasked
    if a.f32:
        from openpi.models_pytorch import gemma_pytorch

        keep = gemma_pytorch.PaliGemmaWithExpertModel.to_bfloat16_for_selected_params
        gemma_pytorch.PaliGemmaWithExpertModel.to_bfloat16_for_selected_params = (
            lambda self, precision="bfloat16": keep(self, "float32")
        )

    t0 = time.perf_counter()
    policy = policy_config.create_trained_policy(cfg, a.ckpt, pytorch_device="cuda:0" if is_pytorch else None)
    if is_pytorch:
        torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    if a.f32:
        # openpi's PyTorch model turns TF32 on; a float32 reference must not use it.
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    z = np.load(a.frames)
    n = len(z["state"])
    rng = np.random.default_rng(a.noise_seed)
    noises = [
        rng.standard_normal((model_cfg.action_horizon, model_cfg.action_dim)).astype(np.float32)
        for _ in range(n)
    ]
    obs = [
        {
            "observation/image": z["image"][i],
            "observation/wrist_image": z["image2"][i],
            "observation/state": z["state"][i],
            "prompt": str(z["task"][i]),
        }
        for i in range(n)
    ]

    def sync(x):
        if is_pytorch:
            torch.cuda.synchronize()
        else:
            jax.block_until_ready(x)
        return x

    sample = policy._sample_actions
    model_ms = []

    def timed_sample(*args, **kwargs):
        t = time.perf_counter()
        out = sync(sample(*args, **kwargs))
        model_ms.append((time.perf_counter() - t) * 1e3)
        return out

    policy._sample_actions = timed_sample

    # The first calls compile (XLA or torch.compile); later warm-up calls may still autotune.
    warm_s = []
    for i in range(a.warm):
        t = time.perf_counter()
        policy.infer(obs[i % n], noise=noises[i % n])
        warm_s.append(round(time.perf_counter() - t, 3))
    model_ms.clear()

    e2e_ms = []
    for i in range(a.chunks):
        t = time.perf_counter()
        policy.infer(obs[i % n], noise=noises[i % n])
        e2e_ms.append((time.perf_counter() - t) * 1e3)
    timed = list(model_ms)
    steady_mib = card_memory_mib()

    policy._sample_actions = sample
    as_shipped = [
        policy.infer(obs[i % n], noise=noises[i % n])["policy_timing"]["infer_ms"] for i in range(10)
    ]

    actions = np.stack(
        [np.asarray(policy.infer(obs[i], noise=noises[i])["actions"], dtype=np.float64) for i in range(n)]
    )
    if a.actions:
        np.savez(a.actions, actions=actions)

    row = {
        "path": ("openpi_pytorch" if is_pytorch else "openpi_jax") + ("_f32" if a.f32 else ""),
        "config": a.config,
        "two_cams": a.two_cams,
        "max_token_len": model_cfg.max_token_len,
        "discrete_state_input": model_cfg.discrete_state_input,
        "unmask_third": a.unmask_third,
        "compile_mode": model_cfg.pytorch_compile_mode if is_pytorch else "jax.jit",
        "frames": n,
        "chunks": a.chunks,
        "model_ms": summary(timed),
        "model_ms_all": [round(x, 3) for x in timed],
        "e2e_ms": summary(e2e_ms),
        "openpi_infer_ms_as_shipped": summary(as_shipped),
        "steady_card_mib": steady_mib,
        "load_s": round(load_s, 1),
        "warmup_s": warm_s,
        "versions": {k: version_of(k) for k in ("jax", "jaxlib", "torch", "transformers", "flax", "numpy")},
    }
    print("RESULT " + json.dumps(row), flush=True)
    with open(a.out, "w") as f:
        json.dump(row, f, indent=1)


if __name__ == "__main__":
    main()
