# pi0.5: ExecuTorch with TensorRT against openpi

Scripts that time openpi's own pi0.5 inference (JAX and PyTorch) and a LeRobot ExecuTorch bundle of the same
weights, on the same GPU, the same 16 real LIBERO frames and the same noise.

Two environments:

- **openpi**: a checkout of [openpi](https://github.com/Physical-Intelligence/openpi) at `215abfb`, set up with its
  `uv sync` (and its `transformers_replace` step for the PyTorch path), as its README says.
- **bundle**: Python 3.12 with `pip install "torch_tensorrt[executorch]"`, then this LeRobot checkout with
  `pip install -e ".[pi,export]"`. The export also needs `nvcc` from CUDA 12.8 or newer on `PATH`.

## 1. Weights and frames

```bash
# openpi's JAX checkpoint, and its PyTorch conversion with openpi's own converter (openpi environment)
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_libero .
python examples/convert_jax_model_to_pytorch.py --config-name pi05_libero \
    --checkpoint-dir pi05_libero --output-path pi05_libero_pytorch

# 16 real LIBERO frames, resized to 224 as openpi's LIBERO client does (openpi environment)
python make_frames.py frames224.npz

# a LeRobot policy folder with openpi's weights and openpi's normalization (bundle environment)
python make_checkpoint.py pi05_libero_pytorch pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
    pi05_openpi_libero
```

## 2. Export

```bash
lerobot-export --policy.path=pi05_openpi_libero --backend.type=executorch_tensorrt --policy.dtype=bfloat16 \
    --task="pick up the book and place it in the back compartment of the caddy" --one_program=true \
    --output_dir=bundles/one
```

Without `--one_program=true` the export keeps the default chain of small programs, which builds on an 8 GB board.

## 3. Time

```bash
export OPENPI_PY=<openpi>/.venv/bin/python OPENPI_SRC=<openpi> BUNDLE_PY=$(which python) BUNDLES=bundles \
    JAX_CKPT=pi05_libero PT_CKPT=pi05_libero_pytorch FRAMES=frames224.npz
bash run_ab.sh 0 out 1 ref_state                  # float32 reference with LeRobot's prompt
bash run_ab.sh 0 out 3 jax585 pt585 bundle:one    # 3 rounds on GPU 0, order rotates each round
python compare.py out/ref_state.npz out/one_r1.npz
```

What is timed:

- openpi: `policy._sample_actions` followed by a device sync, with the inputs already on the GPU. openpi's own
  `infer_ms` has no sync, so for JAX it is far below the real time; it is saved too, for reference.
- bundle: `BundleEngine.predict_chunk`, which uploads the frame, runs the programs and downloads the actions.
  PyTorch is never imported.
- Each run is a fresh process that times 30 chunks after warm-up. Report the median over rounds of each run's
  median.

Same work on both sides: a LeRobot bundle runs 2 cameras and a prompt of 73 tokens (585 prefix tokens). openpi
fills a third camera with zeros and masks it, and pads its prompt to 200 tokens (968 tokens). The rows `jax585` and
`pt585` run openpi with `--two-cams --max-token-len 73`, so both sides do the same work. The rows `jax` and `pt`
run openpi as it ships.

Accuracy: each path against a float32 openpi PyTorch reference on the same frames and noise. LeRobot's pi0.5 always
writes the state into its prompt, and openpi's pi05_libero does not, so a bundle is compared with the reference
`ref_state` (openpi with the state in the prompt), and openpi's rows with `ref`.
