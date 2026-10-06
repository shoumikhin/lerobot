# Compiled LeRobot policies on an NVIDIA Jetson

Export a trained LeRobot policy, then run the compiled folder with `lerobot-rollout`, the same command used for a PyTorch checkpoint. These examples cover ACT, SmolVLA, pi0.5 and GR00T N1.7 through several compiler routes. Latency, memory use and export support depend on the policy, route and device; the results below are provisional.

## Hardware

- [SO-101](../../docs/source/so101.mdx) arm with a scene camera and a wrist camera
- [Jetson AGX Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/), to train, export and run
- [Jetson Orin Nano Super](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/), to export and run ACT with TensorRT (see Limits)

## Routes

A route is how the policy is compiled. Each writes a folder that `lerobot-rollout` can load.

| route                 | what it writes                                                   | loads PyTorch at runtime                    |
| --------------------- | ---------------------------------------------------------------- | ------------------------------------------- |
| `executorch_tensorrt` | Torch-TensorRT into an ExecuTorch `.pte` program                 | yes, through the public ExecuTorch bindings |
| `executorch_cuda`     | ExecuTorch's CUDA backend, without TensorRT: a `.pte` and `.ptd` | yes, through the public ExecuTorch bindings |
| `onnx_tensorrt`       | ONNX, then a TensorRT `.engine`                                  | no for raw-frame folders; see Limits        |
| `torch_tensorrt`      | Torch-TensorRT into an AOTInductor `.pt2` package                | yes                                         |

## Install

On each Jetson (JetPack 7.2.1), with [uv](https://docs.astral.sh/uv/). These instructions assume the latest nightly packages contain the required export and runtime fixes. The final integrated rerun is still pending.

```bash
git clone --branch export https://github.com/shoumikhin/lerobot.git
cd lerobot
uv venv --python 3.12
source .venv/bin/activate
uv pip install \
    -e ".[smolvla,pi,groot,feetech,export]" \
    "torch-tensorrt[executorch]" onnx onnxscript \
    --index-url https://download.pytorch.org/whl/nightly/cu132 \
    --extra-index-url https://pypi.org/simple \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease allow
```

This installs PyTorch, Torch-TensorRT, ExecuTorch and `torch-tensorrt-executorch-runtime` (the TensorRT backend for ExecuTorch programs). The `export` extra supplies CUDA bindings for the ONNX-TensorRT runtime. Exporting requires PyTorch for every route.

## Export a policy

Build on the GPU model that will run the program. For ACT, choose one of these commands:

```bash
python examples/export/act_executorch_tensorrt.py --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model --output_dir=outputs/export/act_executorch_tensorrt
python examples/export/act_executorch_cuda.py --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model --output_dir=outputs/export/act_executorch_cuda
python examples/export/act_onnx_tensorrt.py --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model --output_dir=outputs/export/act_onnx_tensorrt
python examples/export/act_torch_tensorrt.py --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model --output_dir=outputs/export/act_torch_tensorrt
```

| policy  | `executorch_tensorrt`                    | `executorch_cuda`                              | `onnx_tensorrt`                    | `torch_tensorrt`                    |
| ------- | ---------------------------------------- | ---------------------------------------------- | ---------------------------------- | ----------------------------------- |
| ACT     | [script](act_executorch_tensorrt.py)     | [script](act_executorch_cuda.py)               | [script](act_onnx_tensorrt.py)     | [script](act_torch_tensorrt.py)     |
| SmolVLA | [script](smolvla_executorch_tensorrt.py) | no script                                      | [script](smolvla_onnx_tensorrt.py) | [script](smolvla_torch_tensorrt.py) |
| pi0.5   | [script](pi05_executorch_tensorrt.py)    | [script](pi05_executorch_cuda.py)              | [script](pi05_onnx_tensorrt.py)    | [script](pi05_torch_tensorrt.py)    |
| GR00T   | [script](groot_executorch_tensorrt.py)   | [script](groot_executorch_cuda.py), see Limits | [script](groot_onnx_tensorrt.py)   | [script](groot_torch_tensorrt.py)   |

Each script loads the policy, exports it, compiles it and saves the folder. The folder also keeps a test case: an observation and the actions the PyTorch policy gave for it.

SmolVLA and GR00T also need the dataset they were trained on, for the cameras and task. Add `--dataset.root=<folder>` if the dataset is on your disk:

```bash
python examples/export/smolvla_executorch_tensorrt.py \
    --policy.path=outputs/train/smolvla_so101/checkpoints/last/pretrained_model \
    --dataset.repo_id="<user>/so101_dataset" \
    --output_dir=outputs/export/smolvla_executorch_tensorrt
```

pi0.5 needs the training task and the camera slots the robot actually provides. For a scene and wrist camera:

```bash
python examples/export/pi05_executorch_tensorrt.py \
    --policy.path=outputs/train/pi05_so101/checkpoints/last/pretrained_model \
    --task="pick up the block and place it in the cup" \
    --cameras observation.images.base_0_rgb observation.images.left_wrist_0_rgb \
    --output_dir=outputs/export/pi05_executorch_tensorrt
```

The omitted camera slot uses the policy's own padding. Map the robot's names to these slots at rollout, as shown below. The `executorch_cuda` scripts compile kernels with `nvcc`, so first run `export PATH=/usr/local/cuda/bin:$PATH`.

### Export once, build on each device

With `--export_only`, the `executorch_tensorrt` and `onnx_tensorrt` scripts save the exported graph without building its TensorRT engine. Copy the folder to the target device, then build there:

```bash
python examples/export/act_onnx_tensorrt.py \
    --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model \
    --output_dir=outputs/export/act_onnx_tensorrt \
    --export_only
python examples/export/build_engine.py outputs/export/act_onnx_tensorrt
```

`--workspace_gib`, `--tactic_gib` (ONNX only) and `--optimization_level` can limit build resources, but do not guarantee the engine fits. With `--step_engine`, `pi05_onnx_tensorrt.py` splits pi0.5 into three engines: the prompt and cameras, a denoising step, and the actions. This option is not available in the direct ExecuTorch-TensorRT exporter.

## Run it

Use the same calibrated robot and camera settings as the training recording. Adjust the camera device paths, resolution and rate below to match your setup:

```bash
lerobot-rollout \
    --policy.path=outputs/export/act_executorch_tensorrt \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: '/dev/video-wrist', width: 640, height: 480, fps: 30}, scene: {type: opencv, index_or_path: '/dev/video-scene', width: 640, height: 480, fps: 30} }" \
    --duration=30
```

SmolVLA, pi0.5 and GR00T run only the task they were exported for, so pass that same `--task`. When camera names differ, add `--rename_map`. For the pi0.5 export above, add these arguments to either a real-robot or replay rollout:

```bash
--task="pick up the block and place it in the cup" \
--rename_map='{"observation.images.scene": "observation.images.base_0_rgb", "observation.images.wrist": "observation.images.left_wrist_0_rgb"}'
```

Before the robot moves, the rollout runs the saved test case and stops if its action error exceeds the folder's saved tolerance. This is a numerical check, not a robot safety or task-success test.

### Without a robot

[replay_robot](replay_robot) is a LeRobot robot plugin that serves the scene and wrist cameras and joint state from a recorded SO-101 episode, and accepts every action. Set `--robot.fps` to the recording's rate (the example uses 30). Install it, prepare the frames once, then use the official rollout command:

```bash
uv pip install -e examples/export/replay_robot
python examples/export/replay_robot/write_frames.py \
    --dataset.repo_id="<user>/so101_dataset" \
    --episodes 0 1 2 3 \
    --output_dir=outputs/replay_frames
lerobot-rollout \
    --policy.path=outputs/export/act_executorch_tensorrt \
    --robot.type=replay \
    --robot.frames_dir=outputs/replay_frames \
    --robot.fps=30 \
    --duration=60 \
    --return_to_initial_position=false \
    --play_sounds=false
```

Add `--robot.log_path=actions.npz` to save actions and timestamps. Replay tests execution without moving hardware; observations do not respond to the predicted actions.

## Results

> [!NOTE]
> Provisional numbers, pending final rerun. The chunk and action-agreement tables use earlier export scripts and package builds, not this final fork. The separate rollout table uses newer raw-frame folders and public ExecuTorch bindings. These are different measurements, not a single ranking.

Measured on JetPack 7.2.1 with locked CPU, GPU and memory clock settings. Chunk latency is the median of 100 chunks after 20 warm-up chunks, then the median across 3 fresh processes. `torch.compile` has 3 processes for ACT and only 1 for each other policy. These timings exclude compilation and are not control-loop latency.

### Latency per action chunk, ms

| policy  | device    | `executorch_tensorrt` | `executorch_cuda` | `onnx_tensorrt` | `torch_tensorrt` | PyTorch | `torch.compile` |
| ------- | --------- | --------------------- | ----------------- | --------------- | ---------------- | ------- | --------------- |
| ACT     | Thor      | 3.57                  | 37.8              | 3.70            | 3.71             | 15.9    | 13.2            |
| ACT     | Orin Nano | 26.1                  | 132.5             | 26.7            | 26.4             | 64.3    | 60.7            |
| SmolVLA | Thor      | 28.7                  | no script         | 31.8            | 30.0             | 167.1   | 46.2            |
| SmolVLA | Orin Nano | 126.6                 | no script         | 131.4           | 137.7            | 763.2   | 171.5           |
| pi0.5   | Thor      | 106.0                 | 125.0             | 115.4           | 107.4            | 247.5   | 127.1           |
| GR00T   | Thor      | 59.6                  | 76.7              | 63.2            | 61.1             | 213.3   | 209.4           |

The Orin SmolVLA row uses older folders with host-side preprocessing. The GR00T CUDA row also uses an older export; the current raw-frame script has a compiler failure (see Limits). pi0.5 and GR00T Orin rows are omitted because the earlier artifacts do not establish a reproducible export from these scripts.

A separate newer ACT test with runtime replay and shared host inputs measured **3.7645 ms** for ExecuTorch-TensorRT and **3.8072 ms** for ONNX-TensorRT on Thor with public bindings. This is a **near tie**, not an established speed advantage: sequential-run variation exceeds the difference. It used 12 fresh-frame calls per process with 3.3 seconds between calls; reused-frame results did not show the same benefit. Only ACT has native validation of this input-sharing change.

### Rollout latency and memory

These earlier raw-frame measurements ran the official `lerobot-rollout` for 60 seconds on Thor. Inference is the last status-line value, not a median. Memory is the median process resident set (RSS) over the final 30 seconds, including loaded libraries. It is not peak usage or total board/GPU memory. Do not add it to a GPU memory reading: allocations can overlap on a Jetson.

| policy  | route                 | inference, ms | process RSS, MiB |
| ------- | --------------------- | ------------- | ---------------- |
| ACT     | `executorch_tensorrt` | 4.6           | 916.33           |
| ACT     | `executorch_cuda`     | 37.3          | 842.43           |
| ACT     | `onnx_tensorrt`       | 4.0           | 660.73           |
| SmolVLA | `executorch_tensorrt` | 30.2          | 1032.02          |
| SmolVLA | `onnx_tensorrt`       | 31.8          | 602.25           |

The ACT measurements above predate the newer near-tie test. ONNX-TensorRT used less process memory in these runs. No general latency or memory superiority claim follows from these provisional tables.

### Action agreement

Mean absolute difference from the PyTorch policy, in robot action units, over 50 observations on Thor. ACT, SmolVLA and GR00T use dataset frames; pi0.5 uses random observations. The last column compares PyTorch with a different starting-noise draw. It is context for the policy's randomness, not an accuracy allowance.

| policy  | `executorch_tensorrt` | `executorch_cuda` | `onnx_tensorrt` | `torch_tensorrt` | PyTorch, other noise |
| ------- | --------------------- | ----------------- | --------------- | ---------------- | -------------------- |
| ACT     | 0.01                  | 0.01              | 0.01            | 0.01             | no noise             |
| SmolVLA | 0.17                  | no script         | 0.15            | 0.17             | 3.21                 |
| pi0.5   | 0.32                  | 0.26              | 0.36            | 0.32             | 11.42                |
| GR00T   | 0.28                  | 0.25              | 0.28            | 0.28             | 7.80                 |

## Which routes load PyTorch

Observed from the running process's `/proc/<pid>/maps` on Thor with raw-frame ACT and SmolVLA folders and public ExecuTorch bindings:

| route                    | distinct mapped PyTorch libraries |
| ------------------------ | --------------------------------- |
| `onnx_tensorrt`          | 0                                 |
| `executorch_tensorrt`    | 8                                 |
| `executorch_cuda` (ACT)  | 8                                 |
| `torch_tensorrt`         | yes, by design                    |
| PyTorch, `torch.compile` | yes, by design                    |

The counts cover `libtorch` and `libc10` paths in those runs, not every dependency. The public ExecuTorch Python bindings load PyTorch today. Legacy text-processing folders can load it separately.

## Limits

- **Orin Nano memory.** The board has 8 GB. ACT has completed TensorRT exports and rollouts there, but the current raw-frame SmolVLA exports and pi0.5 split-engine build ran out of available memory. Exporting the graph elsewhere does not remove the target engine builder's memory needs. GR00T raw-frame export on Orin is unverified.
- **Older Orin artifacts are not a current export recipe.** Earlier SmolVLA and pi0.5 folders can run, but pi0.5's measured three-program ExecuTorch folder was not produced by a fork export script. Do not treat it as proof that the current scripts can export pi0.5 on Orin.
- **GR00T resize runs outside the program**, using NumPy antialiased bicubic sampling because the operator is not supported by every compiler route. Its cost must be included in a full rollout comparison; the older chunk table is not a timing of this new path.
- **Legacy SmolVLA text-step folders load PyTorch** for tokenization, including with ONNX-TensorRT. Current raw-frame exports keep the fixed task's tokens in the program.
- **CUDA route coverage is incomplete.** SmolVLA has no `executorch_cuda` script. GR00T's current raw-frame CUDA export fails with the tested stock compiler; a full-model retry is still needed. Current ACT CUDA export on Orin is also unverified.
- **Replay and saved-case agreement do not prove physical task success.** Older pi0.5 rollouts stopped at camera validation; the camera selection and full rename map above are required before a final rerun can validate that path.
