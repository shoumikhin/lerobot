# Compiled LeRobot policies on an NVIDIA Jetson

Export a trained LeRobot policy, then run the compiled folder with `lerobot-rollout`, the same command used for a PyTorch checkpoint. These examples cover ACT, SmolVLA, pi0.5 and GR00T N1.7 through several compiler routes. Latency, memory use and export support depend on the policy, route and device; some results below are still pending.

## Hardware

- [SO-101](../../docs/source/so101.mdx) arm with a scene camera and a wrist camera
- [Jetson AGX Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/), to train, export and run
- [Jetson Orin Nano Super](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/), to export and run ACT and SmolVLA (see Limits)

## Routes

A route is how the policy is compiled. Each writes a folder that `lerobot-rollout` can load.

| route                 | what it writes                                                   | loads PyTorch at runtime                    |
| --------------------- | ---------------------------------------------------------------- | ------------------------------------------- |
| `executorch_tensorrt` | Torch-TensorRT into an ExecuTorch `.pte` program                 | yes, through the public ExecuTorch bindings |
| `executorch_cuda`     | ExecuTorch's CUDA backend, without TensorRT: a `.pte` and `.ptd` | yes, through the public ExecuTorch bindings |
| `onnx_tensorrt`       | ONNX, then a TensorRT `.engine`                                  | no for raw-frame folders; see Limits        |
| `torch_tensorrt`      | Torch-TensorRT into an AOTInductor `.pt2` package                | yes                                         |

## Install

On each Jetson (JetPack 7.2.1), with [uv](https://docs.astral.sh/uv/). These instructions assume the latest nightly packages contain the required export and runtime fixes. The results below were measured with ExecuTorch and Torch-TensorRT built from source with those fixes.

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
> Partly provisional. Every number below was measured on 2026-10-06 with this fork and with ExecuTorch and Torch-TensorRT built from source, including the fixes the install step expects in the nightlies. Cells marked "pending" are not measured yet: the `executorch_cuda` routes of pi0.5 and GR00T and pi0.5 `torch_tensorrt` on Thor, and SmolVLA `torch.compile`, pi0.5 and GR00T on the Orin Nano. Until those are in, these tables do not rank Torch-TensorRT for pi0.5.

Thor: JetPack 7.2.1, power mode MAXN, CPU, GPU and memory clocks locked. Orin Nano Super: JetPack 7.2.1, GPU clock locked at 1020 MHz. Every folder was exported on the board that ran it.

### Latency per action chunk, ms

The median time to compute one action chunk: 100 chunks after 20 warm-up chunks in each of 3 fresh processes, then the median of the 3. Loading and compiling are not included, and this is not control-loop latency. `torch.compile` wraps the chunk call in mode `default` on Thor (the `lerobot-rollout` default) and `max-autotune` on the Orin Nano. SmolVLA and pi0.5 also turn on their own `compile_model` setting (mode `max-autotune`), and pi0.5 uses only that, as `lerobot-rollout --use_torch_compile` does. On Thor, pi0.5 and GR00T `torch.compile` ran in 1 process. GR00T times include its image resize (see Limits).

| policy  | device    | `executorch_tensorrt` | `executorch_cuda` | `onnx_tensorrt` | `torch_tensorrt` | PyTorch | `torch.compile` |
| ------- | --------- | --------------------- | ----------------- | --------------- | ---------------- | ------- | --------------- |
| ACT     | Thor      | 3.43                  | 37.64             | 3.50            | 3.70             | 15.91   | 12.80           |
| ACT     | Orin Nano | 25.65                 | 131.5             | 26.36           | 26.26            | 64.60   | 61.53           |
| SmolVLA | Thor      | 28.17                 | no script         | 30.95           | 29.20            | 166.7   | 44.47           |
| SmolVLA | Orin Nano | 125.6                 | no script         | 130.5           | 136.0            | 762.0   | pending         |
| pi0.5   | Thor      | 105.4                 | pending           | 124.7           | pending          | 244.2   | 129.9           |
| pi0.5   | Orin Nano | pending               | pending           | pending         | pending          | pending | pending         |
| GR00T   | Thor      | 93.30                 | pending           | 97.84           | 88.98            | 211.5   | 206.7           |
| GR00T   | Orin Nano | pending               | pending           | pending         | pending          | pending | pending         |

ExecuTorch-TensorRT has the lowest chunk time of the TensorRT routes on ACT and SmolVLA on both boards, and on pi0.5 against ONNX-TensorRT on Thor. On GR00T, Torch-TensorRT is faster. On ACT its lead over the next TensorRT route is small, about 2%: 3.43 against 3.50 ms for ONNX-TensorRT on Thor, and 25.65 against 26.26 ms for Torch-TensorRT on the Orin Nano. On SmolVLA it takes 3.5% less time than Torch-TensorRT and 9% less than ONNX-TensorRT on Thor, and 3.7% less than ONNX-TensorRT and 7.6% less than Torch-TensorRT on the Orin Nano. On pi0.5 it takes 15% less time than ONNX-TensorRT (105.4 against 124.7 ms). On GR00T, Torch-TensorRT takes 4.6% less time than ExecuTorch-TensorRT (88.98 against 93.30 ms), which takes 4.6% less than ONNX-TensorRT (97.84 ms). It is faster than PyTorch in every measured row: 4.6 times on ACT (Thor), 2.5 times on ACT (Orin Nano), 5.9 times on SmolVLA (Thor), 6.1 times on SmolVLA (Orin Nano), 2.3 times on pi0.5 and 2.3 times on GR00T. It also beats `torch.compile` in every measured row, by 1.2 times (pi0.5) to 3.7 times (ACT on Thor).

### Rollout latency and memory

The official `lerobot-rollout` with the [replay robot](replay_robot) at 30 Hz for 60 seconds. Inference is the median time per chunk on the status line at the end of the run (up to the last 30 chunks). Memory is the median process resident set (RSS) over the final 30 seconds, read from outside the process. It includes loaded libraries. It is not GPU memory, and the two should not be added, because a Jetson shares one memory between the CPU and the GPU.

| policy  | device    | route                 | loop, Hz | inference, ms   | process RSS, MiB |
| ------- | --------- | --------------------- | -------- | --------------- | ---------------- |
| ACT     | Thor      | `executorch_tensorrt` | 29.94    | 4.1             | 1107.6           |
| ACT     | Thor      | `executorch_cuda`     | 29.89    | 38.7            | 802.6            |
| ACT     | Thor      | `onnx_tensorrt`       | 29.94    | 4.0             | 644.8            |
| ACT     | Thor      | `torch_tensorrt`      | 29.94    | 4.6             | 1555.6           |
| ACT     | Thor      | PyTorch               | 29.78    | 16.4            | 2036.3           |
| ACT     | Thor      | `torch.compile`       | 29.94    | 14.9            | 2043.0           |
| ACT     | Orin Nano | `executorch_tensorrt` | 29.92    | 26.8            | 1171.3           |
| ACT     | Orin Nano | `executorch_cuda`     | 29.02    | 132.4           | 960.4            |
| ACT     | Orin Nano | `onnx_tensorrt`       | 29.92    | 26.6            | 751.7            |
| ACT     | Orin Nano | `torch_tensorrt`      | 29.92    | 27.8            | 1628.4           |
| ACT     | Orin Nano | PyTorch               | 29.38    | 66.2            | 1904.1           |
| SmolVLA | Thor      | `executorch_tensorrt` | 29.93    | 28.9            | 1249.9           |
| SmolVLA | Thor      | `onnx_tensorrt`       | 29.94    | 31.6            | 753.6            |
| SmolVLA | Thor      | `torch_tensorrt`      | 29.94    | 30.1            | 2417.5           |
| SmolVLA | Thor      | PyTorch               | 27.45    | 171.9           | 3068.6           |
| SmolVLA | Thor      | `torch.compile`       | n/a      | still compiling | 4033.4           |
| SmolVLA | Orin Nano | `executorch_tensorrt` | 28.31    | 126.9           | 1946.9           |
| SmolVLA | Orin Nano | `onnx_tensorrt`       | 28.27    | 130.6           | 1561.0           |
| SmolVLA | Orin Nano | `torch_tensorrt`      | 28.15    | 137.1           | 2286.1           |
| SmolVLA | Orin Nano | PyTorch               | 19.97    | 791.7           | 3386.5           |
| pi0.5   | Thor      | `executorch_tensorrt` | 28.66    | 105.2           | 1334.6           |
| pi0.5   | Thor      | `onnx_tensorrt`       | 28.30    | 128.7           | 860.6            |
| pi0.5   | Thor      | PyTorch               | 26.19    | 240.4           | 2408.2           |
| pi0.5   | Thor      | `torch.compile`       | n/a      | still compiling | 3287.7           |
| GR00T   | Thor      | `executorch_tensorrt` | 28.58    | 96.4            | 1236.3           |
| GR00T   | Thor      | `onnx_tensorrt`       | 28.51    | 99.7            | 759.4            |
| GR00T   | Thor      | `torch_tensorrt`      | 28.69    | 90.6            | 1487.0           |
| GR00T   | Thor      | PyTorch               | 25.36    | 213.4           | 2983.4           |
| GR00T   | Thor      | `torch.compile`       | n/a      | still compiling | 3635.8           |

ONNX-TensorRT uses the least process memory wherever it ran: 644.8 MiB against 1107.6 MiB for ExecuTorch-TensorRT on ACT (Thor), 751.7 against 1171.3 on ACT (Orin Nano), 753.6 against 1249.9 on SmolVLA (Thor), 1561.0 against 1946.9 on SmolVLA (Orin Nano), 860.6 against 1334.6 on pi0.5, and 759.4 against 1236.3 on GR00T. ExecuTorch-CUDA also uses less memory than ExecuTorch-TensorRT on ACT, but it is 11 times slower per chunk on Thor and 5 times slower on the Orin Nano. ExecuTorch-TensorRT uses less memory than Torch-TensorRT, PyTorch and `torch.compile` in every rollout.

Rollout inference is close between the TensorRT routes on ACT, SmolVLA and GR00T. On ACT, ONNX-TensorRT was 0.1 ms lower on Thor and 0.2 ms lower on the Orin Nano. On SmolVLA, ExecuTorch-TensorRT was 1.2 ms lower than Torch-TensorRT and 2.7 ms lower than ONNX-TensorRT on Thor, and 3.7 and 10.2 ms lower than ONNX-TensorRT and Torch-TensorRT on the Orin Nano. On GR00T, Torch-TensorRT was 5.8 ms lower than ExecuTorch-TensorRT, which was 3.3 ms lower than ONNX-TensorRT. On pi0.5 the gap is larger: 105.2 ms with ExecuTorch-TensorRT against 128.7 ms with ONNX-TensorRT. pi0.5 and GR00T take about 100 ms per chunk with ExecuTorch-TensorRT, longer than one 33 ms control tick, so their loops ran a little under 30 Hz: 28.66 and 28.58 Hz, against 26.19 and 25.36 Hz with PyTorch (28.30 and 28.51 Hz with ONNX-TensorRT, 28.69 Hz for GR00T with Torch-TensorRT). SmolVLA on the Orin Nano also takes longer than a tick, so its TensorRT loops ran at 28.15 to 28.31 Hz, and PyTorch at 19.97 Hz. `torch.compile` compiles during the first chunk of a rollout. That took 215 seconds for SmolVLA, 259 seconds for pi0.5 and 103 seconds for GR00T, longer than the 60 second run, so those rows have no loop rate; their compiled chunk times are in the table above.

### Action agreement

Largest and mean absolute difference from the PyTorch policy, in robot action units, over every action of 50 dataset observations. SmolVLA, pi0.5 and GR00T start from random noise, so both sides get the same noise. Every measured folder passed: its largest difference stayed under the tolerance saved in the folder (0.5 for ACT, 5 for the others). That is under 0.1% of a joint's range for ACT and at most 3.1% for SmolVLA, pi0.5 and GR00T. The last column is the mean difference between PyTorch and itself under a different noise draw, for scale. pi0.5 and GR00T run in bf16; against fp32 PyTorch their largest and mean differences with ExecuTorch-TensorRT are 2.18 and 0.112 for pi0.5, and 3.74 and 0.115 for GR00T (the same for GR00T with Torch-TensorRT; 2.09 and 0.113, and 2.55 and 0.111, with ONNX-TensorRT). These checks show that the export computes the same actions as PyTorch, not that the robot completes the task.

| policy  | device    | `executorch_tensorrt` | `executorch_cuda` | `onnx_tensorrt` | `torch_tensorrt` | PyTorch, other noise |
| ------- | --------- | --------------------- | ----------------- | --------------- | ---------------- | -------------------- |
| ACT     | Thor      | 0.11 / 0.005          | 0.13 / 0.003      | 0.12 / 0.005    | 0.11 / 0.005     | no noise             |
| ACT     | Orin Nano | 0.10 / 0.005          | 0.07 / 0.002      | 0.11 / 0.005    | 0.10 / 0.005     | no noise             |
| SmolVLA | Thor      | 2.23 / 0.070          | no script         | 1.46 / 0.061    | 2.23 / 0.070     | 1.84                 |
| SmolVLA | Orin Nano | 1.59 / 0.054          | no script         | 1.46 / 0.058    | 1.59 / 0.054     | 1.83                 |
| pi0.5   | Thor      | 2.98 / 0.108          | pending           | 4.80 / 0.125    | pending          | 5.61                 |
| pi0.5   | Orin Nano | pending               | pending           | pending         | pending          | pending              |
| GR00T   | Thor      | 2.97 / 0.092          | pending           | 1.60 / 0.088    | 2.97 / 0.092     | 3.49                 |
| GR00T   | Orin Nano | pending               | pending           | pending         | pending          | pending              |

## Which routes load PyTorch

Distinct `libtorch` and `libc10` libraries mapped by the running `lerobot-rollout` process, read from `/proc/<pid>/maps` in every rollout above:

| route                    | mapped PyTorch libraries |
| ------------------------ | ------------------------ |
| `onnx_tensorrt`          | 0                        |
| `executorch_tensorrt`    | 8                        |
| `executorch_cuda` (ACT)  | 8                        |
| `torch_tensorrt`         | 8                        |
| PyTorch, `torch.compile` | 8                        |

The public ExecuTorch Python bindings load PyTorch today.

## Limits

- **Orin Nano memory.** The board has 8 GB. ACT exported and ran there on all four routes. SmolVLA's three TensorRT exports finished only by using swap, with peaks of 4.0 to 5.6 GiB. SmolVLA `torch.compile`, pi0.5 and GR00T are still pending there. Exporting the graph elsewhere does not remove the target engine builder's memory needs.
- **GR00T resize runs outside the program**, using NumPy antialiased bicubic sampling because the operator is not supported by every compiler route. The chunk times above include it.
- **Legacy SmolVLA text-step folders load PyTorch** for tokenization, including with ONNX-TensorRT. Current raw-frame exports keep the fixed task's tokens in the program.
- **CUDA route coverage is incomplete.** SmolVLA has no `executorch_cuda` script. GR00T's raw-frame CUDA export failed with the stock compiler in an earlier test; this run's retry is pending.
- **Replay and saved-case agreement do not prove physical task success.** Older pi0.5 rollouts stopped at camera validation. In this run the pi0.5 checkpoint already used the robot's camera names, so its rollouts needed no rename map.
