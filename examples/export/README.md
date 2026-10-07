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
> Partly provisional. Every number below was measured on 2026-10-06 with this fork and with ExecuTorch and Torch-TensorRT built from source, including the fixes the install step expects in the nightlies. Every Thor cell is measured. Cells marked "pending" are pi0.5 and GR00T on the Orin Nano, which are still being exported there.

Thor: JetPack 7.2.1, power mode MAXN, CPU, GPU and memory clocks locked. Orin Nano Super: JetPack 7.2.1, GPU clock locked at 1020 MHz. Every folder was exported on the board that ran it.

### Latency per action chunk, ms

The median time to compute one action chunk: 100 chunks after 20 warm-up chunks in each of 3 fresh processes, then the median of the 3. Loading and compiling are not included, and this is not control-loop latency. `torch.compile` wraps the chunk call in mode `default` on Thor (the `lerobot-rollout` default) and `max-autotune` on the Orin Nano. SmolVLA and pi0.5 also turn on their own `compile_model` setting (mode `max-autotune`), and pi0.5 uses only that, as `lerobot-rollout --use_torch_compile` does. On Thor, pi0.5 and GR00T `torch.compile` ran in 1 process. GR00T times include its image resize (see Limits).

| policy  | device    | `executorch_tensorrt` | `executorch_cuda` | `onnx_tensorrt` | `torch_tensorrt` | PyTorch | `torch.compile` |
| ------- | --------- | --------------------- | ----------------- | --------------- | ---------------- | ------- | --------------- |
| ACT     | Thor      | 3.43                  | 37.64             | 3.50            | 3.70             | 15.91   | 12.80           |
| ACT     | Orin Nano | 25.65                 | 131.5             | 26.36           | 26.26            | 64.60   | 61.53           |
| SmolVLA | Thor      | 28.17                 | no script         | 30.95           | 29.20            | 166.7   | 44.47           |
| SmolVLA | Orin Nano | 125.6                 | no script         | 130.5           | 136.0            | 762.0   | 172.3           |
| pi0.5   | Thor      | 105.4                 | 175.1             | 124.7           | 106.3            | 244.2   | 129.9           |
| pi0.5   | Orin Nano | pending               | pending           | pending         | pending          | pending | pending         |
| GR00T   | Thor      | 93.30                 | export failed     | 97.84           | 88.98            | 211.5   | 206.7           |
| GR00T   | Orin Nano | pending               | pending           | pending         | pending          | pending | pending         |

ExecuTorch-TensorRT has the lowest chunk time of the TensorRT routes on ACT and SmolVLA on both boards and on pi0.5. On GR00T, Torch-TensorRT is faster. On ACT its lead over the next TensorRT route is small, about 2%: 3.43 against 3.50 ms for ONNX-TensorRT on Thor, and 25.65 against 26.26 ms for Torch-TensorRT on the Orin Nano. On SmolVLA it takes 3.5% less time than Torch-TensorRT and 9% less than ONNX-TensorRT on Thor, and 3.7% less than ONNX-TensorRT and 7.6% less than Torch-TensorRT on the Orin Nano. On pi0.5 it is level with Torch-TensorRT, 0.9% less (105.4 against 106.3 ms), and takes 15% less time than ONNX-TensorRT (124.7 ms). On GR00T, Torch-TensorRT takes 4.6% less time than ExecuTorch-TensorRT (88.98 against 93.30 ms), which takes 4.6% less than ONNX-TensorRT (97.84 ms). It is faster than PyTorch in every measured row: 4.6 times on ACT (Thor), 2.5 times on ACT (Orin Nano), 5.9 times on SmolVLA (Thor), 6.1 times on SmolVLA (Orin Nano), 2.3 times on pi0.5 and 2.3 times on GR00T. It also beats `torch.compile` in every measured row, by 1.2 times (pi0.5) to 3.7 times (ACT on Thor). ExecuTorch-CUDA is slower than every TensorRT route: 11 times ExecuTorch-TensorRT's chunk time on ACT (Thor), 5.1 times on ACT (Orin Nano) and 1.7 times on pi0.5.

### Rollout latency and memory

The official `lerobot-rollout` with the [replay robot](replay_robot) at 30 Hz for 60 seconds. Inference is the median time per chunk on the status line at the end of the run (up to the last 30 chunks). Board memory is how much the whole board's available memory (`MemAvailable` in `/proc/meminfo`) dropped from just before the process started to the median of the final 30 seconds. Load peak is the largest drop at any point, loading included; it decides what fits on the 8 GB Orin Nano. Each value is the median of 3 runs, with the lowest and highest in parentheses, and n=2 where only 2 runs counted. We measure the whole board because process RSS (resident memory) on a Jetson misses some GPU allocations. For ExecuTorch, the value after the semicolon is an estimate without PyTorch, explained below the table.

| policy  | device    | route                 | loop, Hz | inference, ms   | board memory, MiB                              | load peak, MiB              |
| ------- | --------- | --------------------- | -------- | --------------- | ---------------------------------------------- | --------------------------- |
| ACT     | Thor      | `executorch_tensorrt` | 29.94    | 4.1             | 816 (805 to 829; est. 520 without PyTorch)     | 817 (807 to 830)            |
| ACT     | Thor      | `executorch_cuda`     | 29.89    | 38.7            | 845 (777 to 846; est. 560 without PyTorch)     | 848 (814 to 854)            |
| ACT     | Thor      | `onnx_tensorrt`       | 29.94    | 4.0             | 544 (529 to 557)                               | 590 (576 to 593)            |
| ACT     | Thor      | `torch_tensorrt`      | 29.94    | 4.6             | 1400 (1273 to 1510)                            | 1983 (1798 to 2050)         |
| ACT     | Thor      | PyTorch               | 29.78    | 16.4            | 2032 (2024 to 2327)                            | 2032 (2026 to 2364)         |
| ACT     | Thor      | `torch.compile`       | 29.94    | 14.9            | 2462 (2442 to 2489)                            | 2466 (2449 to 2494)         |
| ACT     | Orin Nano | `executorch_tensorrt` | 29.92    | 26.8            | 489 (462 to 493; est. 330 without PyTorch)     | 504 (493 to 789)            |
| ACT     | Orin Nano | `executorch_cuda`     | 29.02    | 132.4           | 440 (435 to 446; est. 290 without PyTorch)     | 445 (445 to 457)            |
| ACT     | Orin Nano | `onnx_tensorrt`       | 29.92    | 26.6            | 216 (208 to 223)                               | 268 (237 to 268)            |
| ACT     | Orin Nano | `torch_tensorrt`      | 29.92    | 27.8            | 457 (397 to 579)                               | 1307 (1199 to 1493)         |
| ACT     | Orin Nano | PyTorch               | 29.38    | 66.2            | 1117 (1111 to 1140)                            | 1206 (1120 to 1234)         |
| SmolVLA | Thor      | `executorch_tensorrt` | 29.93    | 28.9            | 1534 (1532 to 1535; est. 1190 without PyTorch) | 1547 (1541 to 1609)         |
| SmolVLA | Thor      | `onnx_tensorrt`       | 29.94    | 31.6            | 1277 (1270 to 1278)                            | 1309 (1303 to 1315)         |
| SmolVLA | Thor      | `torch_tensorrt`      | 29.94    | 30.1            | 2740 (2554 to 3233)                            | 6104 (5615 to 6190)         |
| SmolVLA | Thor      | PyTorch               | 27.45    | 171.9           | 3719 (3589 to 3940)                            | 3887 (3721 to 3977)         |
| SmolVLA | Thor      | `torch.compile`       | n/a      | still compiling | 4305 (4256 to 4354, n=2)                       | 4360 (4266 to 4454, n=2)    |
| SmolVLA | Orin Nano | `executorch_tensorrt` | 28.31    | 126.9           | 884 (636 to 940)                               | 891 (650 to 942)            |
| SmolVLA | Orin Nano | `onnx_tensorrt`       | 28.27    | 130.6           | 354 (314 to 370)                               | 390 (376 to 404)            |
| SmolVLA | Orin Nano | `torch_tensorrt`      | 28.15    | 137.1           | 1338 (1314 to 1499)                            | 5381 (5339 to 5396)         |
| SmolVLA | Orin Nano | PyTorch               | 19.97    | 791.7           | 1983 (1902 to 2034)                            | 2089 (1905 to 2094)         |
| pi0.5   | Thor      | `executorch_tensorrt` | 28.66    | 105.2           | 6081 (6072 to 6082; est. 5790 without PyTorch) | 6084 (6084 to 6090)         |
| pi0.5   | Thor      | `executorch_cuda`     | 27.51    | 176.3           | 7356 (7342 to 7357)                            | 7358 (7344 to 7358)         |
| pi0.5   | Thor      | `onnx_tensorrt`       | 28.30    | 128.7           | 6217 (6204 to 6230)                            | 6322 (6265 to 6514)         |
| pi0.5   | Thor      | `torch_tensorrt`      | 28.57    | 112.1           | 6237 (5976 to 6247)                            | 42586 (42365 to 42633)      |
| pi0.5   | Thor      | PyTorch               | 26.19    | 240.4           | 11609 (11550 to 11610)                         | 11845 (11556 to 11975)      |
| pi0.5   | Thor      | `torch.compile`       | n/a      | still compiling | 12195 (12187 to 12215)                         | 12206 (12191 to 12243)      |
| GR00T   | Thor      | `executorch_tensorrt` | 28.58    | 96.4            | 6007 (5979 to 6007; est. 5700 without PyTorch) | 6031 (6012 to 6043)         |
| GR00T   | Thor      | `onnx_tensorrt`       | 28.51    | 99.7            | 5445 (5427 to 5486)                            | 5783 (5445 to 5793)         |
| GR00T   | Thor      | `torch_tensorrt`      | 28.69    | 90.6            | 6169 (6162 to 6190)                            | 43342 (43326 to 43372)      |
| GR00T   | Thor      | PyTorch               | 25.36    | 213.4           | 15275 (15178 to 15402)                         | 16126 (15282 to 16145)      |
| GR00T   | Thor      | `torch.compile`       | n/a      | still compiling | 16682 (16680 to 16684, n=2)                    | 16689 (16688 to 16689, n=2) |

ExecuTorch-TensorRT uses about half the board memory of PyTorch or less in every measured row, and half of `torch.compile`'s or less: 816 against 2032 MiB on ACT, 1534 against 3719 MiB on SmolVLA, 6081 against 11609 MiB on pi0.5 and 6007 against 15275 MiB on GR00T on Thor, and 489 against 1117 MiB on ACT and 884 against 1983 MiB on SmolVLA on the Orin Nano.

ONNX-TensorRT uses less board memory than ExecuTorch-TensorRT on ACT, SmolVLA and GR00T: 544 against 816 MiB, 1277 against 1534 MiB and 5445 against 6007 MiB on Thor, and 216 against 489 MiB and 354 against 884 MiB on the Orin Nano. On pi0.5 the two are close: 6217 MiB for ONNX-TensorRT against 6081 MiB for ExecuTorch-TensorRT. ONNX-TensorRT is the only route that does not load PyTorch, which is the likely reason.

The public ExecuTorch Python bindings load PyTorch today. ExecuTorch is working to remove that dependency, and its first step, accepting Python buffers as inputs, is already merged. Subtracting the memory those PyTorch libraries take gives an estimate, not a measurement: ExecuTorch-TensorRT would use about 520 MiB on ACT, 1190 MiB on SmolVLA, 5790 MiB on pi0.5 and 5700 MiB on GR00T on Thor, and about 330 MiB on ACT on the Orin Nano. ExecuTorch-CUDA would use about 560 MiB on ACT on Thor and about 290 MiB on the Orin Nano. With these estimates, ExecuTorch-TensorRT would use the least memory on Thor for ACT, SmolVLA and pi0.5, while ONNX-TensorRT would still use the least on GR00T, and on ACT on the Orin Nano.

Torch-TensorRT's load peak reaches about 42 GiB on pi0.5 and GR00T on Thor (42586 and 43342 MiB), about 7 times its steady memory, just before the loop starts. ExecuTorch-TensorRT's load peak stays within 25 MiB of its steady memory on both. On the Orin Nano, Torch-TensorRT peaks at 5381 MiB while loading SmolVLA, against 891 MiB for ExecuTorch-TensorRT and 390 MiB for ONNX-TensorRT.

The `torch.compile` memory runs reused a warm compile cache, so they finished compiling in under 50 seconds. On SmolVLA and GR00T, one of the 3 runs was still compiling for most of the final 30 seconds, so those rows use the other 2 runs (n=2).

Rollout inference is close between the TensorRT routes on ACT, SmolVLA and GR00T, and between ExecuTorch-TensorRT and Torch-TensorRT on pi0.5. On ACT, ONNX-TensorRT was 0.1 ms lower on Thor and 0.2 ms lower on the Orin Nano. On SmolVLA, ExecuTorch-TensorRT was 1.2 ms lower than Torch-TensorRT and 2.7 ms lower than ONNX-TensorRT on Thor, and 3.7 and 10.2 ms lower than ONNX-TensorRT and Torch-TensorRT on the Orin Nano. On GR00T, Torch-TensorRT was 5.8 ms lower than ExecuTorch-TensorRT, which was 3.3 ms lower than ONNX-TensorRT. On pi0.5, ExecuTorch-TensorRT was 6.9 ms lower than Torch-TensorRT (105.2 against 112.1 ms) and 23.5 ms lower than ONNX-TensorRT (128.7 ms). pi0.5 and GR00T take about 100 ms per chunk with ExecuTorch-TensorRT, longer than one 33 ms control tick, so their loops ran a little under 30 Hz: 28.66 and 28.58 Hz, against 26.19 and 25.36 Hz with PyTorch (28.30 and 28.51 Hz with ONNX-TensorRT, 28.57 and 28.69 Hz with Torch-TensorRT). pi0.5 with ExecuTorch-CUDA took 176.3 ms per chunk and ran at 27.51 Hz. SmolVLA on the Orin Nano also takes longer than a tick, so its TensorRT loops ran at 28.15 to 28.31 Hz, and PyTorch at 19.97 Hz. `torch.compile` compiles during the first chunk of a rollout. That took 215 seconds for SmolVLA, 259 seconds for pi0.5 and 103 seconds for GR00T, longer than the 60 second run, so those rows have no loop rate; their compiled chunk times are in the table above.

### Action agreement

Largest and mean absolute difference from the PyTorch policy, in robot action units, over every action of 50 dataset observations. SmolVLA, pi0.5 and GR00T start from random noise, so both sides get the same noise. Every measured folder passed: its largest difference stayed under the tolerance saved in the folder (0.5 for ACT, 5 for the others). That is under 0.1% of a joint's range for ACT and at most 3.1% for SmolVLA, pi0.5 and GR00T. The last column is the mean difference between PyTorch and itself under a different noise draw, for scale. pi0.5 and GR00T run in bf16; against fp32 PyTorch their largest and mean differences with ExecuTorch-TensorRT are 2.18 and 0.112 for pi0.5, and 3.74 and 0.115 for GR00T (the same with Torch-TensorRT; 2.09 and 0.113, and 2.55 and 0.111, with ONNX-TensorRT; 2.99 and 0.102 for pi0.5 with ExecuTorch-CUDA). These checks show that the export computes the same actions as PyTorch, not that the robot completes the task.

| policy  | device    | `executorch_tensorrt` | `executorch_cuda` | `onnx_tensorrt` | `torch_tensorrt` | PyTorch, other noise |
| ------- | --------- | --------------------- | ----------------- | --------------- | ---------------- | -------------------- |
| ACT     | Thor      | 0.11 / 0.005          | 0.13 / 0.003      | 0.12 / 0.005    | 0.11 / 0.005     | no noise             |
| ACT     | Orin Nano | 0.10 / 0.005          | 0.07 / 0.002      | 0.11 / 0.005    | 0.10 / 0.005     | no noise             |
| SmolVLA | Thor      | 2.23 / 0.070          | no script         | 1.46 / 0.061    | 2.23 / 0.070     | 1.84                 |
| SmolVLA | Orin Nano | 1.59 / 0.054          | no script         | 1.46 / 0.058    | 1.59 / 0.054     | 1.83                 |
| pi0.5   | Thor      | 2.98 / 0.108          | 3.33 / 0.102      | 4.80 / 0.125    | 2.98 / 0.108     | 5.61                 |
| pi0.5   | Orin Nano | pending               | pending           | pending         | pending          | pending              |
| GR00T   | Thor      | 2.97 / 0.092          | export failed     | 1.60 / 0.088    | 2.97 / 0.092     | 3.49                 |
| GR00T   | Orin Nano | pending               | pending           | pending         | pending          | pending              |

## Which routes load PyTorch

Distinct `libtorch` and `libc10` libraries mapped by the running `lerobot-rollout` process, read from `/proc/<pid>/maps` in every rollout above:

| route                    | mapped PyTorch libraries |
| ------------------------ | ------------------------ |
| `onnx_tensorrt`          | 0                        |
| `executorch_tensorrt`    | 8                        |
| `executorch_cuda`        | 8                        |
| `torch_tensorrt`         | 8                        |
| PyTorch, `torch.compile` | 8                        |

The public ExecuTorch Python bindings load PyTorch today.

## Limits

- **Orin Nano memory.** The board has 8 GB. ACT exported and ran there on all four routes. SmolVLA's three TensorRT exports finished only by using swap, with peaks of 4.0 to 5.6 GiB. When running, the load peaks there are 504 MiB for ACT and 891 MiB for SmolVLA with ExecuTorch-TensorRT, 268 and 390 MiB with ONNX-TensorRT, and 1307 and 5381 MiB with Torch-TensorRT. pi0.5 and GR00T are still pending there. Exporting the graph elsewhere does not remove the target engine builder's memory needs.
- **GR00T resize runs outside the program**, using NumPy antialiased bicubic sampling because the operator is not supported by every compiler route. The chunk times above include it.
- **Legacy SmolVLA text-step folders load PyTorch** for tokenization, including with ONNX-TensorRT. Current raw-frame exports keep the fixed task's tokens in the program.
- **CUDA route coverage is incomplete.** SmolVLA has no `executorch_cuda` script. GR00T's `executorch_cuda` export fails on Thor: the stock compiler stops with `AttributeError: 'NotImplementedType' object has no attribute 'expr'`.
- **Replay and saved-case agreement do not prove physical task success.** Older pi0.5 rollouts stopped at camera validation. In this run the pi0.5 checkpoint already used the robot's camera names, so its rollouts needed no rename map.
