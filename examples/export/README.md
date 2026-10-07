# Compiled LeRobot policies on an NVIDIA Jetson

Export a trained LeRobot policy, then run the compiled folder with `lerobot-rollout`, the same command used for a PyTorch checkpoint. These examples cover ACT, SmolVLA, pi0.5 and GR00T N1.7 through several compiler routes. Latency, memory use and export support depend on the policy, route and device; some results below are still pending.

## Hardware

- [SO-101](../../docs/source/so101.mdx) arm with a scene camera and a wrist camera
- [Jetson AGX Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/), to train, export and run
- [Jetson Orin Nano Super](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/), to export and run ACT and SmolVLA (see Limits)

## Routes

A route is how the policy is compiled. Each writes a folder that `lerobot-rollout` can load.

| route                 | what it writes                                    | loads PyTorch at runtime                    |
| --------------------- | ------------------------------------------------- | ------------------------------------------- |
| `executorch_tensorrt` | Torch-TensorRT into an ExecuTorch `.pte` program  | yes, through the public ExecuTorch bindings |
| `onnx_tensorrt`       | ONNX, then a TensorRT `.engine`                   | no for raw-frame folders; see Limits        |
| `torch_tensorrt`      | Torch-TensorRT into an AOTInductor `.pt2` package | yes                                         |

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
python examples/export/act_onnx_tensorrt.py --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model --output_dir=outputs/export/act_onnx_tensorrt
python examples/export/act_torch_tensorrt.py --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model --output_dir=outputs/export/act_torch_tensorrt
```

| policy  | `executorch_tensorrt`                    | `onnx_tensorrt`                    | `torch_tensorrt`                    |
| ------- | ---------------------------------------- | ---------------------------------- | ----------------------------------- |
| ACT     | [script](act_executorch_tensorrt.py)     | [script](act_onnx_tensorrt.py)     | [script](act_torch_tensorrt.py)     |
| SmolVLA | [script](smolvla_executorch_tensorrt.py) | [script](smolvla_onnx_tensorrt.py) | [script](smolvla_torch_tensorrt.py) |
| pi0.5   | [script](pi05_executorch_tensorrt.py)    | [script](pi05_onnx_tensorrt.py)    | [script](pi05_torch_tensorrt.py)    |
| GR00T   | [script](groot_executorch_tensorrt.py)   | [script](groot_onnx_tensorrt.py)   | [script](groot_torch_tensorrt.py)   |

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

The omitted camera slot uses the policy's own padding. Map the robot's names to these slots at rollout, as shown below.

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
> Cells marked pending are still being measured.

Five ways to run each policy: three TensorRT routes, `torch.compile` and PyTorch. Bold marks the best value in each row. Thor runs at MAXN power with locked clocks; the Orin Nano runs with a locked GPU clock.

### Accuracy

Difference from PyTorch's actions on 50 dataset samples, with the same random noise: largest / average, in degrees (gripper in %). Lower is better.

Thor:

|         | ExecuTorch-TensorRT |    ONNX-TensorRT | Torch-TensorRT |      `torch.compile` |
| ------- | ------------------: | ---------------: | -------------: | -------------------: |
| ACT     |        0.11 / 0.005 |     0.12 / 0.005 |   0.11 / 0.005 | **0.06** / **0.001** |
| SmolVLA |        2.23 / 0.070 | **1.46** / 0.061 |   2.23 / 0.070 |     1.83 / **0.057** |
| pi0.5   |        2.98 / 0.108 |     4.80 / 0.125 |   2.98 / 0.108 | **1.60** / **0.099** |
| GR00T   |        2.97 / 0.092 |     1.60 / 0.088 |   2.97 / 0.092 | **1.08** / **0.068** |

Orin Nano:

|         | ExecuTorch-TensorRT |    ONNX-TensorRT |   Torch-TensorRT |      `torch.compile` |
| ------- | ------------------: | ---------------: | ---------------: | -------------------: |
| ACT     |        0.10 / 0.005 |     0.11 / 0.005 |     0.10 / 0.005 | **0.07** / **0.002** |
| SmolVLA |    1.59 / **0.054** | **1.46** / 0.058 | 1.59 / **0.054** |         2.05 / 0.059 |
| pi0.5   |             pending |          pending |          pending |              pending |
| GR00T   |             pending |          pending |          pending |              pending |

All routes match PyTorch within the check that runs before the robot moves.

### Latency

Median time to compute one action chunk, in milliseconds. Lower is better.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |            **3.43** |          3.50 |           3.70 |           12.80 |   15.91 |
| SmolVLA |           **28.17** |         30.95 |          29.20 |           44.47 |   166.7 |
| pi0.5   |           **105.4** |         124.7 |          106.3 |           129.9 |   244.2 |
| GR00T   |               93.30 |         97.84 |      **88.98** |           206.7 |   211.5 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |           **25.65** |         26.36 |          26.26 |           61.53 |   64.60 |
| SmolVLA |           **125.6** |         130.5 |          136.0 |           172.3 |   762.0 |
| pi0.5   |             pending |       pending |        pending |         pending | pending |
| GR00T   |             pending |       pending |        pending |         pending | pending |

ExecuTorch-TensorRT is 2 to 6 times faster than PyTorch, and the fastest route on ACT, SmolVLA and pi0.5. Torch-TensorRT is the fastest on GR00T.

### Memory

Board memory used while running, in MiB, the highest across runs. Lower is better. ExecuTorch-TensorRT cells read N (M): N is the estimate once ExecuTorch removes PyTorch from its Python bindings, and M is measured today. Bold compares the first number.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |       **520** (829) |           557 |           1510 |            2489 |    2327 |
| SmolVLA |     **1190** (1535) |          1278 |           3233 |            4422 |    3940 |
| pi0.5   |     **5790** (6082) |          6230 |           6247 |           12215 |   11610 |
| GR00T   |         5700 (6007) |      **5486** |           6190 |           16684 |   15402 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |           330 (493) |       **223** |            579 |            1651 |    1140 |
| SmolVLA |           790 (940) |       **370** |           1499 |            2818 |    2034 |
| pi0.5   |             pending |       pending |        pending |         pending | pending |
| GR00T   |             pending |       pending |        pending |         pending | pending |

ExecuTorch-TensorRT uses about half of PyTorch's memory or less. Today ONNX-TensorRT uses less, as the only route that does not load PyTorch. Without PyTorch, ExecuTorch-TensorRT would use the least on Thor for ACT, SmolVLA and pi0.5.

## Limits

- **Orin Nano memory.** The board has 8 GB. SmolVLA needs swap to compile there. pi0.5 and GR00T are still being measured.
- **GR00T resizes images outside the program**, with NumPy, because not every route supports that operator. Its latency includes the resize.
- **Older SmolVLA folders load PyTorch** for tokenization, even with ONNX-TensorRT. New exports keep the task's tokens in the program.
- **These checks prove the compiled policy matches PyTorch, not that the robot completes the task.**
