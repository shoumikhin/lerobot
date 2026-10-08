# Compiled LeRobot policies on an NVIDIA Jetson

Export a trained LeRobot policy, then run the compiled folder with `lerobot-rollout`, the same command used for a PyTorch checkpoint. These examples cover ACT, SmolVLA, pi0.5 and GR00T N1.7 through several compiler routes. Latency, memory use and export support depend on the policy, route and device. The results distinguish measured routes from checks that have not run.

## Hardware

- [SO-101](../../docs/source/so101.mdx) arm with a scene camera and a wrist camera
- [Jetson AGX Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/), to train, export and run
- [Jetson Orin Nano Super](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/), to export and run the policies that fit with the split recipes below (see Limits)

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

With `--export_only`, the ACT and SmolVLA `executorch_tensorrt` and `onnx_tensorrt` scripts save the exported graph without building its TensorRT engine. Copy the folder to the target device, then build there:

```bash
python examples/export/act_onnx_tensorrt.py \
    --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model \
    --output_dir=outputs/export/act_onnx_tensorrt \
    --export_only
python examples/export/build_engine.py outputs/export/act_onnx_tensorrt
```

For these deferred exports, `--workspace_gib`, `--tactic_gib` (ONNX only) and `--optimization_level` can limit build resources, but do not guarantee the engine fits.

Every pi0.5 script exports the chunk as a chain of programs: the image and prompt embeddings, the language model in groups of three layers, one denoising step, and the actions. Each program is built in its own process, which reads only its own weights from the checkpoint, so the export runs on a device with less memory than the whole policy. `lerobot-rollout` runs the chain.

Every GR00T script builds the same chain: vision, groups of six language layers, groups of eight diffusion blocks, and actions. Each part is built in its own process. ExecuTorch-TensorRT and Torch-TensorRT keep intermediate tensors on the GPU. ONNX-TensorRT shares GPU buffers between programs and updates the diffusion state on the host. Run these split recipes on each target board; the deferred `--export_only` workflow above is for ACT and SmolVLA.

The direct ACT and SmolVLA ExecuTorch-TensorRT scripts enable CUDA graph replay because it reduced chunk latency on both boards. pi0.5 and GR00T keep replay disabled. The deferred builder keeps the default setting.

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
> Every route uses one recipe per policy. On the 8 GB Orin Nano, pi0.5 and GR00T run out of memory with `torch.compile` and PyTorch.

Five ways to run each policy: three TensorRT routes, `torch.compile` and PyTorch. Bold marks the best value in each row. pi0.5 and GR00T use a chain of small programs for all three export routes. Splitting reduces build memory, but does not guarantee the complete folder fits. Thor runs at MAXN power with locked clocks; the Orin Nano runs with a locked GPU clock.

### Accuracy

Difference from PyTorch's actions on 50 dataset samples, with the same random noise: largest / average, in degrees (gripper in %). Lower is better.

Thor:

|         | ExecuTorch-TensorRT |    ONNX-TensorRT | Torch-TensorRT |      `torch.compile` |
| ------- | ------------------: | ---------------: | -------------: | -------------------: |
| ACT     |        0.11 / 0.005 |     0.12 / 0.005 |   0.11 / 0.005 | **0.06** / **0.001** |
| SmolVLA |        2.23 / 0.070 | **1.46** / 0.061 |   2.23 / 0.070 |     1.83 / **0.057** |
| pi0.5   |        2.44 / 0.121 |     4.72 / 0.140 |   2.44 / 0.121 | **1.60** / **0.099** |
| GR00T   |        2.97 / 0.094 |     1.53 / 0.087 |   2.97 / 0.094 | **1.08** / **0.069** |

Orin Nano:

|         |  ExecuTorch-TensorRT |        ONNX-TensorRT |   Torch-TensorRT |      `torch.compile` |
| ------- | -------------------: | -------------------: | ---------------: | -------------------: |
| ACT     |         0.10 / 0.005 |         0.11 / 0.005 |     0.10 / 0.005 | **0.07** / **0.002** |
| SmolVLA |     1.59 / **0.054** |     **1.46** / 0.058 | 1.59 / **0.054** |         2.05 / 0.059 |
| pi0.5   |     3.80 / **0.138** | **3.16** / **0.138** |    fails to load |        out of memory |
| GR00T   | **1.42** / **0.094** |         2.75 / 0.107 | 1.50 / **0.094** |        out of memory |

Every numerical accuracy cell passed the 50-observation comparison. Exported folders also passed their saved startup checks; the `torch.compile` checks do not have one.

### Latency

Median of three per-process medians for one action chunk, in milliseconds. Each fresh process measures 100 chunks after 20 warmups. The Thor `torch.compile` pi0.5 cell uses one process. Lower is better.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |            **3.43** |          3.50 |           3.70 |           12.80 |   15.91 |
| SmolVLA |           **28.13** |         30.95 |          29.20 |           44.47 |   166.7 |
| pi0.5   |               120.4 |     **116.0** |          119.2 |           129.9 |   244.2 |
| GR00T   |               95.62 |         94.66 |      **93.28** |           205.8 |   211.1 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` |       PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------------: |
| ACT     |           **25.34** |         26.36 |          26.26 |           61.53 |         64.60 |
| SmolVLA |           **125.6** |         130.5 |          136.0 |           172.3 |         762.0 |
| pi0.5   |               630.7 |     **625.5** |  fails to load |   out of memory | out of memory |
| GR00T   |           **272.9** |         275.0 |          276.4 |   out of memory | out of memory |

ExecuTorch-TensorRT is the fastest route on ACT and SmolVLA on both boards, and on GR00T on the Orin Nano. ONNX-TensorRT is fastest on pi0.5 on both boards, and Torch-TensorRT on GR00T on Thor. On Thor the three TensorRT routes run GR00T within 2.4 ms of each other.

### Memory

Board memory used while running, in MiB, the highest steady value across three valid runs, plus any memory the run pushed to swap. Process memory alone misses some GPU allocations on a Jetson. Lower is better. ExecuTorch-TensorRT cells read estimated (measured): estimates subtract the previously measured PyTorch library overhead, rounded to 10 MiB; they are not direct measurements or an available measured runtime. GR00T on the Orin Nano uses the smallest overhead measured on that board. An unavailable estimate is labeled explicitly. Bold compares the first number.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |           560 (873) |       **557** |           1510 |            2489 |    2327 |
| SmolVLA |     **1190** (1534) |          1278 |           3233 |            4422 |    3940 |
| pi0.5   |         6530 (6822) |      **5914** |           7072 |           12215 |   11610 |
| GR00T   |         5470 (5775) |      **5426** |           6489 |           16138 |   15722 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` |       PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------------: |
| ACT     |           330 (490) |       **223** |            579 |            1651 |          1140 |
| SmolVLA |           520 (669) |       **370** |           1708 |            2957 |          2034 |
| pi0.5   |         5560 (5727) |      **4875** |  fails to load |   out of memory | out of memory |
| GR00T   |         4510 (4658) |      **4411** |           7605 |   out of memory | out of memory |

ONNX-TensorRT uses the least memory on most rows, as the only route that does not load PyTorch. On the Orin Nano, pi0.5 runs with ExecuTorch-TensorRT and ONNX-TensorRT, and Torch-TensorRT fails while loading it. GR00T with Torch-TensorRT needs most of the board's 8 GB and pushes about 3 GB to swap.

## Limits

- **Orin Nano memory.** The board has 8 GB. SmolVLA, pi0.5 and GR00T may need swap to compile there. The pi0.5 and GR00T recipes build one group of layers at a time, so they fit.
- **GR00T resizes images outside the program**, with NumPy, because not every route supports that operator. Its latency includes the resize.
- **Older SmolVLA folders load PyTorch** for tokenization, even with ONNX-TensorRT. New exports keep the task's tokens in the program.
- **These checks prove the compiled policy matches PyTorch, not that the robot completes the task.**
