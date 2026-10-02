# Compiled LeRobot policies on an NVIDIA Jetson

Run LeRobot policies as TensorRT engines on Jetson devices, without the policy's PyTorch code on the robot.

## Hardware

- SO-101 arm with a scene camera and a wrist camera
- Jetson AGX Thor, to train, export and run
- Jetson Orin Nano, to export and run a pretrained policy

## Backends

- `onnx_tensorrt`: ONNX using `torch.export`, then TensorRT, into an `.engine` file
- `executorch_tensorrt`: `torch.export`, then Torch-TensorRT, into an ExecuTorch `.pte` file
- `aoti_tensorrt`: `torch.export`, then Torch-TensorRT, into an AOTInductor `.pt2` package. Unlike the other two, its file runs only with PyTorch and Torch-TensorRT installed. A `.pte` or `.engine` file can also run without them.

## Policies

- [ACT](../../docs/source/act.mdx)
- [SmolVLA](../../docs/source/smolvla.mdx)
- [pi0.5](../../docs/source/pi05.mdx)
- [GR00T N1.7](../../docs/source/groot.mdx)

## Install

On each Jetson, with [uv](https://docs.astral.sh/uv/) installed:

```bash
git clone --branch export https://github.com/shoumikhin/lerobot.git
cd lerobot
uv venv --python 3.12
source .venv/bin/activate
uv pip install \
    -e ".[smolvla,pi,groot,feetech]" \
    "torch-tensorrt[executorch]" onnx onnxscript \
    --index-url https://download.pytorch.org/whl/nightly/cu132 \
    --extra-index-url https://pypi.org/simple \
    --extra-index-url https://pypi.nvidia.com \
    --index-strategy unsafe-best-match \
    --prerelease allow
```

This brings nightly PyTorch, Torch-TensorRT, ExecuTorch, TensorRT and ONNX for CUDA 13.2.

## Export

Export a policy you trained with `lerobot-train` on each device it's intended to run, because a TensorRT engine only runs on the GPU model that built it:

```bash
python examples/export/act_executorch_tensorrt.py \
    --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model \
    --output_dir=outputs/export/act_executorch_tensorrt
```

Every policy has one script per backend:

| policy  | `executorch_tensorrt`                                            | `onnx_tensorrt`                                      | `aoti_tensorrt`                                      |
| ------- | ---------------------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| ACT     | [act_executorch_tensorrt.py](act_executorch_tensorrt.py)         | [act_onnx_tensorrt.py](act_onnx_tensorrt.py)         | [act_aoti_tensorrt.py](act_aoti_tensorrt.py)         |
| SmolVLA | [smolvla_executorch_tensorrt.py](smolvla_executorch_tensorrt.py) | [smolvla_onnx_tensorrt.py](smolvla_onnx_tensorrt.py) | [smolvla_aoti_tensorrt.py](smolvla_aoti_tensorrt.py) |
| pi0.5   | [pi05_executorch_tensorrt.py](pi05_executorch_tensorrt.py)       | [pi05_onnx_tensorrt.py](pi05_onnx_tensorrt.py)       | [pi05_aoti_tensorrt.py](pi05_aoti_tensorrt.py)       |
| GR00T   | [groot_executorch_tensorrt.py](groot_executorch_tensorrt.py)     | [groot_onnx_tensorrt.py](groot_onnx_tensorrt.py)     | [groot_aoti_tensorrt.py](groot_aoti_tensorrt.py)     |

SmolVLA also needs the dataset it was trained on, because its checkpoint does not record the robot's cameras. The export reads their names, their sizes and the task from it:

```bash
python examples/export/smolvla_executorch_tensorrt.py \
    --policy.path=outputs/train/smolvla_so101/checkpoints/last/pretrained_model \
    --dataset.repo_id=<user>/so101_dataset \
    --output_dir=outputs/export/smolvla_executorch_tensorrt
```

Add `--dataset.root=<folder>` if the dataset is only on your disk.

GR00T takes the same options as SmolVLA. Its export also fixes the task and the camera size, because the program holds the task's tokens and the image layout as constants:

```bash
python examples/export/groot_executorch_tensorrt.py \
    --policy.path=outputs/train/groot_so101/checkpoints/last/pretrained_model \
    --dataset.repo_id=<user>/so101_dataset \
    --output_dir=outputs/export/groot_executorch_tensorrt
```

pi0.5's checkpoint names the robot's cameras but not the task, so give the task it was trained on:

```bash
python examples/export/pi05_executorch_tensorrt.py \
    --policy.path=outputs/train/pi05_so101/checkpoints/last/pretrained_model \
    --task="Pick up the block and place it in the cup" \
    --output_dir=outputs/export/pi05_executorch_tensorrt
```

pi0.5 and GR00T do not run on the Orin Nano. Both export scripts load the PyTorch policy, which does not fit in its 7.3 GiB. TensorRT also runs out of GPU memory when it builds an engine on the Orin Nano from an ONNX file exported on the Thor, about 5 GiB, and an engine built on the Thor does not run there. Export and run them on the Thor.

### Export once, build on each device

To export once and build on several devices, export with `--export_only`: the script writes the folder without the engine. The ONNX route keeps `model.onnx` and its weights, and the ExecuTorch route saves the exported program as `model.pt2`. Copy the folder to each device, then build the engine there:

```bash
python examples/export/act_onnx_tensorrt.py \
    --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model \
    --output_dir=outputs/export/act_onnx_tensorrt \
    --export_only
python examples/export/build_engine.py outputs/export/act_onnx_tensorrt \
    --workspace_gib=2 --optimization_level=3
```

`build_engine.py` builds the engine for the local GPU, writes the file the rollout loads, and checks it against the test case the export saved. Lower `--workspace_gib`, `--tactic_gib` (ONNX only) and `--optimization_level` make the build need less memory, and the engine may run slower. Every ExecuTorch and ONNX export script takes `--export_only`, with the same options as without it.

## Run

Then run the exported folder like any LeRobot policy, with the same `--robot.*` options you recorded with, including `--robot.cameras`:

```bash
lerobot-rollout \
    --policy.path=outputs/export/act_executorch_tensorrt \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, scene: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }" \
    --duration=30
```

Before the robot moves, the rollout replays a test case saved at export time, and stops if the actions differ from what the PyTorch policy produced.

For SmolVLA and pi0.5, give the task as you would with the PyTorch policy, for example `--task="Pick up the brick and put it in the bin"`.

For GR00T, give the task it was exported for. The rollout refuses any other task, so export again to change it.

To compare backends, export with another script and run its folder.

## Benchmarks

Time for one action chunk, and the memory a process adds to run it. Each cell is the median of 3 runs unless marked 1 run, each in a fresh process, through the rollout's own inference engines. Each run times at least 100 chunks after 20 warmup chunks. The policies were trained on an SO-101 arm with two cameras.

Setup: JetPack 7.2.1, with the nightly PyTorch 2.15, Torch-TensorRT, ExecuTorch and TensorRT 11.3 from the install above.

### Accuracy against the PyTorch policy

The exported programs were run next to the PyTorch policy on the same inputs and the same starting noise: 50 frames from the dataset the policy was trained on, every action of every chunk. The table gives the typical difference and the 95th percentile of the difference, in the robot's action units, over all six joints. pi0.5 was checked on 50 random inputs, not dataset frames.

| policy  | `executorch_tensorrt` | `onnx_tensorrt` | `aoti_tensorrt` | PyTorch with other noise |
| ------- | --------------------- | --------------- | --------------- | ------------------------ |
| ACT     | 0.01 (0.02)           | 0.01 (0.02)     | 0.01 (0.02)     | the policy is not random |
| SmolVLA | 0.17 (0.56)           | 0.15 (0.52)     | 0.17 (0.56)     | 3.2 (11.9)               |
| pi0.5   | 0.31 (1.17)           | 0.33 (1.37)     | 0.31 (1.23)     | 11.4 (44.8)              |
| GR00T   | 0.28 (0.91)           | 0.28 (0.89)     | 0.28 (0.91)     | 7.8 (22.6)               |

SmolVLA, pi0.5 and GR00T start each chunk from random noise, so the PyTorch policy itself gives different actions every time. The last column is that difference, from the same frame with other noise. PyTorch differs from itself 20 to 40 times more than the exported programs differ from it. For ACT, each joint moves 0.3 to 0.8 on average from one dataset frame to the next, against 0.01 for the exported programs.

### Chunk latency, median (99th percentile)

| policy  | device    | `executorch_tensorrt` | `onnx_tensorrt`  | `aoti_tensorrt`  | PyTorch          | PyTorch with `torch.compile` |
| ------- | --------- | --------------------- | ---------------- | ---------------- | ---------------- | ---------------------------- |
| ACT     | Thor      | 4.5 ms (4.8)          | 4.2 ms (4.5)     | 4.4 ms (5.0)     | 19.2 ms (20.5)   | 14.8 ms (15.2)               |
| ACT     | Orin Nano | 27.3 ms (28.6)        | 27.5 ms (28.4)   | not measured     | 65.1 ms (66.7)   | 63.6 ms (64.5)               |
| SmolVLA | Thor      | 30.1 ms (34.8)        | 31.6 ms (35.9)   | 29.9 ms (34.3)   | 164.5 ms (168.2) | 58.1 ms, 1 run               |
| SmolVLA | Orin Nano | 136.3 ms (137.1)      | 132.0 ms (133.2) | not measured     | 773.5 ms (777.5) | 186.9 ms, 1 run              |
| pi0.5   | Thor      | 108.4 ms (109.3)      | 125.2 ms (125.7) | 108.5 ms (109.7) | 235.9 ms (238.8) | 135.6 ms (136.3), 1 run      |
| pi0.5   | Orin Nano | does not fit          | does not fit     | does not fit     | does not fit     | does not fit                 |
| GR00T   | Thor      | 70.1 ms (74.2)        | 78.5 ms (80.6)   | 69.5 ms (72.7)   | 216.2 ms (224.7) | 212.0 ms (219.2), 1 run      |
| GR00T   | Orin Nano | does not fit          | does not fit     | does not fit     | does not fit     | does not fit                 |

`torch.compile` runs as `lerobot-rollout --use_torch_compile` does: ACT and SmolVLA with `--torch_compile_mode=max-autotune`, GR00T in the default mode. SmolVLA and pi0.5 also compile through their own `compile_model` option, in `max-autotune` mode by default, and pi0.5 uses only that one. pi0.5 was faster with `--policy.compile_mode=default`: 130.0 ms. The first chunk waits for the compile: about 4.3 minutes for ACT, 6.1 for SmolVLA, 7.7 for pi0.5 and 2.1 for GR00T on the Thor, and 8.6 for ACT and 12.2 for SmolVLA on the Orin Nano.

`aoti_tensorrt` builds its TensorRT engine the same way as `executorch_tensorrt`, so a chunk takes about as long, but it loads slower and holds more GPU memory: on the Thor, loading takes 5.8 s for ACT, 13.8 s for SmolVLA, 50.3 s for pi0.5 and 52.6 s for GR00T, against 0.2 s, 4.2 s, 8.9 s and 7.0 s for `executorch_tensorrt`. It was not measured on the Orin Nano.

On the Orin Nano, GR00T's PyTorch policy runs out of memory while it moves its float32 weights, 11.7 GiB, to the GPU, which shares the board's 7.3 GiB. `torch.compile` starts from the same policy.

### Memory added: process, GPU

| policy  | device    | `executorch_tensorrt` | `onnx_tensorrt` | `aoti_tensorrt` | PyTorch         |
| ------- | --------- | --------------------- | --------------- | --------------- | --------------- |
| ACT     | Thor      | 0.5 GB, 0.4 GB        | 0.5 GB, 0.5 GB  | 0.9 GB, 1.8 GB  | 1.4 GB, 1.4 GB  |
| ACT     | Orin Nano | 0.6 GB, 0.2 GB        | 0.7 GB, 0.2 GB  | not measured    | 1.2 GB, 0.7 GB  |
| SmolVLA | Thor      | 0.9 GB, 1.3 GB        | 0.9 GB, 1.6 GB  | 0.9 GB, 3.3 GB  | 2.4 GB, 3.0 GB  |
| SmolVLA | Orin Nano | 1.5 GB, 0.6 GB        | 1.5 GB, 0.5 GB  | not measured    | 2.5 GB, 1.4 GB  |
| pi0.5   | Thor      | 1.3 GB, 6.9 GB        | 1.3 GB, 7.8 GB  | 1.3 GB, 12.5 GB | 1.7 GB, 10.4 GB |
| GR00T   | Thor      | 1.2 GB, 5.9 GB        | 1.2 GB, 6.7 GB  | 1.2 GB, 12.9 GB | 2.0 GB, 14.3 GB |
| GR00T   | Orin Nano | does not fit          | does not fit    | does not fit    | does not fit    |

The Thor and the Orin Nano share one memory between the CPU and the GPU, so the sum of the two numbers is an upper bound. Both count only what loading and running the policy adds. Each process also needs about 0.6 GB for PyTorch and the CUDA context, measured on the Thor, which is not counted here.
