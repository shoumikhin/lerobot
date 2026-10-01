# Compiled LeRobot policies on an NVIDIA Jetson

Run LeRobot policies as TensorRT engines on Jetson devices, without the policy's PyTorch code on the robot.

## Hardware

- SO-101 arm with a scene camera and a wrist camera
- Jetson AGX Thor, to train, export and run
- Jetson Orin Nano, to export and run a pretrained policy

## Backends

- `onnx_tensorrt`: ONNX using `torch.export`, then TensorRT, into an `.engine` file
- `executorch_tensorrt`: `torch.export`, then Torch-TensorRT, into an ExecuTorch `.pte` file

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

| policy  | `executorch_tensorrt`                                            | `onnx_tensorrt`                                      |
| ------- | ---------------------------------------------------------------- | ---------------------------------------------------- |
| ACT     | [act_executorch_tensorrt.py](act_executorch_tensorrt.py)         | [act_onnx_tensorrt.py](act_onnx_tensorrt.py)         |
| SmolVLA | [smolvla_executorch_tensorrt.py](smolvla_executorch_tensorrt.py) | [smolvla_onnx_tensorrt.py](smolvla_onnx_tensorrt.py) |
| pi0.5   | [pi05_executorch_tensorrt.py](pi05_executorch_tensorrt.py)       | [pi05_onnx_tensorrt.py](pi05_onnx_tensorrt.py)       |
| GR00T   | [groot_executorch_tensorrt.py](groot_executorch_tensorrt.py)     | [groot_onnx_tensorrt.py](groot_onnx_tensorrt.py)     |

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

pi0.5 does not fit in the Orin Nano's 8 GB: its export and its TensorRT engine build both run out of memory there. Export and run it on the Thor.

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

To compare the two backends, export with the other script and run its folder.

## Benchmarks

Time for one action chunk, and the memory a process adds to run it. Each cell is the median of 3 runs unless marked 1 run, each in a fresh process, through the rollout's own inference engines. Each run times at least 100 chunks after 20 warmup chunks. The policies were trained on an SO-101 arm with two cameras. "WIP" means not measured yet.

Setup: JetPack 7.2.1, nightly PyTorch 2.15, TensorRT 11.3, and Torch-TensorRT and ExecuTorch built from source with two fixes: [pytorch/executorch#23301](https://github.com/pytorch/executorch/pull/23301) and [pytorch/TensorRT#4767](https://github.com/pytorch/TensorRT/pull/4767).

### Chunk latency, median (99th percentile)

| policy  | device    | `executorch_tensorrt` | `onnx_tensorrt`  | PyTorch          | PyTorch with `torch.compile` |
| ------- | --------- | --------------------- | ---------------- | ---------------- | ---------------------------- |
| ACT     | Thor      | 4.5 ms (4.8)          | 4.2 ms (4.5)     | 19.2 ms (20.5)   | 14.8 ms (15.2)               |
| ACT     | Orin Nano | 27.3 ms (28.6)        | 27.5 ms (28.4)   | 65.1 ms (66.7)   | 63.6 ms (64.5)               |
| SmolVLA | Thor      | 30.1 ms (34.8)        | 31.6 ms (35.9)   | 164.5 ms (168.2) | 58.1 ms, 1 run               |
| SmolVLA | Orin Nano | 136.3 ms (137.1)      | 132.0 ms (133.2) | 773.5 ms (777.5) | 186.9 ms, 1 run              |
| pi0.5   | Thor      | 108.4 ms (109.3)      | 125.2 ms (125.7) | 235.9 ms (238.8) | 135.6 ms (136.3), 1 run      |
| pi0.5   | Orin Nano | does not fit          | does not fit     | does not fit     | does not fit                 |
| GR00T   | Thor      | 70.1 ms (74.2)        | 78.5 ms (80.6)   | 216.2 ms (224.7) | 212.0 ms (219.2), 1 run      |
| GR00T   | Orin Nano | WIP                   | WIP              | WIP              | WIP                          |

`torch.compile` runs as `lerobot-rollout --use_torch_compile` does: ACT and SmolVLA with `--torch_compile_mode=max-autotune`, GR00T in the default mode. SmolVLA and pi0.5 also compile through their own `compile_model` option, in `max-autotune` mode by default, and pi0.5 uses only that one. pi0.5 was faster with `--policy.compile_mode=default`: 130.0 ms. The first chunk waits for the compile: about 4.3 minutes for ACT, 6.1 for SmolVLA, 7.7 for pi0.5 and 2.1 for GR00T on the Thor, and 8.6 for ACT and 12.2 for SmolVLA on the Orin Nano.

### Memory added: process, GPU

| policy  | device    | `executorch_tensorrt` | `onnx_tensorrt` | PyTorch         |
| ------- | --------- | --------------------- | --------------- | --------------- |
| ACT     | Thor      | 0.5 GB, 0.4 GB        | 0.5 GB, 0.5 GB  | 1.4 GB, 1.4 GB  |
| ACT     | Orin Nano | 0.6 GB, 0.2 GB        | 0.7 GB, 0.2 GB  | 1.2 GB, 0.7 GB  |
| SmolVLA | Thor      | 0.9 GB, 1.3 GB        | 0.9 GB, 1.6 GB  | 2.4 GB, 3.0 GB  |
| SmolVLA | Orin Nano | 1.5 GB, 0.6 GB        | 1.5 GB, 0.5 GB  | 2.5 GB, 1.4 GB  |
| pi0.5   | Thor      | 1.3 GB, 6.9 GB        | 1.3 GB, 7.8 GB  | 1.7 GB, 10.4 GB |
| GR00T   | Thor      | 1.2 GB, 5.9 GB        | 1.2 GB, 6.7 GB  | 2.0 GB, 14.3 GB |
| GR00T   | Orin Nano | WIP                   | WIP             | WIP             |

The Thor and the Orin Nano share one memory between the CPU and the GPU, so the sum of the two numbers is an upper bound. Both count only what loading and running the policy adds. Each process also needs about 0.6 GB for PyTorch and the CUDA context, measured on the Thor, which is not counted here.
