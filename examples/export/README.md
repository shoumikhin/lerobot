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
