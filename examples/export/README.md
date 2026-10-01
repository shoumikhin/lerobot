# Compiled LeRobot policies on an NVIDIA Jetson

Run LeRobot policies as TensorRT engines on Jetson devices, with no PyTorch loaded on the robot.

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
- [GR00T](../../docs/source/groot.mdx)

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

To compare the two backends, export with the other script and run its folder.
