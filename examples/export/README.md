# Compiled LeRobot policies on an NVIDIA Jetson

Run LeRobot policies as compiled programs on Jetson devices, without the policy's PyTorch code on the robot.

## Hardware

- [SO-101](../../docs/source/so101.mdx) arm with a scene camera and a wrist camera
- [Jetson AGX Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/), to train, export and run
- [Jetson Orin Nano Super](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/), to export and run a pretrained policy

## Backends

- `executorch_tensorrt`: Torch-TensorRT into an ExecuTorch `.pte` file
- `executorch_cuda`: ExecuTorch's CUDA backend, without TensorRT, into a `.pte` file and its `.ptd` weights
- `onnx_tensorrt`: ONNX, then TensorRT, into an `.engine` file
- `torch_tensorrt`: Torch-TensorRT into an AOTInductor `.pt2` package, which needs PyTorch to run

## Policies

- [ACT](../../docs/source/act.mdx)
- [SmolVLA](../../docs/source/smolvla.mdx)
- [pi0.5](../../docs/source/pi05.mdx)
- [GR00T N1.7](../../docs/source/groot.mdx)

## Install

On each Jetson, with [uv](https://docs.astral.sh/uv/) installed. It is one of the environment managers in LeRobot's [installation guide](../../docs/source/installation.mdx), and it creates the environment without the `python3-venv` package, which JetPack may not include:

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

## Export

Export on each device the policy will run on, because the compiled program only runs on the same GPU model that built it:

```bash
python examples/export/act_executorch_tensorrt.py \
    --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model \
    --output_dir=outputs/export/act_executorch_tensorrt
```

| policy  | `executorch_tensorrt`                                            | `executorch_cuda`                                    | `onnx_tensorrt`                                      | `torch_tensorrt`                                       |
| ------- | ---------------------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------ |
| ACT     | [act_executorch_tensorrt.py](act_executorch_tensorrt.py)         | [act_executorch_cuda.py](act_executorch_cuda.py)     | [act_onnx_tensorrt.py](act_onnx_tensorrt.py)         | [act_torch_tensorrt.py](act_torch_tensorrt.py)         |
| SmolVLA | [smolvla_executorch_tensorrt.py](smolvla_executorch_tensorrt.py) | not supported yet                                    | [smolvla_onnx_tensorrt.py](smolvla_onnx_tensorrt.py) | [smolvla_torch_tensorrt.py](smolvla_torch_tensorrt.py) |
| pi0.5   | [pi05_executorch_tensorrt.py](pi05_executorch_tensorrt.py)       | [pi05_executorch_cuda.py](pi05_executorch_cuda.py)   | [pi05_onnx_tensorrt.py](pi05_onnx_tensorrt.py)       | [pi05_torch_tensorrt.py](pi05_torch_tensorrt.py)       |
| GR00T   | [groot_executorch_tensorrt.py](groot_executorch_tensorrt.py)     | [groot_executorch_cuda.py](groot_executorch_cuda.py) | [groot_onnx_tensorrt.py](groot_onnx_tensorrt.py)     | [groot_torch_tensorrt.py](groot_torch_tensorrt.py)     |

SmolVLA and GR00T also need the dataset they were trained on, for the cameras and the task:

```bash
python examples/export/smolvla_executorch_tensorrt.py \
    --policy.path=outputs/train/smolvla_so101/checkpoints/last/pretrained_model \
    --dataset.repo_id=<user>/so101_dataset \
    --output_dir=outputs/export/smolvla_executorch_tensorrt
```

Add `--dataset.root=<folder>` if the dataset is only on your disk.

pi0.5 needs the task it was trained on:

```bash
python examples/export/pi05_executorch_tensorrt.py \
    --policy.path=outputs/train/pi05_so101/checkpoints/last/pretrained_model \
    --task="Pick up the block and place it in the cup" \
    --output_dir=outputs/export/pi05_executorch_tensorrt
```

The `executorch_cuda` scripts compile CUDA kernels with `nvcc`, so add it to your `PATH` first: `export PATH=/usr/local/cuda/bin:$PATH`. GR00T's needs an ExecuTorch with its CUDA backend fixes for attention masks and view weights.

GR00T runs only on the Thor, and pi0.5 runs on the Orin Nano only with INT8 weights (below). Their bfloat16 weights do not fit in the Orin Nano's memory.

### Export once, build on each device

With `--export_only`, the `executorch_tensorrt` and `onnx_tensorrt` scripts write the folder without the engine. Copy it to each device and build the engine there:

```bash
python examples/export/act_onnx_tensorrt.py \
    --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model \
    --output_dir=outputs/export/act_onnx_tensorrt \
    --export_only
python examples/export/build_engine.py outputs/export/act_onnx_tensorrt
```

Lower `--workspace_gib` and `--optimization_level` make the build use less memory.

With `--step_engine`, `pi05_onnx_tensorrt.py` exports pi0.5 as three smaller engines instead of one, and the rollout runs the denoising step engine once per step. On the Thor it runs about as fast as one engine.

With `--int8_weights`, `build_engine.py` rewrites the ONNX files of an `onnx_tensorrt` folder to store their large weights in INT8, and the layers still compute in bfloat16. This halves the memory of the weights, so pi0.5 exported with `--step_engine` fits on the Orin Nano. It runs slower and drifts a little more from the PyTorch policy: on the Thor, 170.3 ms against 117.0 ms in bfloat16.

## Run

Run the exported folder like any LeRobot policy, with the same `--robot.*` options you recorded with:

```bash
lerobot-rollout \
    --policy.path=outputs/export/act_executorch_tensorrt \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, scene: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }" \
    --duration=30
```

For SmolVLA, add the task, for example `--task="Pick up the brick and put it in the bin"`. pi0.5 and GR00T run only the task they were exported for, so pass that same `--task`.

Before the robot moves, the rollout checks the program against a test case saved at export time, and stops if its actions differ from the PyTorch policy's.

## Benchmarks

Measured with JetPack 7.2.1 and the packages above. Each latency and memory cell is the median of 3 runs, each in a fresh process, except one run for `torch.compile` with SmolVLA, pi0.5 and GR00T.

### Accuracy

Typical difference from the PyTorch policy, in robot action units, with the 95th percentile in parentheses, over 50 dataset frames (random inputs for pi0.5). The last column is how much the PyTorch policy differs from itself with other starting noise.

| policy              | `executorch_tensorrt` | `executorch_cuda` | `onnx_tensorrt` | `torch_tensorrt`  | PyTorch, other noise |
| ------------------- | --------------------- | ----------------- | --------------- | ----------------- | -------------------- |
| ACT                 | 0.01 (0.02)           | 0.01 (0.02)       | 0.01 (0.02)     | 0.01 (0.02)       | no noise             |
| SmolVLA             | 0.17 (0.56)           | not supported yet | 0.15 (0.52)     | 0.17 (0.56)       | 3.2 (11.9)           |
| pi0.5               | 0.31 (1.17)           | 0.26 (0.95)       | 0.33 (1.37)     | 0.31 (1.23)       | 11.4 (44.8)          |
| pi0.5, INT8 weights | not supported yet     | not supported yet | 0.51 (2.00)     | not supported yet | 11.4 (44.8)          |
| GR00T               | 0.28 (0.91)           | 0.49 (1.39)       | 0.28 (0.89)     | 0.28 (0.91)       | 7.8 (22.6)           |

### Latency per action chunk, median (99th percentile)

| policy              | device    | `executorch_tensorrt` | `executorch_cuda` | `onnx_tensorrt`  | `torch_tensorrt`  | PyTorch           | `torch.compile`   |
| ------------------- | --------- | --------------------- | ----------------- | ---------------- | ----------------- | ----------------- | ----------------- |
| ACT                 | Thor      | 4.5 ms (4.8)          | 38.7 ms (39.0)    | 4.2 ms (4.5)     | 4.4 ms (5.0)      | 19.2 ms (20.5)    | 14.8 ms (15.2)    |
| ACT                 | Orin Nano | 27.3 ms (28.6)        | 152.3 ms (153.8)  | 27.5 ms (28.4)   | 27.7 ms (28.5)    | 65.1 ms (66.7)    | 63.6 ms (64.5)    |
| SmolVLA             | Thor      | 30.1 ms (34.8)        | not supported yet | 31.6 ms (35.9)   | 29.9 ms (34.3)    | 164.5 ms (168.2)  | 58.1 ms           |
| SmolVLA             | Orin Nano | 136.3 ms (137.1)      | not supported yet | 132.0 ms (133.2) | 137.9 ms (140.8)  | 773.5 ms (777.5)  | 186.9 ms          |
| pi0.5               | Thor      | 108.4 ms (109.3)      | 122.6 ms (123.2)  | 125.2 ms (125.7) | 108.5 ms (109.7)  | 235.9 ms (238.8)  | 135.6 ms (136.3)  |
| pi0.5               | Orin Nano | does not fit          | does not fit      | does not fit     | does not fit      | does not fit      | does not fit      |
| pi0.5, INT8 weights | Thor      | not supported yet     | not supported yet | 170.3 ms         | not supported yet | not supported yet | not supported yet |
| pi0.5, INT8 weights | Orin Nano | not supported yet     | not supported yet | 861.5 ms (864.8) | not supported yet | not supported yet | not supported yet |
| GR00T               | Thor      | 70.1 ms (74.2)        | 78.1 ms (82.3)    | 78.5 ms (80.6)   | 69.5 ms (72.7)    | 216.2 ms (224.7)  | 212.0 ms (219.2)  |

### Memory added: process, GPU

| policy              | device    | `executorch_tensorrt` | `executorch_cuda` | `onnx_tensorrt`  | `torch_tensorrt`       | PyTorch           |
| ------------------- | --------- | --------------------- | ----------------- | ---------------- | ---------------------- | ----------------- |
| ACT                 | Thor      | 0.5 GiB, 0.4 GiB      | 0.2 GiB, 0.4 GiB  | 0.5 GiB, 0.5 GiB | 0.9 GiB, 1.8 GiB       | 1.4 GiB, 1.4 GiB  |
| ACT                 | Orin Nano | 0.6 GiB, 0.2 GiB      | 0.4 GiB, 0.1 GiB  | 0.7 GiB, 0.2 GiB | 1.2 GiB, 0.5 GiB       | 1.2 GiB, 0.7 GiB  |
| SmolVLA             | Thor      | 0.9 GiB, 1.3 GiB      | not supported yet | 0.9 GiB, 1.6 GiB | 0.9 GiB, 3.3 GiB       | 2.4 GiB, 3.0 GiB  |
| SmolVLA             | Orin Nano | 1.5 GiB, 0.6 GiB      | not supported yet | 1.5 GiB, 0.5 GiB | 0.7 GiB, under 0.3 GiB | 2.5 GiB, 1.4 GiB  |
| pi0.5               | Thor      | 1.3 GiB, 6.9 GiB      | 0.8 GiB, 6.3 GiB  | 1.3 GiB, 7.8 GiB | 1.3 GiB, 12.5 GiB      | 1.7 GiB, 10.4 GiB |
| pi0.5               | Orin Nano | does not fit          | does not fit      | does not fit     | does not fit           | does not fit      |
| pi0.5, INT8 weights | Orin Nano | not supported yet     | not supported yet | 3.8 GiB, 2.8 GiB | not supported yet      | not supported yet |
| GR00T               | Thor      | 1.2 GiB, 5.9 GiB      | 0.8 GiB, 5.7 GiB  | 1.2 GiB, 6.7 GiB | 1.2 GiB, 12.9 GiB      | 2.0 GiB, 14.3 GiB |

The Jetson CPU and GPU share one memory, so the two numbers can overlap.
