# Compiled LeRobot policies on an NVIDIA Jetson

![An SO-101 arm handing over a candy](https://raw.githubusercontent.com/shoumikhin/lerobot/assets/candy.gif)

_An SO-101 arm hands over a candy, driven by a policy exported with ExecuTorch-TensorRT on a Jetson Orin Nano._

Train a policy as usual, export it into a compiled folder, and run that folder with `lerobot-rollout`, the same command you use for a PyTorch checkpoint. In the results below, a compiled policy runs 2 to 6 times faster than in PyTorch, and ExecuTorch-TensorRT and ONNX-TensorRT need about half of PyTorch's memory or less, so the larger policies also fit on a small board.

Supported policies: ACT, SmolVLA, pi0.5 and GR00T N1.7.

## Hardware

- [SO-101](../../docs/source/so101.mdx) arm with a scene camera and a wrist camera
- [Jetson AGX Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/), to train, export and run
- [Jetson Orin Nano Super](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/), to export and run in 8 GB

## Install

On each Jetson (JetPack 7.2.1), with [uv](https://docs.astral.sh/uv/):

```bash
git clone --branch export https://github.com/shoumikhin/lerobot.git
cd lerobot
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[smolvla,pi,groot,feetech,export]" torch-tensorrt "executorch>=1.6.0.dev20261021" \
    --index-url https://download.pytorch.org/whl/nightly/cu132 --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match --prerelease allow
```

The ExecuTorch-TensorRT route also needs `torch-tensorrt-executorch-runtime` built for ExecuTorch 1.6.0.dev20261021 or newer. Until a matching nightly is published, [build it from source](https://github.com/pytorch/TensorRT/blob/main/py/torch-tensorrt-executorch-runtime/README.md#build). The Torch-TensorRT 2.15 release is meant to make all of this a single `pip install`.

## Export

Export on the kind of board that will run the policy: a compiled folder runs only on the kind of GPU (graphics processor) that built it.

```bash
python examples/export/to_executorch_tensorrt.py \
    --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model \
    --output_dir=outputs/export/act_executorch_tensorrt
```

Each script compiles any supported policy one way. All use [TensorRT](https://developer.nvidia.com/tensorrt), NVIDIA's compiler for its GPUs. If you are not sure, start with ExecuTorch-TensorRT: it is fast and small, and it runs without PyTorch.

| script                      | writes                      | needs PyTorch to run |
| --------------------------- | --------------------------- | -------------------- |
| `to_executorch_tensorrt.py` | ExecuTorch `.pte` programs  | no                   |
| `to_onnx_tensorrt.py`       | TensorRT `.engine` files    | no                   |
| `to_torch_tensorrt.py`      | AOTInductor `.pt2` packages | yes                  |

Some policies need more arguments:

- **SmolVLA and GR00T** take their camera names and task from the dataset they were trained on: add `--dataset.repo_id=<user>/so101_dataset`, and `--task="<one of its tasks>"` to pick a task other than its first.
- **pi0.5** takes a task it was trained on: add `--task="pick up the block and place it in the cup"`.

A few checkpoints cannot be exported: ACT with temporal ensembling, and pi0.5 with visual or proprioceptive memory. A GR00T checkpoint that decodes its actions with its embodiment statistics exports only with `to_onnx_tensorrt.py`.

To run an ONNX-TensorRT export on another board, copy the folder there and rebuild its engines with `python examples/export/build_engine.py <folder>`.

## Run

Use the same calibrated robot and camera settings as when you recorded the training data:

```bash
lerobot-rollout \
    --policy.path=outputs/export/act_executorch_tensorrt \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: '/dev/video-wrist', width: 640, height: 480, fps: 30}, scene: {type: opencv, index_or_path: '/dev/video-scene', width: 640, height: 480, fps: 30} }" \
    --duration=30
```

pi0.5 and GR00T run any task: pass it with `--task`, or change it while the robot runs. Their folders keep the policy's own tokenizer and its whole word table, and the rollout reads only the rows of the task's words. A GR00T folder takes tasks up to 32 tokens (about 25 words); export with `--max_task_tokens` for longer ones, at a small cost in time. SmolVLA runs only the task it was exported for, so leave out `--task` or pass the same one. Without `--task`, a folder runs the task it was exported with.

pi0.5 also runs with any of the cameras it was trained with: it fills in a camera the robot does not send, as the PyTorch policy does. If your camera names differ from the policy's, add `--rename_map`, for example `--rename_map='{"observation.images.scene": "observation.images.base_0_rgb"}'`.

Before the robot moves, the rollout checks that the compiled policy reproduces the actions PyTorch computed at export time, and stops if it does not.

### Without a robot

[replay_robot](replay_robot) plays back recorded episodes and accepts every action, so you can try a policy at your desk:

```bash
uv pip install -e examples/export/replay_robot
python examples/export/replay_robot/write_frames.py \
    --dataset.repo_id=<user>/so101_dataset --episodes 0 1 2 3 --output_dir=outputs/replay_frames
lerobot-rollout \
    --policy.path=outputs/export/act_executorch_tensorrt \
    --robot.type=replay --robot.frames_dir=outputs/replay_frames \
    --duration=60 --return_to_initial_position=false --play_sounds=false
```

## Results

The tables compare each policy, on each board, compiled each way and in plain PyTorch. For time and accuracy, the PyTorch column runs the same parts the export compiles, in PyTorch, through the same engine as the compiled folders. For memory, it runs the policy's own PyTorch checkpoint with `lerobot-rollout`.

There is no `torch.compile` column: it was measured only in its default mode, without CUDA graphs, so it is left out until it is measured with them.

- Both boards ran at maximum power, with the GPU clock locked.
- **Bold** marks the best time or memory in a row.
- OOM (out of memory) means the policy did not fit on the 8 GB Orin Nano.

Measured with PyTorch 2.15.0.dev20261005, TensorRT 11.3.0.99, and this fork's scripts at commits up to [f6951b7c9](https://github.com/shoumikhin/lerobot/commit/f6951b7c9).

### Time per chunk

A policy returns a chunk of actions at a time (ACT 100, SmolVLA and pi0.5 50, GR00T 16). `lerobot-rollout` plays the whole chunk, then runs the policy again while the arm holds still, so this time is the pause between chunks. Milliseconds, lower is better.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | PyTorch |
| ------- | ------------------: | ------------: | -------------: | ------: |
| ACT     |                   4 |         **3** |              4 |      10 |
| SmolVLA |              **30** |        **30** |             31 |     168 |
| pi0.5   |                 118 |       **113** |            114 |     240 |
| GR00T   |              **92** |            93 |             94 |     215 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | PyTorch |
| ------- | ------------------: | ------------: | -------------: | ------: |
| ACT     |              **26** |        **26** |             27 |      53 |
| SmolVLA |             **124** |       **124** |            138 |     757 |
| pi0.5   |                 672 |       **655** |            OOM |     OOM |
| GR00T   |             **268** |           276 |            276 |     OOM |

Each value is the median of three runs of 100 chunks. Differences of a few milliseconds are within run-to-run noise. The ExecuTorch-TensorRT and ONNX-TensorRT times were measured with CUDA graphs (recorded GPU work that replays with less overhead), except GR00T on ONNX-TensorRT. The Torch-TensorRT times were measured without them. The current scripts turn CUDA graphs on for every route.

### Memory

Extra memory the board needs while the policy runs: the drop in available memory plus any growth in swap, in MiB (the main processor, or CPU, and the GPU share it on a Jetson). Lower is better. Measured during a `lerobot-rollout` run with the replay robot.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT |      PyTorch |
| ------- | ------------------: | ------------: | -------------: | -----------: |
| ACT     |             **538** |           552 |           1163 |         2047 |
| SmolVLA |            **1225** |          1316 |           2681 |         3782 |
| pi0.5   |                5996 |      **5922** |           7059 |        11507 |
| GR00T   |            **5310** |          5412 |           6007 | not measured |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | PyTorch |
| ------- | ------------------: | ------------: | -------------: | ------: |
| ACT     |             **206** |           224 |            846 |    1126 |
| SmolVLA |             **346** |           354 |           1716 |    2300 |
| pi0.5   |                4974 |      **4871** |            OOM |     OOM |
| GR00T   |            **4421** |          4422 |           4980 |     OOM |

On the Orin Nano these values can be up to about 0.9 GiB too low: the GPU driver reuses memory it already holds, and the board does not count that memory as newly used. On Thor the values are exact.

### Accuracy

Mean / p95 of each frame's largest action error, in degrees (percent for the gripper), against PyTorch, over 50 frames (p95: 95% of the frames have a smaller error). The frames are spread evenly over an SO-101 dataset, and every run uses the same random noise. Lower is better. Every compiled folder passed the export's own check.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT |
| ------- | ------------------: | ------------: | -------------: |
| ACT     |         0.04 / 0.11 |   0.03 / 0.10 |    0.04 / 0.11 |
| SmolVLA |         0.46 / 1.55 |   0.50 / 1.49 |    0.46 / 1.55 |
| pi0.5   |         0.57 / 0.81 |   0.57 / 0.83 |    0.57 / 0.81 |
| GR00T   |         0.58 / 1.31 |   0.60 / 1.36 |    0.58 / 1.31 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT |
| ------- | ------------------: | ------------: | -------------: |
| ACT     |         0.04 / 0.13 |   0.04 / 0.12 |    0.04 / 0.13 |
| SmolVLA |         0.39 / 1.36 |   0.40 / 1.22 |    0.39 / 1.36 |
| pi0.5   |         0.57 / 0.85 |   0.56 / 0.77 |            OOM |
| GR00T   |         0.61 / 1.51 |   0.63 / 1.59 |    0.61 / 1.51 |

ExecuTorch-TensorRT and Torch-TensorRT build the same TensorRT engines, so their accuracy is the same.

### Reproduce

Run each step on the board you measure. For SmolVLA, pi0.5 and GR00T, give `benchmark.py pytorch` the same extra arguments as the export script.

```bash
# 50 frames from the dataset
python examples/export/benchmark.py frames --dataset.repo_id=<user>/so101_dataset \
    --output=outputs/benchmark/frames.npz
# the reference, in PyTorch
python examples/export/benchmark.py pytorch --policy.path=<checkpoint> \
    --frames=outputs/benchmark/frames.npz --output=outputs/benchmark/act_pytorch.npz
# each compiled folder
python examples/export/benchmark.py outputs/export/act_executorch_tensorrt \
    --frames=outputs/benchmark/frames.npz --output=outputs/benchmark/act_executorch_tensorrt.npz
# time per chunk and accuracy against PyTorch
python examples/export/benchmark.py report outputs/benchmark/act_pytorch.npz outputs/benchmark/act_*.npz
# memory, with the replay robot from above; pass --policy.path=<checkpoint> for PyTorch
python examples/export/benchmark_memory.py -- lerobot-rollout --policy.path=outputs/export/act_executorch_tensorrt \
    --robot.type=replay --robot.frames_dir=outputs/replay_frames \
    --duration=60 --return_to_initial_position=false --play_sounds=false
```

Torch-TensorRT, and PyTorch on Thor, keep freeing memory for a minute or two after loading, so their memory was measured with `--duration=180`.

## Good to know

- **Orin Nano memory.** Compiling SmolVLA, pi0.5 or GR00T on the 8 GB board may need swap space.
- **Changing the task.** A pi0.5 or GR00T folder tokenizes a new task in a short-lived process, which takes about 2 s on the Orin Nano while the arm holds still. The task the folder was exported with needs none.
- **The checks show that the compiled policy matches PyTorch.** They do not show that the robot is safe or that it does the task.
