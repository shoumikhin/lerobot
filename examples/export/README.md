# Compiled LeRobot policies on an NVIDIA Jetson

![An SO-101 arm handing over a candy](https://raw.githubusercontent.com/shoumikhin/lerobot/assets/candy.gif)

_An SO-101 arm hands over a candy, driven by a policy exported with ExecuTorch-TensorRT on a Jetson Orin Nano._

Train a policy as usual, export it into a compiled folder, and run that folder with `lerobot-rollout`, the same command you use for a PyTorch checkpoint. In the results below, a compiled policy runs 1.5 to 7 times faster than in PyTorch, and ExecuTorch-TensorRT and ONNX-TensorRT need a quarter to three quarters of PyTorch's memory, so pi0.5 and GR00T also run on an 8 GB Orin Nano, where they do not fit in PyTorch.

Supported policies: ACT, SmolVLA, pi0, pi0.5 and GR00T N1.7.

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

- **SmolVLA and GR00T** take their camera names and a task from the dataset they were trained on: add `--dataset.repo_id=<user>/so101_dataset`, and `--task="<one of its tasks>"` to pick a task other than its first. A GR00T folder then runs any task you give it; a SmolVLA folder runs only that task.
- **pi0** takes the task it will run: add `--task="pick up the block"`. Like a SmolVLA folder, the folder runs only that task. For a float32 checkpoint such as `lerobot/pi0_base` that program is about 10 GB, so export it on a board with more memory than the Orin Nano.
- **pi0.5** takes one task it was trained on, for the export's own check: add `--task="pick up the block and place it in the cup"`. The folder then runs any task you give it.

A few checkpoints cannot be exported: ACT with temporal ensembling, and pi0.5 with visual or proprioceptive memory.

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

pi0.5 and GR00T run any task: pass it with `--task`, or change it while the robot runs. Their folders keep the policy's own tokenizer and its whole word table, and the rollout reads only the rows of the task's words. A GR00T folder takes tasks up to 32 tokens (about 25 words); export with `--max_task_tokens` for longer ones, at a small cost in time. A pi0.5 folder takes tasks as long as the policy's prompt allows; export it with a smaller `--max_task_tokens` to make every chunk faster. SmolVLA and pi0 run only the task they were exported for, so leave out `--task` or pass the same one. Without `--task`, a folder runs the task it was exported with.

pi0.5 also runs with any of the cameras it was trained with: it fills in a camera the robot does not send, as the PyTorch policy does. If your robot never has some of them, export with `--cameras` and the slots it can fill: the program then leaves the others out, which saves time. If your camera names differ from the policy's, add `--rename_map`, for example `--rename_map='{"observation.images.scene": "observation.images.base_0_rgb"}'`.

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

The tables compare each policy, on each board, compiled each way, with `torch.compile`, and in plain PyTorch. The `torch.compile` and PyTorch columns run the same parts the export compiles, through the same engine as the compiled folders, so only what runs the parts differs. Every policy was exported with the commands above and their defaults: pi0.5 and GR00T take any task at run time, pi0.5 with its full prompt length, and pi0.5 takes every camera it was trained with.

- Both boards ran at maximum power, with the GPU clock locked.
- **Bold** marks the best time or memory in a row.
- OOM (out of memory) means the policy did not fit on the 8 GB Orin Nano.

Measured at [82a5bfc96](https://github.com/shoumikhin/lerobot/commit/82a5bfc96ad979826f5e4a16439a8fb39d47448d), with PyTorch 2.15.0.dev20261005, TensorRT 11.3.0.99, ExecuTorch 1.6.0.dev20261021 and Torch-TensorRT 2.16.0.dev20261021.

### Time per chunk

A policy returns a chunk of actions at a time (ACT 100, SmolVLA and pi0.5 50, GR00T 16). `lerobot-rollout` plays the whole chunk, then runs the policy again while the arm holds still, so this time is the pause between chunks. Milliseconds, lower is better.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |                 3.6 |       **3.4** |            3.5 |             6.2 |     9.9 |
| SmolVLA |                  29 |            30 |         **28** |              40 |     167 |
| pi0.5   |                  81 |            88 |         **79** |              91 |     220 |
| GR00T   |                  62 |        **60** |             61 |              66 |      91 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |                  26 |        **25** |             26 |              42 |      48 |
| SmolVLA |             **116** |           121 |        **116** |             162 |     761 |
| pi0.5   |                 468 |       **450** |            OOM |             OOM |     OOM |
| GR00T   |                 210 |       **206** |            209 |             OOM |     OOM |

Each value is the median of three runs of 100 chunks, after 20 warm-up chunks. Differences of a few milliseconds are within run-to-run noise. Every compiled policy, and `torch.compile`, replays its programs as CUDA graphs (recorded GPU work that replays with less overhead).

### Memory

Memory the policy's process holds while it runs, in MiB: its own memory, what it swapped out, and its GPU memory (the main processor, or CPU, and the GPU share one memory on a Jetson). Files it only reads, such as the folder's programs, are not counted, since the board can drop them from memory. Lower is better.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |             **438** |           444 |           1005 |            2065 |    1796 |
| SmolVLA |            **1105** |          1222 |           2582 |       4585-5351 |    3376 |
| pi0.5   |                5609 |      **5501** |           7322 |            7911 |    7425 |
| GR00T   |                5316 |      **5263** |           6525 |            7533 |    7050 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` |   PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | --------: |
| ACT     |             **404** |           416 |           1091 |            1703 |      1505 |
| SmolVLA |            **1061** |          1177 |           2488 |            4064 | 2980-3396 |
| pi0.5   |                5536 |      **5431** |            OOM |             OOM |       OOM |
| GR00T   |                5297 |      **5274** |           6470 |             OOM |       OOM |

Each value is the median of three runs, over the last 30 s of 120 s of chunks after the warm-up chunks; a range means the three runs differed by more than 5%. `torch.compile` was measured with its compile cache already filled, as on every start after the first.

### Accuracy

How far each column's actions are from the original LeRobot policy run in full float32 (no bfloat16, no TensorFloat-32), on 50 frames spread evenly over an SO-101 dataset, with the same random noise. For each frame, take the largest difference over its actions and joints; each cell is the mean / 95th percentile of that over the 50 frames, in degrees (gripper in %). Lower is better. Every compiled folder passed the export's own check.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` |     PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ----------: |
| ACT     |         0.03 / 0.08 |   0.04 / 0.10 |    0.03 / 0.08 |     0.04 / 0.09 | 0.04 / 0.09 |
| SmolVLA |         0.47 / 1.58 |   0.37 / 1.72 |    0.45 / 1.57 |     0.24 / 0.91 | 0.43 / 1.40 |
| pi0.5   |         0.46 / 0.63 |   0.47 / 0.74 |    0.46 / 0.63 |     0.40 / 0.59 | 0.53 / 0.95 |
| GR00T   |         0.68 / 2.09 |   0.66 / 1.82 |    0.68 / 2.09 |     0.67 / 2.20 | 0.68 / 2.05 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` |     PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ----------: |
| ACT     |         0.03 / 0.10 |   0.04 / 0.12 |    0.03 / 0.10 |     0.03 / 0.08 | 0.03 / 0.07 |
| SmolVLA |         0.36 / 1.44 |   0.36 / 1.35 |    0.36 / 1.44 |     0.24 / 0.78 | 0.40 / 1.49 |
| pi0.5   |         0.49 / 0.83 |   0.46 / 0.70 |            OOM |             OOM |         OOM |
| GR00T   |         0.60 / 1.51 |   0.71 / 1.98 |    0.60 / 1.51 |             OOM |         OOM |

ExecuTorch-TensorRT and Torch-TensorRT build the same TensorRT engines, so their accuracy is the same up to the kernels TensorRT picks in each build. `torch.compile` changes its actions a little from one compile to the next: SmolVLA's 95th percentile on Thor ranged from 0.9 to 1.4 degrees over four compiles. Each cell comes from one build. The reference ran on Thor: in float32, pi0.5 and GR00T do not fit on the Orin Nano.

### pi0, on an H100

pi0 was exported with `to_executorch_tensorrt.py` and measured with `benchmark.py` on an NVIDIA H100, not on a Jetson, with the checkpoint `lerobot/pi0_base`. No dataset has its cameras, so the frames are 50 random ones shaped like its inputs, made with `frame_from_policy` from `common.py`. Times are the median of three runs. Error is the mean / 95th percentile of each frame's largest difference from the original policy in float32. `lerobot/pi0_base` saves no normalization statistics, so its actions, and the errors, are in the policy's normalized units.

|                     | ExecuTorch-TensorRT |       PyTorch |
| ------------------- | ------------------: | ------------: |
| Time per chunk (ms) |            **42.9** |         176.4 |
| Error               |       0.002 / 0.005 | 0.003 / 0.007 |

Measured with PyTorch 2.16.0.dev20261026, TensorRT 11.4.0.106, and ExecuTorch 1.6.0 and Torch-TensorRT 2.16.0.dev0 built from source.

### Reproduce

Run each step on the board you measure, except the reference: it is the same on every board, so run it once where the policy fits in float32. For SmolVLA, pi0, pi0.5 and GR00T, give `benchmark.py pytorch` and `torch.compile` the same extra arguments as the export script.

```bash
# 50 frames from the dataset
python examples/export/benchmark.py frames --dataset.repo_id=<user>/so101_dataset \
    --output=outputs/benchmark/frames.npz
# the reference: the original policy in float32, given an export folder for its inputs, task and noise
python examples/export/benchmark.py reference outputs/export/act_executorch_tensorrt --policy.path=<checkpoint> \
    --frames=outputs/benchmark/frames.npz --output=outputs/benchmark/reference_act.npz
# PyTorch and torch.compile
python examples/export/benchmark.py pytorch --policy.path=<checkpoint> \
    --frames=outputs/benchmark/frames.npz --output=outputs/benchmark/act_pytorch.npz
python examples/export/benchmark.py torch.compile --policy.path=<checkpoint> \
    --frames=outputs/benchmark/frames.npz --output=outputs/benchmark/act_torch_compile.npz
# each compiled folder
python examples/export/benchmark.py outputs/export/act_executorch_tensorrt \
    --frames=outputs/benchmark/frames.npz --output=outputs/benchmark/act_executorch_tensorrt.npz
# time per chunk, and accuracy against the reference, which comes first
python examples/export/benchmark.py report outputs/benchmark/reference_act.npz outputs/benchmark/act_*.npz
# memory of any of them: the same command with --duration=120 instead of --output
python examples/export/benchmark_memory.py -- python examples/export/benchmark.py outputs/export/act_executorch_tensorrt \
    --frames=outputs/benchmark/frames.npz --duration=120
```

## Good to know

- **Orin Nano memory.** Compiling SmolVLA, pi0.5 or GR00T on the 8 GB board may need swap space.
- **Changing the task.** A pi0.5 or GR00T folder tokenizes a new task in a short-lived process, which takes about 2 s on the Orin Nano while the arm holds still. The task the folder was exported with, and any task it has already run, needs none.
- **The checks show that the compiled policy matches PyTorch.** They do not show that the robot is safe or that it does the task.
