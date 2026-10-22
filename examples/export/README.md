# Compiled LeRobot policies on an NVIDIA Jetson

Train a policy as usual, export it into a compiled folder, and run that folder with `lerobot-rollout`, the same command you use for a PyTorch checkpoint. In the results below, a compiled policy runs 2 to 6 times faster than in PyTorch and needs a sixth to a half of its memory, so the larger policies also fit on a small board.

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
uv pip install -e ".[smolvla,pi,groot,feetech,export]" "torch-tensorrt[executorch]" "executorch>=1.6.0.dev20261021" \
    --index-url https://download.pytorch.org/whl/nightly/cu132 --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match --prerelease allow
```

## Export

Export on the kind of board that will run the policy: a compiled folder runs only on the kind of GPU (graphics processor) that built it.

```bash
python examples/export/to_executorch_tensorrt.py \
    --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model \
    --output_dir=outputs/export/act
```

Each script compiles any supported policy one way. All use [TensorRT](https://developer.nvidia.com/tensorrt), NVIDIA's compiler for its GPUs. If you are not sure, start with ExecuTorch-TensorRT: it is fast and small, and it runs without PyTorch.

| script                      | writes                      | needs PyTorch to run |
| --------------------------- | --------------------------- | -------------------- |
| `to_executorch_tensorrt.py` | ExecuTorch `.pte` programs  | no                   |
| `to_onnx_tensorrt.py`       | TensorRT `.engine` files    | no                   |
| `to_torch_tensorrt.py`      | AOTInductor `.pt2` packages | yes                  |

Some policies need more arguments:

- **SmolVLA and GR00T** take their camera names and task from the dataset they were trained on: add `--dataset.repo_id=<user>/so101_dataset`, and `--task="<one of its tasks>"` to pick a task other than its first.
- **pi0.5** takes a task it was trained on, and the camera slots your robot fills: add `--task="pick up the block and place it in the cup" --cameras observation.images.base_0_rgb observation.images.left_wrist_0_rgb`.

A few checkpoints cannot be exported: ACT with temporal ensembling, and pi0.5 with visual or proprioceptive memory. A GR00T checkpoint that decodes its actions with its embodiment statistics exports only with `to_onnx_tensorrt.py`.

pi0.5 and GR00T are too large to compile in one piece on a small board, so the scripts build them as a chain of smaller programs, one at a time.

To run an ONNX-TensorRT export on another board, copy the folder there and rebuild its engines with `python examples/export/build_engine.py <folder>`.

## Run

Use the same calibrated robot and camera settings as when you recorded the training data:

```bash
lerobot-rollout \
    --policy.path=outputs/export/act \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: '/dev/video-wrist', width: 640, height: 480, fps: 30}, scene: {type: opencv, index_or_path: '/dev/video-scene', width: 640, height: 480, fps: 30} }" \
    --duration=30
```

pi0.5 and GR00T run any task: pass it with `--task`, or change it while the robot runs. Their folders keep the policy's own tokenizer and its whole word table, and the rollout reads only the rows of the task's words. A GR00T folder takes tasks up to 32 tokens (about 25 words); export with `--max_task_tokens` for longer ones, at a small cost in time. SmolVLA runs only the task it was exported for, so leave out `--task` or pass the same one. Without `--task`, a folder runs the task it was exported with. If your camera names differ from the policy's, add `--rename_map`, for example `--rename_map='{"observation.images.scene": "observation.images.base_0_rgb"}'`.

Before the robot moves, the rollout checks that the compiled policy reproduces the actions PyTorch computed at export time, and stops if it does not.

### Without a robot

[replay_robot](replay_robot) plays back recorded episodes and accepts every action, so you can try a policy at your desk:

```bash
uv pip install -e examples/export/replay_robot
python examples/export/replay_robot/write_frames.py \
    --dataset.repo_id=<user>/so101_dataset --episodes 0 1 2 3 --output_dir=outputs/replay_frames
lerobot-rollout \
    --policy.path=outputs/export/act \
    --robot.type=replay --robot.frames_dir=outputs/replay_frames \
    --duration=60 --return_to_initial_position=false --play_sounds=false
```

## Results

The tables compare each policy, on each board, compiled each way, with `torch.compile` and plain PyTorch. The `torch.compile` and PyTorch columns run the same pieces of the policy that the export compiles, through the same inference engine, so only what runs those pieces differs. [benchmark.py](benchmark.py) measures time and accuracy, and [benchmark_memory.py](benchmark_memory.py) measures memory around the same run.

- All four policies were trained on the same SO-101 dataset: 304 episodes, recorded at 30 frames per second.
- Both boards ran at maximum power, with the GPU clock locked.
- **Bold** marks the best value in a row.
- OOM (out of memory) means the policy did not fit on the 8 GB Orin Nano.

Measured with PyTorch 2.15.0.dev20261005 and TensorRT 11.3.0.99.

### Time per chunk

A policy returns a chunk of actions at a time (ACT 100, SmolVLA and pi0.5 50, GR00T 16). `lerobot-rollout` plays the whole chunk, then runs the policy again while the arm holds still, so this time is the pause between chunks. Milliseconds, lower is better.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |               **3** |         **3** |              4 |              13 |      16 |
| SmolVLA |              **29** |            30 |             30 |              44 |     164 |
| pi0.5   |                 125 |       **115** |            121 |             124 |     237 |
| GR00T   |              **94** |            96 |         **94** |             206 |     211 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |              **25** |        **25** |             26 |              61 |      64 |
| SmolVLA |                 125 |       **121** |            137 |             170 |     749 |
| pi0.5   |                 630 |       **612** |            OOM |             OOM |     OOM |
| GR00T   |             **266** |           274 |            275 |             OOM |     OOM |

Each value is the median of three runs of 100 chunks. Differences of a few milliseconds are within run-to-run noise. Every compiled policy, and `torch.compile`, replays its programs as CUDA graphs.

### Memory

Memory the policy's process holds while it runs, in MiB: its own memory, what it swapped out, and its GPU memory (the main processor, or CPU, and the GPU share one memory on a Jetson). Files it only reads are not counted, since the board can drop them from memory. Lower is better. Measured over the last 30 s of 90 s of chunks, after the same 20 warm-up chunks as the time per chunk.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |             **545** |           551 |           1523 |            2404 |    2039 |
| SmolVLA |            **1247** |          1310 |           2836 |            4706 |    4568 |
| pi0.5   |                6050 |      **5932** |           7600 |           13186 |   11618 |
| GR00T   |            **5405** |          5420 |           6068 |           16361 |   15708 |

Orin Nano:

|         | ExecuTorch-TensorRT | ONNX-TensorRT | Torch-TensorRT | `torch.compile` | PyTorch |
| ------- | ------------------: | ------------: | -------------: | --------------: | ------: |
| ACT     |             **207** |           222 |            841 |            1677 |    1136 |
| SmolVLA |             **373** |           475 |           1764 |            2493 |    2447 |
| pi0.5   |                4983 |      **4882** |            OOM |             OOM |     OOM |
| GR00T   |            **4402** |          4408 |           7534 |             OOM |     OOM |

### Accuracy

How far each column's actions are from the original LeRobot policy's, run in full float32 (no bfloat16 and no TensorFloat-32), on 50 dataset samples with the same random noise. For each sample, take the largest difference over its actions and joints; each cell is the mean / 95th percentile / largest of that over the 50 samples, in degrees (gripper in %). Lower is better. Every cell passed the export's own check. The reference is the same on both boards, so it runs once, on Thor: in float32, pi0.5 and GR00T do not fit on the Orin Nano.

Thor:

|         | ExecuTorch-TensorRT | ONNX-TensorRT |   Torch-TensorRT |      `torch.compile` |
| ------- | ------------------: | ------------: | ---------------: | -------------------: |
| ACT     |        0.14 / 0.004 |  0.14 / 0.004 |     0.14 / 0.004 | **0.05** / **0.001** |
| SmolVLA |        1.72 / 0.048 |  2.00 / 0.049 |     1.72 / 0.048 | **1.68** / **0.045** |
| pi0.5   |    **0.66** / 0.088 |  1.33 / 0.090 | **0.66** / 0.088 |     0.71 / **0.078** |
| GR00T   |    **1.62** / 0.089 |  1.76 / 0.087 | **1.62** / 0.089 |     3.06 / **0.064** |

Orin Nano:

|         |  ExecuTorch-TensorRT |    ONNX-TensorRT |       Torch-TensorRT |      `torch.compile` |
| ------- | -------------------: | ---------------: | -------------------: | -------------------: |
| ACT     |     **0.13** / 0.004 |     0.15 / 0.004 |     **0.13** / 0.004 |     0.20 / **0.002** |
| SmolVLA |         4.03 / 0.058 | 2.76 / **0.044** |         4.03 / 0.058 | **1.42** / **0.044** |
| pi0.5   |     0.97 / **0.097** | **0.70** / 0.098 |                  OOM |                  OOM |
| GR00T   | **1.79** / **0.085** |     2.14 / 0.091 | **1.79** / **0.085** |                  OOM |

The largest error usually comes from one or two samples where the policy is close to a decision, so the mean is the steadier number. ExecuTorch-TensorRT and Torch-TensorRT build the same TensorRT engines, so their accuracy is the same.

## Good to know

- **Orin Nano memory.** Compiling SmolVLA, pi0.5 or GR00T on the 8 GB board may need swap space.
- **Changing the task.** A pi0.5 or GR00T folder tokenizes a new task in a short-lived process, which takes about 2 s on the Orin Nano while the arm holds still. The task the folder was exported with needs none.
- **The checks show that the compiled policy matches PyTorch.** They do not show that the robot is safe or that it does the task.
