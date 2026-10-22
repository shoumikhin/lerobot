# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""What the policy recipes and the route scripts share: the command line, the part loop, the folder.

A recipe turns one policy into parts a compiler can take: modules with example inputs, and named inputs
and outputs. A whole policy is one part. A policy too large to build at once is a chain of parts, built
one per process, so only one part's weights are ever in memory. A route script compiles each part; the
recipe then writes the test case and `export.json`, which `lerobot-rollout` reads.
"""

import argparse
import dataclasses
import datetime as dt
import gc
import importlib
import json
import subprocess
import sys
from collections.abc import Callable
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from build_engine import WORKSPACE_BYTES, check_folder
from safetensors.numpy import save_file
from safetensors.torch import save_file as save_tensors
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.inference.export.engine import CAMERAS_PRESENT, EXPORT_INFO, NOISE
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    DeviceProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    to_absolute_actions,
)
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR
from lerobot.utils.feature_utils import dataset_to_policy_features

POLICIES = ("act", "smolvla", "pi05", "groot")
TEST_CASE = "test_case.safetensors"
# The random test images are blends of a coarse grid of random colors (rows, columns).
IMAGE_CELLS = (6, 8)
# The same for every policy: they fit the builds into an 8 GB Jetson Orin Nano beside the rest.
TENSORRT_OPTIONS = {
    "min_block_size": 1,
    # Flow-matching time embeddings are computed in float64, which TensorRT does not support.
    "truncate_double": True,
    # On a Jetson the CPU and GPU share memory: CPU memory can swap during the build, GPU memory cannot.
    "offload_module_to_cpu": True,
    "workspace_size": WORKSPACE_BYTES,
    # Attention as plain matrix products, as the ONNX route writes it, so both TensorRT routes build alike.
    "decompose_attention": True,
}


@dataclasses.dataclass
class Part:
    """One program of the chunk: a module, its example inputs, and the names of its inputs and outputs."""

    name: str
    module: nn.Module
    inputs: tuple[Tensor, ...]
    input_names: list[str]
    output_names: list[str]
    # Torch-TensorRT options this part needs on top of TENSORRT_OPTIONS.
    tensorrt_options: dict[str, Any] = dataclasses.field(default_factory=dict)


def parse_args(
    description: str,
    route: str,
    parents: tuple[argparse.ArgumentParser, ...] = (),
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, Any]:
    """The route's command line, or `argv`, with the arguments of the checkpoint's recipe and of `parents`,
    and that recipe."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        parents=list(parents),
    )
    parser.add_argument("--policy.path", dest="policy_path", required=True, help="Trained policy checkpoint.")
    parser.add_argument(
        "--output_dir", type=Path, help="Folder to write. Default: a new one under outputs/export."
    )
    parser.add_argument("--part", help=argparse.SUPPRESS)
    # The checkpoint names the recipe, which adds its own arguments, so it is read before help is shown.
    checkpoint = argparse.ArgumentParser(add_help=False)
    checkpoint.add_argument("--policy.path", dest="policy_path")
    policy_path = checkpoint.parse_known_args(argv)[0].policy_path
    recipe = None
    if policy_path is not None:
        policy_type = PreTrainedConfig.from_pretrained(policy_path).type
        if policy_type not in POLICIES:
            raise SystemExit(
                f"No export recipe for {policy_type!r} policies; supported: {', '.join(POLICIES)}."
            )
        recipe = importlib.import_module(policy_type)
        parser.add_argument(
            "--tolerance",
            type=float,
            default=recipe.TOLERANCE,
            help="Largest allowed action error, robot units.",
        )
        recipe.add_arguments(parser)
    parser.add_argument(
        "-h", "--help", action="help", help="Show this help; with --policy.path, the recipe's too."
    )
    parser.set_defaults(route=route)
    return parser.parse_args(argv), recipe


def run_route(
    script: str,
    args: argparse.Namespace,
    recipe: Any,
    compile_part: Callable[[Part, Path], None],
    suffix: str,
) -> None:
    """Compile each part of the checkpoint with `compile_part` into `<part><suffix>`, write the folder, and
    check that the folder reproduces its test case, as `lerobot-rollout` does before the robot moves.

    Each part is built by running `script` again, because a process that built a part keeps some of its
    memory until it exits. That process runs the parts before its own in PyTorch to make its example inputs.
    The process that started the export then checks the folder with only the compiled programs in memory.
    """
    names = recipe.part_names(args)
    if args.part is None:
        args.output_dir = make_output_dir(args.output_dir, f"{recipe.__name__}_{args.route}")
        for name in names:
            command = [
                sys.executable,
                script,
                *sys.argv[1:],
                f"--output_dir={args.output_dir}",
                f"--part={name}",
            ]
            subprocess.run(command, check=True)
        check_folder(args.output_dir)
        return
    if args.part not in names:
        raise SystemExit(f"Unknown part {args.part!r}; choose from {names}.")
    exporter = recipe.Exporter(args)
    programs = {}
    for part in exporter.parts():
        programs[part.name] = {
            "file": f"{part.name}{suffix}",
            "inputs": part.input_names,
            "outputs": part.output_names,
        }
        if part.name == args.part:
            print(f"Compiling {part.name}", flush=True)
            compile_part(part, args.output_dir / programs[part.name]["file"])
            break
        del part
        free_memory()
    else:
        raise SystemExit(f"The recipe built no part named {args.part!r}; its parts are {list(programs)}.")
    if args.part == names[-1]:
        exporter.write(programs)


def write_folder(
    args: argparse.Namespace,
    config: PreTrainedConfig,
    frame: dict[str, np.ndarray],
    noise: Tensor | None,
    expected_actions: np.ndarray,
    programs: dict[str, dict],
    **info: Any,
) -> None:
    """Save the test case, the policy config and `export.json`, which `lerobot-rollout` reads."""
    case = {**frame, "expected_actions": expected_actions}
    if noise is not None:
        case[NOISE] = noise.cpu().numpy()
        info["noise_shape"] = list(noise.shape)
    save_file(case, args.output_dir / TEST_CASE)
    config.save_pretrained(args.output_dir)
    info = {
        "route": args.route,
        "programs": programs,
        "inputs": [
            *frame,
            *([CAMERAS_PRESENT] if "optional_cameras" in info else []),
            *([NOISE] if noise is not None else []),
        ],
        "test_case": TEST_CASE,
        "tolerance": args.tolerance,
        **info,
    }
    (args.output_dir / EXPORT_INFO).write_text(json.dumps(info, indent=2) + "\n")
    print(f"Wrote {args.output_dir}")


def save_embedding_table(weight: Tensor, path: Path) -> None:
    """Save a token embedding table for the rollout to read rows from, in bfloat16 when that holds it exactly.

    GR00T's checkpoint stores its bfloat16 table in float32, which would double the file and every row read.
    """
    weight = weight.detach()
    if weight.dtype == torch.float32 and torch.equal(weight, weight.to(torch.bfloat16).float()):
        weight = weight.to(torch.bfloat16)
    save_tensors({"weight": weight.contiguous()}, path)


def make_output_dir(output_dir: Path | None, job_name: str) -> Path:
    """Create the export folder, by default a new dated one under outputs/export, as lerobot-train does."""
    if output_dir is None:
        now = dt.datetime.now()
        output_dir = Path("outputs/export") / f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{job_name}"
    if output_dir.is_dir():
        raise SystemExit(f"{output_dir} already exists. Pick another --output_dir so nothing is overwritten.")
    output_dir.mkdir(parents=True)
    return output_dir


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def processors(
    config: PreTrainedConfig, policy_path: str
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """The checkpoint's saved processors with every step on this machine's device."""
    overrides = {"device_processor": {"device": str(device())}}
    return make_pre_post_processors(
        config,
        pretrained_path=policy_path,
        preprocessor_overrides=overrides,
        postprocessor_overrides=overrides,
    )


def to_policy_input(name: str, x: Tensor) -> Tensor:
    """What `prepare_observation_for_inference` does to one array of the robot's frame, inside the program."""
    if "image" in name:
        x = (x.float() / 255).permute(2, 0, 1)
    return x.unsqueeze(0)


def random_image(height: int, width: int, channels: int) -> np.ndarray:
    """A random uint8 HWC image that changes smoothly across the frame, as a camera's image does.

    On independent random pixels, which look like no camera image, a vision-language policy's bfloat16
    actions differ by several robot units between two correct compilations, so its check would measure noise.
    """
    coarse = torch.rand(1, channels, *IMAGE_CELLS) * 255
    image = nn.functional.interpolate(coarse, size=(height, width), mode="bilinear")
    # Contiguous, because safetensors saves a strided array in memory order, which scrambles it.
    return image[0].permute(1, 2, 0).round().to(torch.uint8).contiguous().numpy()


def frame_from_policy(config: PreTrainedConfig) -> dict[str, np.ndarray]:
    """A random observation shaped like the policy's inputs: uint8 HWC images, float32 state."""
    frame = {}
    for name, feature in config.input_features.items():
        if name in config.image_features:
            channels, height, width = feature.shape
            frame[name] = random_image(height, width, channels)
        else:
            frame[name] = np.random.rand(*feature.shape).astype(np.float32)
    return frame


def frame_from_dataset(features: dict[str, dict]) -> dict[str, np.ndarray]:
    """A random observation named and shaped like the robot's in the dataset: uint8 HWC images, float32."""
    frame = {}
    for name, feature in features.items():
        if not name.startswith(f"{OBS_STR}."):
            continue
        shape = feature["shape"]
        if feature["dtype"] in ("image", "video"):
            # The robot's cameras give HWC frames; a dataset may store the shape channel first.
            if feature["names"][2] not in ("channel", "channels"):
                shape = (shape[1], shape[2], shape[0])
            frame[name] = random_image(*shape)
        else:
            frame[name] = np.random.rand(*shape).astype(np.float32)
    return frame


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    """For a checkpoint naming placeholder cameras: the robot's cameras and tasks come from its dataset."""
    parser.add_argument(
        "--dataset.repo_id", dest="dataset", required=True, help="The dataset the policy was trained on."
    )
    parser.add_argument("--dataset.root", dest="dataset_root", type=Path, help="Local copy of that dataset.")
    parser.add_argument("--task", help="One of the dataset's tasks to export for. Default: its first task.")


def dataset_task(metadata: LeRobotDatasetMetadata, task: str | None) -> str:
    """`task`, which must be one of the dataset's tasks, or the dataset's first task."""
    tasks = [str(t) for t in metadata.tasks.index]
    if task is None:
        return tasks[0]
    if task not in tasks:
        raise ValueError(f"--task must be one of the dataset's tasks: {tasks}.")
    return task


def dataset_frame(args: argparse.Namespace, config: PreTrainedConfig) -> tuple[str, dict, PreTrainedConfig]:
    """The task, a frame named like the robot's in the training dataset, and `config` naming that frame.

    The folder takes the robot's cameras, so its config names them, not the checkpoint's placeholders.
    """
    metadata = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    frame = frame_from_dataset(metadata.features)
    folder_config = copy(config)
    features = dataset_to_policy_features(metadata.features)
    folder_config.input_features = {name: features[name] for name in frame}
    return dataset_task(metadata, args.task), frame, folder_config


def renamed_inputs(preprocessor: PolicyProcessorPipeline) -> dict[str, str]:
    """The saved preprocessor's renames, from the robot's observation names to the policy's."""
    return {
        old: new
        for step in preprocessor.steps
        if isinstance(step, RenameObservationsProcessorStep)
        for old, new in step.rename_map.items()
    }


def anchored_postprocessor(postprocessor: PolicyProcessorPipeline) -> tuple[PolicyProcessorPipeline, Any]:
    """The postprocessor's steps before its relative-to-absolute step, and that step.

    That step adds the state the preprocessor cached in Python, which an exported program would keep as a
    constant. `ChunkActions` applies it to the state the program takes as an input instead.
    """
    steps = list(postprocessor.steps)
    index = next(
        (i for i, s in enumerate(steps) if isinstance(s, AbsoluteActionsProcessorStep) and s.enabled), None
    )
    if index is None:
        return postprocessor, None
    if not all(isinstance(s, DeviceProcessorStep) for s in steps[index + 1 :]):
        raise ValueError("The relative-to-absolute action step must be the postprocessor's last step.")
    return dataclasses.replace(postprocessor, steps=steps[:index]), steps[index]


class ChunkActions(nn.Module):
    """The end of a chain: the denoised actions in, the actions to play out, in robot units.

    A policy trained on actions relative to the state turns them back around the robot's state, which the
    program then takes as an input too.
    """

    def __init__(self, n_action_steps: int, action_dim: int, postprocessor: PolicyProcessorPipeline):
        super().__init__()
        self.n_action_steps, self.action_dim = n_action_steps, action_dim
        self.postprocessor, self.absolute_step = anchored_postprocessor(postprocessor)

    def forward(self, actions: Tensor, state: Tensor | None = None) -> Tensor:
        actions = self.postprocessor(actions[None, : self.n_action_steps, : self.action_dim])
        if self.absolute_step is not None:
            # The relative step has no public mask accessor; LeRobot's own RTC and stats code call this one too.
            mask = self.absolute_step.relative_step._build_mask(actions.shape[-1])
            actions = to_absolute_actions(actions, state[None], mask)
        return actions[0]

    def part(self, actions: Tensor, state: Tensor) -> Part:
        """The last part, which takes the state only if it reads it.

        The rollout hands a program every input on the GPU, but the export places on the GPU only the inputs
        the program reads, so a state it never reads would stay a host input that the rollout cannot feed.
        """
        if self.absolute_step is None:
            return Part("actions", self, (actions,), ["x_t"], [ACTION])
        return Part("actions", self, (actions, state), ["x_t", OBS_STATE], [ACTION])


def hold_linear_weights_in_bfloat16(module: nn.Module) -> None:
    """Store the linear layers of `module`, which runs them only under bfloat16 autocast, in bfloat16.

    Autocast casts a float32 linear layer's weight and bias to bfloat16 on every call, so this is the same
    arithmetic without copying the weights every chunk: what a TensorRT build does once, when it folds the
    cast into the weights.
    """
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            layer.to(torch.bfloat16)


def free_memory() -> None:
    """Return freed GPU memory to the next part or the build."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
