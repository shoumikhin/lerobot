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

"""The generic recipe, for the policies in SPECS: one action chunk of LeRobot's own policy, as one program.

As for SmolVLA, the program goes from the robot's frame and the noise to actions in robot units, and holds
the task as constants, so the folder runs the one task it was exported for. The cameras are the checkpoint's
own. A policy is exported only if it keeps LeRobot's own contract for running one chunk from the current
frame, `chunk_inference_spec`, and if PyTorch computes the same chunk twice from the same frame and noise:
a folder cannot reproduce frames from earlier ticks or random numbers the policy draws itself.
"""

import argparse
import dataclasses
from collections.abc import Iterator
from copy import copy

import numpy as np
import torch
from common import NOISE, Part, TaskChunk, device, frame_from_policy, processors, write_folder
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.pi0.processor_pi0 import Pi0NewLineProcessor
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import TokenizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_IMAGES

# pi0 checkpoints are float32 by default, as ACT's are; a bfloat16 one may need a larger --tolerance.
TOLERANCE = 0.5


@dataclasses.dataclass(frozen=True)
class Spec:
    """What the recipe needs to know about one policy type."""

    # The preprocessor steps that turn the task into tokens, which run once, at export.
    text_steps: tuple[type, ...] = ()


SPECS = {
    "pi0": Spec(text_steps=(Pi0NewLineProcessor, TokenizerProcessorStep)),
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", default="", help="The task the folder runs, for a policy that reads one.")


def part_names(args: argparse.Namespace) -> list[str]:
    return ["model"]


def robot_frame(config: PreTrainedConfig) -> dict[str, np.ndarray]:
    """A random frame of what the robot sends: the policy fills in its empty camera slots itself."""
    empty_cameras = {f"{OBS_IMAGES}.empty_camera_{i}" for i in range(getattr(config, "empty_cameras", 0))}
    return {name: value for name, value in frame_from_policy(config).items() if name not in empty_cameras}


class Exporter:
    """Loads a trained checkpoint of a policy in SPECS, and writes the folder around its compiled program."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        config = PreTrainedConfig.from_pretrained(args.policy_path)
        # As `lerobot-rollout` does: how the policy runs, not the saved checkpoint, decides what compiles it.
        if hasattr(config, "compile_model"):
            config.compile_model = False
        self.spec = SPECS[config.type]
        if self.spec.text_steps and not args.task:
            raise SystemExit(f"A {config.type} folder runs one task: give it with --task.")
        self.policy = get_policy_class(config.type).from_pretrained(args.policy_path, config=config)
        self.policy = self.policy.to(device()).eval()
        try:
            self.policy.chunk_inference_spec()
        except ValueError as error:
            raise SystemExit(f"This {config.type} checkpoint cannot be exported: {error}") from error
        self.frame = robot_frame(self.policy.config)
        self.noise = torch.randn(
            self.policy.config.chunk_size, self.policy.config.max_action_dim, device=device()
        )
        self.expected_actions = self.rollout_actions(self.frame)
        if not np.array_equal(self.expected_actions, self.rollout_actions(self.frame)):
            raise SystemExit(
                f"This {config.type} policy computes another chunk from the same frame and noise each time, so "
                "a folder cannot reproduce it."
            )

    def parts(self) -> Iterator[Part]:
        module = TaskChunk(
            self.policy,
            *processors(self.policy.config, self.args.policy_path),
            list(self.frame),
            self.observation(self.frame),
            self.spec.text_steps,
        )
        # A text step that read the frame would freeze it, and the folder's check reuses that frame: try another.
        frame = robot_frame(self.policy.config)
        with torch.inference_mode():
            actions = module(*self.inputs(frame)).cpu().numpy()
        if not np.array_equal(actions, self.rollout_actions(frame)):
            raise SystemExit(f"The {self.policy.config.type} program would not follow the robot's frame.")
        yield Part("model", module, self.inputs(self.frame), [*self.frame, NOISE], [ACTION])

    def inputs(self, frame: dict[str, np.ndarray]) -> tuple[Tensor, ...]:
        """The program's inputs for `frame`: its arrays and the noise."""
        return (*(torch.from_numpy(x).to(device()) for x in frame.values()), self.noise)

    def observation(self, frame: dict[str, np.ndarray]) -> dict:
        """`frame` and the task as `lerobot-rollout` hands them to the policy's preprocessor."""
        return prepare_observation_for_inference(dict(frame), device(), self.args.task)

    def rollout_actions(self, frame: dict[str, np.ndarray]) -> np.ndarray:
        """The chunk the PyTorch policy computes for `frame`, the task and the noise, as `lerobot-rollout` does."""
        preprocessor, postprocessor = processors(self.policy.config, self.args.policy_path)
        self.policy.reset()
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(
                preprocessor(self.observation(frame)), noise=self.noise[None]
            )
            actions = postprocessor(actions[:, : self.policy.config.n_action_steps])
        return actions[0].cpu().numpy()

    def write(self, programs: dict[str, dict]) -> None:
        # The folder's config lists only what the robot sends, as the frame does, like pi0.5's.
        config = copy(self.policy.config)
        config.input_features = {name: config.input_features[name] for name in self.frame}
        write_folder(
            self.args,
            config,
            self.frame,
            self.noise,
            self.expected_actions,
            programs,
            # The program holds the task, so the folder runs only this task.
            task=self.args.task,
        )
