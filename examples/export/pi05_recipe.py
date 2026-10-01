#!/usr/bin/env python

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

"""The pi0.5 export recipe: what both backend scripts compile, and the folder they write.

As for SmolVLA, the compiled module is one action chunk of LeRobot's own code: the checkpoint's saved
preprocessor, the policy's `predict_action_chunk` and the saved postprocessor, with the starting noise
as the program's last input. One thing differs:

- The state. pi0.5 reads the robot's state only as text: the preprocessor normalizes it, rounds it into
  256 bins and writes the bins into the prompt before the tokenizer. So every step up to the tokenizer
  runs before the program, which takes the cameras, the token ids and the attention mask.

The checkpoint names the robot's cameras, so they come from its config, as for ACT, except the empty
cameras pi0.5 fills in itself. The folder's config names only the cameras the program takes. The
checkpoint does not record the task, so the export takes it on the command line.
"""

import argparse
import gc
import json
from copy import copy
from pathlib import Path

import numpy as np
import torch
from act_recipe import TEST_CASE, gpu_processors, make_output_dir, random_robot_frame
from safetensors.numpy import save_file
from smolvla_recipe import NOISE, TEXT_STEPS
from torch import Tensor, nn

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, TokenizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


class PI05Chunk(nn.Module):
    """One action chunk: the cameras, the prompt's tokens and the noise in, the actions out."""

    def __init__(
        self,
        policy: PI05Policy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        input_names: list[str],
    ):
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.input_names = input_names

    def forward(self, *inputs: Tensor) -> Tensor:
        *observation, noise = inputs
        batch = self.preprocessor(dict(zip(self.input_names[:-1], observation, strict=True)))
        actions = self.policy.predict_action_chunk(batch, noise=noise)
        return self.postprocessor(actions[:, : self.policy.config.n_action_steps])


class PI05Export:
    """Loads a trained pi0.5 checkpoint, and writes the exported folder around a compiled program."""

    def __init__(self, policy_path: str, task: str, output_dir: Path | None, job_name: str):
        self.output_dir = make_output_dir(output_dir, job_name)
        self.policy_path = policy_path
        self.task = task
        self.policy = PI05Policy.from_pretrained(policy_path).to("cuda").eval()
        self.config = self.policy.config
        preprocessor, postprocessor = self.processors()
        tokenizer = next(
            i for i, step in enumerate(preprocessor.steps) if isinstance(step, TokenizerProcessorStep)
        )
        self.text_steps = PolicyProcessorPipeline(steps=preprocessor.steps[: tokenizer + 1])
        tensor_steps = PolicyProcessorPipeline(steps=preprocessor.steps[tokenizer + 1 :])

        config = self.policy.config
        # pi0.5 fills in its empty cameras itself, so the robot does not send them.
        empty_cameras = {f"{OBS_IMAGES}.empty_camera_{i}" for i in range(config.empty_cameras)}
        self.frame = {
            name: value
            for name, value in random_robot_frame(self.policy).items()
            if name not in empty_cameras
        }
        self.input_features = {name: config.input_features[name] for name in self.frame}
        observation = self.text_steps(
            prepare_observation_for_inference(dict(self.frame), torch.device("cuda"), task)
        )
        cameras = [name for name in self.frame if name in config.image_features]
        names = [*cameras, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, NOISE]
        self.noise = torch.randn(1, config.chunk_size, config.max_action_dim, device="cuda")
        self.inputs = (*(observation[name] for name in names[:-1]), self.noise)
        self.module = PI05Chunk(self.policy, tensor_steps, postprocessor, names)
        self.input_names = names

    def release_policy(self) -> None:
        """Compute the test case's actions, the policy's last use, then free it for the TensorRT step."""
        self.expected_actions = self.rollout_actions()
        del self.policy, self.module
        gc.collect()
        torch.cuda.empty_cache()

    def write(self, backend: str, program_file: str, tolerance: float) -> None:
        """Save the test case, the policy config, the text steps and `export.json` beside the program."""
        case = {**self.frame, NOISE: self.noise.cpu().numpy(), "expected_actions": self.expected_actions}
        save_file(case, self.output_dir / TEST_CASE)
        config = copy(self.config)
        config.input_features = self.input_features
        config.save_pretrained(self.output_dir)
        self.text_steps.save_pretrained(self.output_dir, config_filename=TEXT_STEPS)
        info = {
            "backend": backend,
            "file": program_file,
            "inputs": self.input_names,
            "text_steps": TEXT_STEPS,
            "task": self.task,
            "noise_shape": list(self.noise.shape),
            "output": ACTION,
            "test_case": TEST_CASE,
            "tolerance": tolerance,
        }
        (self.output_dir / "export.json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"Wrote {self.output_dir}")

    def processors(self) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
        """The checkpoint's processors on the GPU, with the tokenizer padding every prompt to one width."""
        preprocessor, postprocessor = gpu_processors(self.policy.config, self.policy_path)
        # The prompt carries the state, so its length can change every step; the program takes one shape.
        next(s for s in preprocessor.steps if isinstance(s, TokenizerProcessorStep)).padding = "max_length"
        return preprocessor, postprocessor

    def rollout_actions(self) -> np.ndarray:
        """The chunk the PyTorch policy computes for the test frame, task and noise, as lerobot-rollout runs it."""
        # Fresh processors: compiling the module leaves its own holding tensors that cannot leave the GPU.
        preprocessor, postprocessor = self.processors()
        self.policy.reset()
        observation = prepare_observation_for_inference(dict(self.frame), torch.device("cuda"), self.task)
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(preprocessor(observation), noise=self.noise)
            actions = postprocessor(actions[:, : self.policy.config.n_action_steps])
        return actions[0].cpu().numpy()


def parse_args(description: str, backend: str) -> argparse.Namespace:
    """The command line both pi0.5 export scripts share."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy.path", dest="policy_path", required=True, help="Trained pi0.5 checkpoint.")
    parser.add_argument("--task", required=True, help="The task the policy was trained on, as a sentence.")
    parser.add_argument(
        "--output_dir", type=Path, help="Folder to write. Default: a new one under outputs/export."
    )
    parser.add_argument("--job_name", default=f"pi05_{backend}", help="Names the default output folder.")
    # As for SmolVLA, the backbone runs in bfloat16 and TensorRT reorders its math, so a pi0.5 engine's actions
    # drift from PyTorch's by up to about 4 on this arm, against 15 or more for wrong noise. Swapped cameras often
    # stay under 5, so the check catches gross mistakes, not small ones.
    parser.add_argument(
        "--tolerance", type=float, default=5.0, help="Largest allowed action error, robot units."
    )
    return parser.parse_args()
