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

"""The SmolVLA export recipe: what both backend scripts compile, and the folder they write.

As for ACT, the compiled module is what `lerobot-rollout` runs for one action chunk, all of it LeRobot's
own code: the checkpoint's saved preprocessor, the policy's `predict_action_chunk`, and the saved
postprocessor. Two things differ from ACT:

- The task text. A TensorRT engine cannot tokenize, so the preprocessor's text steps (the newline and
  the tokenizer) run before the program, which takes the token ids and the attention mask.
- The noise. Each chunk starts from random noise. The program takes it as an input, so the rollout draws
  it, and the test case gives the program the noise the PyTorch policy got.

The robot's cameras and the task come from the dataset the policy was trained on, because SmolVLA's
checkpoint only names placeholder cameras. The folder's config names the robot's cameras, so the rollout's
camera check matches the robot without a --rename_map.
"""

import argparse
import gc
import json
from copy import copy
from pathlib import Path

import numpy as np
import torch
from act_recipe import TEST_CASE, gpu_processors, make_output_dir
from safetensors.numpy import save_file
from torch import Tensor, nn

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import NewLineTaskProcessorStep, PolicyProcessorPipeline, TokenizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STR
from lerobot.utils.feature_utils import dataset_to_policy_features

NOISE = "noise"
TEXT_STEPS = "text_steps.json"


class SmolVLAChunk(nn.Module):
    """One action chunk: the robot's observation, the task tokens and the noise in, the actions out."""

    def __init__(
        self,
        policy: SmolVLAPolicy,
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


class SmolVLAExport:
    """Loads a trained SmolVLA checkpoint, and writes the exported folder around a compiled program."""

    def __init__(
        self,
        policy_path: str,
        dataset: str,
        dataset_root: Path | None,
        output_dir: Path | None,
        job_name: str,
    ):
        self.output_dir = make_output_dir(output_dir, job_name)
        self.policy_path = policy_path
        self.policy = SmolVLAPolicy.from_pretrained(policy_path).to("cuda").eval()
        self.config = self.policy.config
        preprocessor, postprocessor = self.processors()
        text_steps = [
            s for s in preprocessor.steps if isinstance(s, NewLineTaskProcessorStep | TokenizerProcessorStep)
        ]
        self.text_steps = PolicyProcessorPipeline(steps=text_steps)
        tensor_steps = PolicyProcessorPipeline(steps=[s for s in preprocessor.steps if s not in text_steps])

        metadata = LeRobotDatasetMetadata(dataset, root=dataset_root)
        self.task = str(metadata.tasks.index[0])
        self.frame = random_robot_frame(metadata.features)
        # The folder takes the robot's cameras, so its config names them, not the checkpoint's placeholders.
        self.input_features = {
            name: feature
            for name, feature in dataset_to_policy_features(metadata.features).items()
            if name in self.frame
        }
        observation = self.text_steps(
            prepare_observation_for_inference(dict(self.frame), torch.device("cuda"), self.task)
        )
        names = [*self.frame, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, NOISE]
        config = self.policy.config
        self.noise = torch.randn(1, config.chunk_size, config.max_action_dim, device="cuda")
        self.inputs = (*(observation[name] for name in names[:-1]), self.noise)
        self.module = SmolVLAChunk(self.policy, tensor_steps, postprocessor, names)
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
        """The checkpoint's processors on the GPU, with the tokenizer padding every task to one width."""
        preprocessor, postprocessor = gpu_processors(self.policy.config, self.policy_path)
        # One width for every task, so the program's token inputs keep the shape they were exported with.
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


def random_robot_frame(features: dict[str, dict]) -> dict[str, np.ndarray]:
    """A random observation shaped and named like the robot's in the dataset: uint8 HWC images, float32 state."""
    frame = {}
    for name, feature in features.items():
        if not name.startswith(f"{OBS_STR}."):
            continue
        if feature["dtype"] in ("image", "video"):
            frame[name] = np.random.randint(0, 256, feature["shape"], dtype=np.uint8)
        else:
            frame[name] = np.random.rand(*feature["shape"]).astype(np.float32)
    return frame


def parse_args(description: str, backend: str) -> argparse.Namespace:
    """The command line both SmolVLA export scripts share."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--policy.path", dest="policy_path", required=True, help="Trained SmolVLA checkpoint."
    )
    parser.add_argument(
        "--dataset.repo_id", dest="dataset", required=True, help="The dataset the policy was trained on."
    )
    parser.add_argument(
        "--dataset.root",
        dest="dataset_root",
        type=Path,
        help="Local copy of that dataset, if not on the Hub.",
    )
    parser.add_argument(
        "--output_dir", type=Path, help="Folder to write. Default: a new one under outputs/export."
    )
    parser.add_argument("--job_name", default=f"smolvla_{backend}", help="Names the default output folder.")
    # The backbone runs in bfloat16 and TensorRT reorders its math, so a SmolVLA engine's actions drift from
    # PyTorch's by up to about 4 on this arm, against 13 or more for wrong noise and usually 5 or more for swapped
    # cameras. So the check catches gross mistakes, not small ones.
    parser.add_argument(
        "--tolerance", type=float, default=5.0, help="Largest allowed action error, robot units."
    )
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Write the folder without the engine, to build it with build_engine.py on each device.",
    )
    return parser.parse_args()
