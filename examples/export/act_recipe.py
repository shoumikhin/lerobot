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

"""The ACT export recipe: what both backend scripts compile, and the folder they write.

The compiled module is what `lerobot-rollout` runs for one action chunk after
`prepare_observation_for_inference`, all of it LeRobot's own code: the checkpoint's saved
preprocessor, the policy's `predict_action_chunk` trimmed to `n_action_steps` as `select_action`
does, and the saved postprocessor. The folder also gets a test case, the actions `lerobot-rollout`
plays with PyTorch for one random frame, which the exported engine replays before it runs.
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.rollout.inference import SyncInferenceEngine
from lerobot.utils.constants import ACTION

TEST_CASE = "test_case.safetensors"


class ACTChunk(nn.Module):
    """One action chunk: the observation tensors in, the actions to play out, in robot units."""

    def __init__(
        self, policy: ACTPolicy, preprocessor: PolicyProcessorPipeline, postprocessor: PolicyProcessorPipeline
    ):
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.input_names = list(policy.config.input_features)

    def forward(self, *inputs: Tensor) -> Tensor:
        observation = self.preprocessor(dict(zip(self.input_names, inputs, strict=True)))
        actions = self.policy.predict_action_chunk(observation)
        return self.postprocessor(actions[:, : self.policy.config.n_action_steps])


class ACTExport:
    """Loads a trained ACT checkpoint, and writes the exported folder around a compiled program."""

    def __init__(self, policy_path: str, output_dir: Path | None, job_name: str):
        self.policy_path = policy_path
        self.output_dir = make_output_dir(output_dir, job_name)
        self.policy = ACTPolicy.from_pretrained(policy_path).to("cuda").eval()
        # Keep every step on the GPU: the saved postprocessor would otherwise copy the actions to the CPU.
        preprocessor, postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
            postprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        self.module = ACTChunk(self.policy, preprocessor, postprocessor)
        self.frame = random_robot_frame(self.policy)
        observation = prepare_observation_for_inference(dict(self.frame), torch.device("cuda"))
        self.inputs = tuple(observation[name] for name in self.module.input_names)

    def write(self, backend: str, program_file: str, tolerance: float) -> None:
        """Save the test case, the policy config, and `export.json` beside the compiled program."""
        save_file({**self.frame, "expected_actions": self.rollout_actions()}, self.output_dir / TEST_CASE)
        self.policy.config.save_pretrained(self.output_dir)
        info = {
            "backend": backend,
            "file": program_file,
            "inputs": self.module.input_names,
            "output": ACTION,
            "test_case": TEST_CASE,
            "tolerance": tolerance,
        }
        (self.output_dir / "export.json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"Wrote {self.output_dir}")

    def rollout_actions(self) -> np.ndarray:
        """The chunk `lerobot-rollout` plays with PyTorch for the test frame, one action per tick."""
        preprocessor, postprocessor = make_pre_post_processors(
            self.policy.config, pretrained_path=self.policy_path
        )
        engine = SyncInferenceEngine(
            self.policy, preprocessor, postprocessor, {}, [], task="", device="cuda", robot_type=""
        )
        engine.reset()
        steps = self.policy.config.n_action_steps
        return np.stack([engine.get_action(dict(self.frame)).numpy() for _ in range(steps)])


def random_robot_frame(policy: ACTPolicy) -> dict[str, np.ndarray]:
    """A random observation shaped like the robot's: uint8 HWC images, float32 state."""
    frame = {}
    for name, feature in policy.config.input_features.items():
        if name in policy.config.image_features:
            channels, height, width = feature.shape
            frame[name] = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        else:
            frame[name] = np.random.rand(*feature.shape).astype(np.float32)
    return frame


def make_output_dir(output_dir: Path | None, job_name: str) -> Path:
    """Create the export folder, by default a new dated one under outputs/export, as lerobot-train does."""
    if output_dir is None:
        now = dt.datetime.now()
        output_dir = Path("outputs/export") / f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{job_name}"
    if output_dir.is_dir():
        raise SystemExit(f"{output_dir} already exists. Pick another --output_dir so nothing is overwritten.")
    output_dir.mkdir(parents=True)
    return output_dir


def parse_args(description: str, backend: str) -> argparse.Namespace:
    """The command line both ACT export scripts share."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy.path", dest="policy_path", required=True, help="Trained ACT checkpoint.")
    parser.add_argument(
        "--output_dir", type=Path, help="Folder to write. Default: a new one under outputs/export."
    )
    parser.add_argument("--job_name", default=f"act_{backend}", help="Names the default output folder.")
    parser.add_argument(
        "--tolerance", type=float, default=0.5, help="Largest allowed action error, robot units."
    )
    return parser.parse_args()
