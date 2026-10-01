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
import gc
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from torch import Tensor, nn

from lerobot.configs.policies import PreTrainedConfig
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
        self.policy = ACTPolicy.from_pretrained(policy_path).to("cuda").eval()
        if self.policy.config.temporal_ensemble_coeff is not None:
            # The ensembler averages chunks across ticks, which one program call per chunk cannot do.
            raise ValueError(
                "ACT with temporal_ensemble_coeff set cannot be exported: its ensemble spans ticks."
            )
        self.output_dir = make_output_dir(output_dir, job_name)
        self.policy_path = policy_path
        self.config = self.policy.config
        self.module = ACTChunk(self.policy, *gpu_processors(self.config, policy_path))
        self.input_names = self.module.input_names
        self.frame = random_robot_frame(self.policy)
        observation = prepare_observation_for_inference(dict(self.frame), torch.device("cuda"))
        self.inputs = tuple(observation[name] for name in self.input_names)

    def release_policy(self) -> None:
        """Compute the test case's actions, the policy's last use, then free it for the TensorRT step."""
        self.expected_actions = self.rollout_actions()
        del self.policy, self.module
        gc.collect()
        torch.cuda.empty_cache()

    def write(self, backend: str, program_file: str, tolerance: float) -> None:
        """Save the test case, the policy config, and `export.json` beside the compiled program."""
        save_file({**self.frame, "expected_actions": self.expected_actions}, self.output_dir / TEST_CASE)
        self.config.save_pretrained(self.output_dir)
        info = {
            "backend": backend,
            "file": program_file,
            "inputs": self.input_names,
            "output": ACTION,
            "test_case": TEST_CASE,
            "tolerance": tolerance,
        }
        (self.output_dir / "export.json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"Wrote {self.output_dir}")

    def rollout_actions(self) -> np.ndarray:
        """The chunk `lerobot-rollout` plays with PyTorch for the test frame, one action per tick."""
        # Fresh processors: compiling the module leaves its own holding tensors that cannot leave the GPU.
        engine = SyncInferenceEngine(
            self.policy,
            *gpu_processors(self.policy.config, self.policy_path),
            {},
            [],
            task="",
            device="cuda",
            robot_type="",
        )
        engine.reset()
        steps = self.policy.config.n_action_steps
        return np.stack([engine.get_action(dict(self.frame)).numpy() for _ in range(steps)])


def gpu_processors(
    config: PreTrainedConfig, policy_path: str
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """The checkpoint's saved processors with every step on the GPU, whatever device they were saved with."""
    return make_pre_post_processors(
        config,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
        postprocessor_overrides={"device_processor": {"device": "cuda"}},
    )


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
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Write the folder without the engine, to build it with build_engine.py on each device.",
    )
    return parser.parse_args()
