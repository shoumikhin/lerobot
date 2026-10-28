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

"""Recipes: turn a trained policy into the parts a backend compiles, and the test case of its bundle."""

from __future__ import annotations

import abc
import inspect
from collections.abc import Iterator
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies import PreTrainedPolicy, prepare_observation_for_inference
from lerobot.policies.factory import _import_sibling_policy_module
from lerobot.processor import PolicyProcessorPipeline, RenameObservationsProcessorStep
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STR
from lerobot.utils.feature_utils import dataset_to_policy_features

from .configs import ExportConfig

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDatasetMetadata

# A compiler that reorders bfloat16 math moves SmolVLA's actions by up to about 4 robot units, while a
# wrong noise input moves them by 13 or more. Float32 policies stay well within 0.5.
FLOAT32_TOLERANCE = 0.5
BFLOAT16_TOLERANCE = 5.0
# A random test image blends a coarse grid of random colors (rows, columns), so it changes smoothly like a
# camera image. On independent random pixels a vision-language policy's bfloat16 actions differ by several
# robot units between two correct compilations, so the check would measure noise.
IMAGE_CELLS = (6, 8)


@dataclass
class Part:
    """One program of a policy.

    Args:
        name (`str`):
            The program's name, such as `model`.
        module (`nn.Module`):
            What the program computes.
        example_inputs (`tuple[Tensor, ...]`):
            Inputs to trace `module` with, on the backend's device.
        input_names (`list[str]`):
            The names of the inputs: keys of the robot's frame, `noise`, or outputs of earlier parts.
        output_names (`list[str]`):
            The names of the outputs. The last part outputs `action`.
    """

    name: str
    module: nn.Module
    example_inputs: tuple[Tensor, ...]
    input_names: list[str]
    output_names: list[str]


@dataclass
class TestCase:
    """A frame, the starting noise, and the actions PyTorch computes for them.

    Args:
        frame (`dict[str, np.ndarray]`):
            The frame as the robot sends it: uint8 HWC images and float32 arrays.
        noise (`np.ndarray`, *optional*):
            The starting noise, for a policy that takes one.
        actions (`np.ndarray`):
            The float32 actions to play, in robot units, shaped `[steps, action_dim]`.
    """

    frame: dict[str, np.ndarray]
    noise: np.ndarray | None
    actions: np.ndarray


class ExportRecipe(abc.ABC):
    """Knows one kind of policy, never a compiler: turns it into parts and makes the test case.

    A policy gets its own recipe only when the default one, [`ActionChunkRecipe`], does not fit. It lives
    beside the policy's configuration, in `export_<type>.py`, as a class named like the policy class with
    `ExportRecipe` in place of `Policy`, such as `PI05ExportRecipe`.

    Args:
        policy (`PreTrainedPolicy`):
            The trained policy, in eval mode on the backend's device.
        preprocessor (`PolicyProcessorPipeline`):
            Its saved preprocessor.
        postprocessor (`PolicyProcessorPipeline`):
            Its saved postprocessor.
        cfg (`ExportConfig`):
            The `lerobot-export` configuration.

    **Attributes**:
        - **config** (`PreTrainedConfig`) -- The policy config the bundle saves, naming the frame it takes.
        - **task** (`str | None`) -- The task the parts hold, or `None` when the policy reads none.
        - **tolerance** (`float`) -- The largest allowed action difference on the test case, robot units.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        cfg: ExportConfig,
    ) -> None:
        """Keep the policy, its processors and the configuration."""
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.cfg = cfg
        self.config: PreTrainedConfig = policy.config
        self.task: str | None = None
        self.tolerance = cfg.tolerance if cfg.tolerance is not None else _default_tolerance(policy)

    @abc.abstractmethod
    def parts(self) -> Iterator[Part]:
        """Yield the parts in the order they run, one at a time, so that memory holds one."""

    @abc.abstractmethod
    def test_case(self) -> TestCase:
        """The frame, noise and actions the bundle's programs must reproduce."""


class ActionChunkRecipe(ExportRecipe):
    """The default recipe: one program that does what `lerobot-rollout` does for one action chunk.

    The program takes the robot's frame (uint8 HWC images, float32 state) and, for a policy that samples
    from noise, the starting noise. It converts the images, runs the saved preprocessor,
    `predict_action_chunk` trimmed to the actions rollout plays, and the saved postprocessor, and returns
    float32 actions in robot units. A program cannot tokenize text, so the task is tokenized once, at
    export, and the bundle runs only that task.

    It covers policies that keep LeRobot's contract for one chunk from the current frame,
    `PreTrainedPolicy.chunk_inference_spec`, and that compute the same chunk twice from the same frame and
    noise.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        cfg: ExportConfig,
    ) -> None:
        """Check that the policy can be exported this way, and make the test case."""
        super().__init__(policy, preprocessor, postprocessor, cfg)
        policy_type = policy.config.type
        try:
            self._steps = policy.chunk_inference_spec().execution_steps
        except ValueError as error:
            raise ValueError(f"This {policy_type} checkpoint cannot be exported: {error}") from error
        self._device = torch.device(cfg.backend.device)
        self._generator = torch.Generator().manual_seed(cfg.seed)
        self._metadata: LeRobotDatasetMetadata | None = None
        if cfg.dataset is not None:
            from lerobot.datasets import LeRobotDatasetMetadata

            dataset = cfg.dataset
            self._metadata = LeRobotDatasetMetadata(
                dataset.repo_id, root=dataset.root, revision=dataset.revision
            )
        self._frame_shapes = self._robot_frame_shapes()
        task = cfg.task or ("" if self._metadata is None else str(self._metadata.tasks.index[0]))
        frame = self._random_frame()
        if self._reads_task(frame, task):
            if not task:
                raise ValueError(
                    f"This {policy_type} policy reads the task: give the one to export with --task."
                )
            self.task = task
        self._noise_shape = self._sampling_noise_shape()
        noise = self._random_noise()
        actions = self._reference(frame, noise)
        if not np.array_equal(actions, self._reference(frame, noise)):
            raise ValueError(
                f"This {policy_type} policy computes another chunk from the same frame and noise each time, "
                "so a compiled copy cannot be checked against it."
            )
        self._case = TestCase(frame, noise, actions)

    def parts(self) -> Iterator[Part]:
        """The one part, after checking that it computes what PyTorch does for two different frames."""
        module = _ActionChunk(
            self.policy,
            self.preprocessor,
            self.postprocessor,
            list(self._frame_shapes),
            self.task,
            self._steps,
        )
        frame, noise = self._random_frame(), self._random_noise()
        for case in (self._case, TestCase(frame, noise, self._reference(frame, noise))):
            with torch.no_grad():
                actions = module(*self._inputs(case.frame, case.noise)).cpu().numpy()
            if not np.array_equal(actions, case.actions):
                raise ValueError(
                    f"The {self.policy.config.type} policy computes something else as one module than it "
                    "does in lerobot-rollout, for example because a preprocessor step reads the frame in "
                    "Python. It needs its own export recipe."
                )
        input_names = [*self._frame_shapes, *(["noise"] if self._noise_shape else [])]
        yield Part("model", module, self._inputs(self._case.frame, self._case.noise), input_names, [ACTION])

    def test_case(self) -> TestCase:
        """The random frame and noise, and the actions PyTorch computes for them as `lerobot-rollout` does."""
        return self._case

    def _robot_frame_shapes(self) -> dict[str, tuple[int, ...]]:
        """The shape of each array of the robot's frame: the policy's inputs, or the dataset's cameras."""
        config = self.policy.config
        if self._metadata is None:
            # The policy fills its empty camera slots itself, so the robot does not send them.
            empty = {f"{OBS_IMAGES}.empty_camera_{i}" for i in range(getattr(config, "empty_cameras", 0))}
            features = {name: f for name, f in (config.input_features or {}).items() if name not in empty}
        else:
            features = self._dataset_features(self._metadata)
        self.config = copy(config)
        self.config.input_features = features
        return {
            name: (f.shape[1], f.shape[2], f.shape[0]) if name.startswith(OBS_IMAGES) else tuple(f.shape)
            for name, f in features.items()
        }

    def _dataset_features(self, metadata: LeRobotDatasetMetadata) -> dict:
        """The dataset's observations that the policy reads, after the saved preprocessor's renames."""
        renames = {
            old: new
            for step in self.preprocessor.steps
            if isinstance(step, RenameObservationsProcessorStep)
            for old, new in step.rename_map.items()
        }
        features = dataset_to_policy_features(metadata.features)
        inputs = self.policy.config.input_features or {}
        return {
            name: f
            for name, f in features.items()
            if name.startswith(f"{OBS_STR}.") and renames.get(name, name) in inputs
        }

    def _random_frame(self) -> dict[str, np.ndarray]:
        frame = {}
        for name, shape in self._frame_shapes.items():
            if name.startswith(OBS_IMAGES):
                height, width, channels = shape
                coarse = torch.rand(1, channels, *IMAGE_CELLS, generator=self._generator) * 255
                image = nn.functional.interpolate(coarse, size=(height, width), mode="bilinear")
                frame[name] = image[0].permute(1, 2, 0).round().to(torch.uint8).contiguous().numpy()
            else:
                frame[name] = torch.rand(shape, generator=self._generator).numpy()
        return frame

    def _sampling_noise_shape(self) -> tuple[int, ...] | None:
        """The starting noise the policy samples from, or None for a policy that takes none.

        LeRobot's flow-matching policies draw it shaped `(chunk_size, max_action_dim)` for one sample.
        """
        if "noise" not in inspect.signature(self.policy.predict_action_chunk).parameters:
            return None
        config = self.policy.config
        max_action_dim = getattr(config, "max_action_dim", None)
        if max_action_dim is None:
            raise ValueError(
                f"The {config.type} policy takes a starting noise of a shape the default recipe does not "
                "know. It needs its own export recipe."
            )
        return (config.chunk_size, max_action_dim)

    def _random_noise(self) -> np.ndarray | None:
        if self._noise_shape is None:
            return None
        return torch.randn(self._noise_shape, generator=self._generator).numpy()

    def _observation(self, frame: dict[str, np.ndarray], task: str | None) -> dict:
        """The frame and task as `lerobot-rollout` hands them to the preprocessor."""
        return prepare_observation_for_inference(dict(frame), self._device, task)

    def _reads_task(self, frame: dict[str, np.ndarray], task: str) -> bool:
        # Not inference_mode: tensors the processors cache here end up in the program, and a compiler
        # cannot take inference tensors.
        with torch.no_grad():
            first = self.preprocessor(self._observation(frame, task))
            second = self.preprocessor(self._observation(frame, f"{task} then stop"))
        tensors = {name for name, value in first.items() if isinstance(value, Tensor)}
        return any(
            not isinstance(second.get(name), Tensor) or not torch.equal(first[name], second[name])
            for name in tensors
        )

    def _reference(self, frame: dict[str, np.ndarray], noise: np.ndarray | None) -> np.ndarray:
        """The actions `lerobot-rollout` plays for one chunk with the PyTorch policy."""
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        with torch.no_grad():
            batch = self.preprocessor(self._observation(frame, self.task))
            actions = self.policy.predict_action_chunk(batch, **self._noise_kwargs(noise))
            actions = self.postprocessor(actions[:, : self._steps])
        return actions[0].float().cpu().numpy()

    def _noise_kwargs(self, noise: np.ndarray | None) -> dict:
        return {} if noise is None else {"noise": torch.from_numpy(noise).to(self._device)[None]}

    def _inputs(self, frame: dict[str, np.ndarray], noise: np.ndarray | None) -> tuple[Tensor, ...]:
        arrays = [*frame.values(), *([] if noise is None else [noise])]
        return tuple(torch.from_numpy(x).to(self._device) for x in arrays)


class _ActionChunk(nn.Module):
    """The program of [`ActionChunkRecipe`]: the robot's frame, and the noise if any, in; the actions out."""

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        frame_names: list[str],
        task: str | None,
        steps: int,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.frame_names = frame_names
        self.task = task or ""
        self.steps = steps

    def forward(self, *inputs: Tensor) -> Tensor:
        observation = {
            name: _policy_input(name, x) for name, x in zip(self.frame_names, inputs, strict=False)
        }
        observation["task"] = self.task
        observation["robot_type"] = ""
        kwargs = {"noise": inputs[-1][None]} if len(inputs) > len(self.frame_names) else {}
        actions = self.policy.predict_action_chunk(self.preprocessor(observation), **kwargs)
        return self.postprocessor(actions[:, : self.steps])[0].float()


def _policy_input(name: str, x: Tensor) -> Tensor:
    """What `prepare_observation_for_inference` does to one array of the frame, inside the program."""
    if "image" in name:
        x = (x.float() / 255).permute(2, 0, 1)
    return x.unsqueeze(0)


def _default_tolerance(policy: PreTrainedPolicy) -> float:
    if any(p.dtype == torch.bfloat16 for p in policy.parameters()):
        return BFLOAT16_TOLERANCE
    return FLOAT32_TOLERANCE


def make_recipe(
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    cfg: ExportConfig,
) -> ExportRecipe:
    """The policy's own recipe, from `export_<type>.py` beside its configuration, or [`ActionChunkRecipe`]."""
    config_class = type(policy.config)
    module = _import_sibling_policy_module(config_class, "export")
    if module is None:
        return ActionChunkRecipe(policy, preprocessor, postprocessor, cfg)
    recipe_class = getattr(module, config_class.__name__.removesuffix("Config") + "ExportRecipe")
    return recipe_class(policy, preprocessor, postprocessor, cfg)
