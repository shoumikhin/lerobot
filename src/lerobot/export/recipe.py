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
import logging
from collections.abc import Iterator
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.overrides import TorchFunctionMode

from lerobot.configs import FeatureType, PreTrainedConfig
from lerobot.inference.bundle.manifest import NOISE
from lerobot.policies import PreTrainedPolicy, prepare_observation_for_inference
from lerobot.policies.factory import _import_sibling_policy_module
from lerobot.processor import PolicyProcessorPipeline, RenameObservationsProcessorStep
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STR
from lerobot.utils.feature_utils import dataset_to_policy_features

from .configs import ExportConfig

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDatasetMetadata

logger = logging.getLogger(__name__)

# In the policy's normalized action units (robot units divided by each action's spread in the training data),
# about twice the largest test case difference measured with correct compilers. Float32 bundles of ACT and
# pi0 stayed within 8.3e-6 of eager PyTorch on every frame tried, about what eager on a CPU and on a GPU
# differ by. Bfloat16 keeps 3 significant digits: GR00T and SmolVLA bundles differed from their test case by
# up to 0.036, and eager PyTorch with another attention kernel moves their actions by up to 0.05.
FLOAT32_TOLERANCE = 2e-5
BFLOAT16_TOLERANCE = 0.075
# A random test image blends a coarse grid of random colors (rows, columns), so it changes smoothly like a
# camera image. On independent random pixels a vision-language policy's bfloat16 actions differ more between
# two correct compilations, so the check would measure noise.
IMAGE_CELLS = (6, 8)
# Two tasks whose first words differ, so they stay different when a tokenizer cuts them short.
PROBE_TASKS = ("pick up the red cube", "stop")
_RANDOM_NORMAL = {torch.randn, torch.randn_like, torch.normal, Tensor.normal_}


@dataclass
class Part:
    """One program of a policy.

    Args:
        name (`str`):
            The program's name, such as `model`.
        module (`nn.Module`):
            What the program computes, on the backend's device.
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
        action_scale (`np.ndarray`):
            For each action component, the robot units of one step of the policy's normalized actions.
    """

    frame: dict[str, np.ndarray]
    noise: np.ndarray | None
    actions: np.ndarray
    action_scale: np.ndarray


@dataclass
class ProgramCall:
    """One run of a compiled part, with the names its inputs and outputs take in this run.

    Args:
        part (`str`):
            The name of the part to run.
        inputs (`list[str]`):
            The names of the values it reads, in the order of the part's inputs.
        outputs (`list[str]`):
            The names of the values it writes, in the order of the part's outputs.
    """

    part: str
    inputs: list[str]
    outputs: list[str]


@dataclass
class TaskPrompt:
    """How a policy's task becomes the `task_embeddings` and `task_length` its parts take.

    The bundle runs any task this way, without PyTorch: see `lerobot.inference.bundle.TaskTokensSpec`.

    Args:
        text (`list[list[str]]`):
            The ops that make the prompt text from the task, as `TaskTokensSpec.text` describes.
        tokenizer (`Path`):
            The policy's tokenizer file, in the format of the `tokenizers` library, copied as it is.
        table (`Tensor`):
            The token embedding table, `[vocabulary, width]`, in float32 or bfloat16.
        known (`dict[str, list[int]]`):
            The token ids of the prompt of each task known at export, such as the dataset's tasks.
        max_tokens (`int`):
            The most tokens the parts take.
    """

    text: list[list[str]]
    tokenizer: Path
    table: Tensor
    known: dict[str, list[int]]
    max_tokens: int


@dataclass
class RunPlan:
    """How the compiled parts of a bundle run, for a recipe whose parts do not simply run once in order.

    Args:
        calls (`list[ProgramCall]`, *optional*):
            The runs in order. A part may run several times, for example a denoising step. `None` runs each
            part once, in the order it was yielded, with its own names.
        constants (`dict[str, np.ndarray]`):
            Named arrays the runs can read, such as each step's timestep.
        task_prompt (`TaskPrompt`, *optional*):
            How the task becomes the `task_embeddings` and `task_length` the parts take, for a policy that
            reads it.
    """

    calls: list[ProgramCall] | None = None
    constants: dict[str, np.ndarray] = field(default_factory=dict)
    task_prompt: TaskPrompt | None = None


class ExportRecipe(abc.ABC):
    """Knows one kind of policy, never a compiler: turns it into parts and makes the test case.

    A policy gets its own recipe only when the default one, [`ActionChunkRecipe`], does not fit. It lives
    beside the policy's configuration, in `export_<type>.py`, as a class named like the configuration class
    with `ExportRecipe` in place of `Config`, such as `PI05ExportRecipe` for `PI05Config`.

    Args:
        policy (`PreTrainedPolicy`):
            The trained policy, in eval mode on the CPU. The recipe moves what it compiles to the backend's
            device.
        preprocessor (`PolicyProcessorPipeline`):
            Its saved preprocessor, set to run on the backend's device.
        postprocessor (`PolicyProcessorPipeline`):
            Its saved postprocessor.
        cfg (`ExportConfig`):
            The `lerobot-export` configuration.

    **Attributes**:
        - **config** (`PreTrainedConfig`) -- The policy config the bundle records, naming the frame it takes.
        - **task** (`str | None`) -- The task the parts hold, or `None` when the policy reads none.
        - **tolerance** (`float`) -- The largest allowed action difference on the test case, in normalized
          action units.
    """

    @classmethod
    def load(cls, cfg: ExportConfig) -> ExportRecipe:
        """The recipe for the checkpoint `cfg.policy` names.

        By default it loads the whole policy. A recipe that builds its parts one at a time, to export on a
        small board, overrides this to load only what each part needs.
        """
        return cls(*load_policy(cfg), cfg)

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
        """Yield the parts in the order they run. `lerobot-export` traces and frees each before the next."""

    @abc.abstractmethod
    def test_case(self) -> TestCase:
        """The frame, noise and actions the bundle's programs must reproduce."""

    def plan(self) -> RunPlan:
        """How the compiled parts run, asked once `parts` is used up. By default each part runs once."""
        return RunPlan()


class ActionChunkRecipe(ExportRecipe):
    """The default recipe: one program that does what `lerobot-rollout` does for one action chunk.

    The program takes the robot's frame (uint8 HWC images, float32 state) and, for a policy that samples
    from noise, the starting noise. It converts the images, runs the saved preprocessor,
    `predict_action_chunk` trimmed to the actions rollout plays, and the saved postprocessor, and returns
    float32 actions in robot units. A program cannot tokenize text, so the task is tokenized once, at
    export, and the bundle runs only that task.

    It covers policies that keep LeRobot's contract for one chunk from the current frame,
    `PreTrainedPolicy.chunk_inference_spec`, draw at most one starting noise, and compute the same chunk
    twice from the same frame and noise.
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
        policy.to(self._device)
        self._generator = torch.Generator().manual_seed(cfg.seed)
        self._metadata: LeRobotDatasetMetadata | None = None
        if cfg.dataset is not None:
            from lerobot.datasets import LeRobotDatasetMetadata

            dataset = cfg.dataset
            self._metadata = LeRobotDatasetMetadata(
                dataset.repo_id, root=dataset.root, revision=dataset.revision
            )
        self._frame_shapes = self._robot_frame_shapes()
        frame = self._random_frame()
        self.task = self._task(frame)
        self._noise_shape = self._sampled_noise_shape(frame)
        noise = self._random_noise()
        actions = self._reference(frame, noise)
        action_scale = self._action_scale()
        if not np.array_equal(actions, self._reference(frame, noise)):
            raise ValueError(
                f"This {policy_type} policy computes another chunk from the same frame and noise each time, "
                "so a compiled copy cannot be checked against it."
            )
        self._case = TestCase(frame, noise, actions, action_scale)

    def parts(self) -> Iterator[Part]:
        """The one part, after checking that it computes what PyTorch does for two different frames."""
        module = _ActionChunk(
            self.policy,
            self.preprocessor,
            self.postprocessor,
            list(self._frame_shapes),
            self._visual,
            self.task,
            self._steps,
        )
        frame, noise = self._random_frame(), self._random_noise()
        for case_frame, case_noise, expected in (
            (self._case.frame, self._case.noise, self._case.actions),
            (frame, noise, self._reference(frame, noise)),
        ):
            with torch.no_grad():
                actions = module(*self._inputs(case_frame, case_noise)).cpu().numpy()
            if not np.array_equal(actions, expected):
                raise ValueError(
                    f"The {self.policy.config.type} policy computes something else as one module than it "
                    "does in lerobot-rollout, for example because a preprocessor step reads the frame in "
                    "Python. It needs its own export recipe."
                )
        input_names = [*self._frame_shapes, *([NOISE] if self._noise_shape else [])]
        yield Part("model", module, self._inputs(self._case.frame, self._case.noise), input_names, [ACTION])

    def test_case(self) -> TestCase:
        """The random frame and noise, and the actions PyTorch computes for them as `lerobot-rollout` does."""
        return self._case

    def _robot_frame_shapes(self) -> dict[str, tuple[int, ...]]:
        """The shape of each array of the robot's frame: the policy's inputs, or the dataset's cameras."""
        config = self.policy.config
        # The policy fills its empty camera slots itself, so the robot does not send them.
        empty = {f"{OBS_IMAGES}.empty_camera_{i}" for i in range(getattr(config, "empty_cameras", 0))}
        inputs = {name: f for name, f in (config.input_features or {}).items() if name not in empty}
        features = inputs if self._metadata is None else self._dataset_features(self._metadata, inputs)
        self.config = copy(config)
        self.config.input_features = features
        self._visual = {name for name, f in features.items() if f.type is FeatureType.VISUAL}
        return {
            name: (f.shape[1], f.shape[2], f.shape[0]) if name in self._visual else tuple(f.shape)
            for name, f in features.items()
        }

    def _dataset_features(self, metadata: LeRobotDatasetMetadata, inputs: dict) -> dict:
        """The dataset's observations that the policy reads, after the saved preprocessor's renames."""
        renames = {
            old: new
            for step in self.preprocessor.steps
            if isinstance(step, RenameObservationsProcessorStep)
            for old, new in step.rename_map.items()
        }
        features = dataset_to_policy_features(metadata.features)
        kept = {
            name: f
            for name, f in features.items()
            if name.startswith(f"{OBS_STR}.") and renames.get(name, name) in inputs
        }
        missing = sorted(set(inputs) - {renames.get(name, name) for name in kept})
        if missing:
            raise ValueError(
                f"The dataset gives nothing the policy reads as {missing}, even after the renames its "
                f"preprocessor saved ({renames}). The dataset has {sorted(features)}."
            )
        return kept

    def _task(self, frame: dict[str, np.ndarray]) -> str | None:
        """The task the program holds: the one given, or the dataset's first for a policy that reads text."""
        if self.cfg.task:
            return self.cfg.task
        if not self._reads_task(frame):
            return None
        if self._metadata is None:
            raise ValueError(
                f"This {self.policy.config.type} policy reads the task: give the one to export with --task."
            )
        tasks = [str(task) for task in self._metadata.tasks.index]
        if len(tasks) > 1:
            logger.warning(
                "No --task given: exporting the dataset's first task, %r, of %d.", tasks[0], len(tasks)
            )
        return tasks[0]

    def _reads_task(self, frame: dict[str, np.ndarray]) -> bool:
        # Not inference_mode: tensors the processors cache here end up in the program, and a compiler
        # cannot take inference tensors.
        with torch.no_grad():
            first, second = (self.preprocessor(self._observation(frame, task)) for task in PROBE_TASKS)
        return any(
            not isinstance(second.get(name), Tensor) or not torch.equal(value, second[name])
            for name, value in first.items()
            if isinstance(value, Tensor)
        )

    def _random_frame(self) -> dict[str, np.ndarray]:
        frame = {}
        for name, shape in self._frame_shapes.items():
            if name in self._visual:
                height, width, channels = shape
                coarse = torch.rand(1, channels, *IMAGE_CELLS, generator=self._generator) * 255
                image = nn.functional.interpolate(coarse, size=(height, width), mode="bilinear")
                frame[name] = image[0].permute(1, 2, 0).round().to(torch.uint8).contiguous().numpy()
            else:
                frame[name] = torch.rand(shape, generator=self._generator).numpy()
        return frame

    def _sampled_noise_shape(self, frame: dict[str, np.ndarray]) -> tuple[int, ...] | None:
        """The shape of the one starting noise the policy draws for a chunk, or `None` if it takes none.

        The policy is run once without a noise and its random normal draws are recorded, so the shape is
        the one the policy itself uses.
        """
        if NOISE not in inspect.signature(self.policy.predict_action_chunk).parameters:
            return None
        shapes: list[tuple[int, ...]] = []
        self._reset()
        with torch.no_grad(), _RecordNormalDraws(shapes):
            self.policy.predict_action_chunk(self.preprocessor(self._observation(frame, self.task)))
        if len(shapes) != 1 or shapes[0][0] != 1:
            raise ValueError(
                f"The {self.policy.config.type} policy draws {len(shapes)} random tensors shaped {shapes} for "
                "one chunk, not one starting noise for one sample. It needs its own export recipe."
            )
        return shapes[0][1:]

    def _random_noise(self) -> np.ndarray | None:
        if self._noise_shape is None:
            return None
        return torch.randn(self._noise_shape, generator=self._generator).numpy()

    def _observation(self, frame: dict[str, np.ndarray], task: str | None) -> dict:
        """The frame and task as `lerobot-rollout` hands them to the preprocessor."""
        return prepare_observation_for_inference(dict(frame), self._device, task)

    def _reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def _reference(self, frame: dict[str, np.ndarray], noise: np.ndarray | None) -> np.ndarray:
        """The actions `lerobot-rollout` plays for one chunk with the PyTorch policy."""
        self._reset()
        kwargs = {} if noise is None else {NOISE: torch.from_numpy(noise).to(self._device)[None]}
        with torch.no_grad():
            batch = self.preprocessor(self._observation(frame, self.task))
            predicted = self.policy.predict_action_chunk(batch, **kwargs)[:, : self._steps]
            self._model_width = predicted.shape[-1]
            actions = self.postprocessor(predicted)
        return actions[0].float().cpu().numpy()

    def _action_scale(self) -> np.ndarray:
        """Robot units per normalized unit for each action, measured on the postprocessor of the last chunk."""
        with torch.no_grad():
            low, high = (
                self.postprocessor(torch.full((1, 1, self._model_width), value, device=self._device))[0, 0]
                .float()
                .cpu()
                .numpy()
                for value in (0.0, 1.0)
            )
        # A joint that never moved in the training data has almost no spread, and float32 rounding of its
        # value would then read as a large difference.
        return np.maximum(np.abs(high - low), 1e-4 * np.abs(low) + 1e-6).astype(np.float32)

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
        visual: set[str],
        task: str | None,
        steps: int,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.frame_names = frame_names
        self.visual = visual
        self.task = task or ""
        self.steps = steps

    def forward(self, *inputs: Tensor) -> Tensor:
        # What prepare_observation_for_inference does to each array of the frame, as tensor operations.
        observation: dict[str, Any] = {
            name: ((x.float() / 255).permute(2, 0, 1).contiguous() if name in self.visual else x).unsqueeze(0)
            for name, x in zip(self.frame_names, inputs, strict=False)
        }
        observation["task"] = self.task
        observation["robot_type"] = ""
        kwargs = {NOISE: inputs[-1][None]} if len(inputs) > len(self.frame_names) else {}
        actions = self.policy.predict_action_chunk(self.preprocessor(observation), **kwargs)
        return self.postprocessor(actions[:, : self.steps])[0].float()


class _RecordNormalDraws(TorchFunctionMode):
    """Records the shape of every random normal tensor drawn while it is active."""

    def __init__(self, shapes: list[tuple[int, ...]]) -> None:
        super().__init__()
        self.shapes = shapes

    def __torch_function__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func in _RANDOM_NORMAL:
            self.shapes.append(tuple(result.shape))
        return result


def _default_tolerance(policy: PreTrainedPolicy) -> float:
    if any(p.dtype == torch.bfloat16 for p in policy.parameters()):
        return BFLOAT16_TOLERANCE
    return FLOAT32_TOLERANCE


def recipe_class(config: PreTrainedConfig) -> type[ExportRecipe]:
    """The policy's own recipe, from `export_<type>.py` beside its configuration, or [`ActionChunkRecipe`]."""
    config_class = type(config)
    module = _import_sibling_policy_module(config_class, "export")
    if module is None:
        return ActionChunkRecipe
    name = config_class.__name__.removesuffix("Config") + "ExportRecipe"
    found = getattr(module, name, None)
    if found is None:
        raise ValueError(f"{module.__name__} must define the export recipe {name}.")
    return found


def load_policy(
    cfg: ExportConfig,
) -> tuple[PreTrainedPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """The whole trained policy, in eval mode on the CPU, and its processors set to the backend's device."""
    from lerobot.policies.factory import load_pretrained_policy, make_pre_post_processors

    config = cast(PreTrainedConfig, cfg.policy)
    # Weights load on the CPU; the recipe moves what it compiles to the backend's device.
    config.device = "cpu"
    # How the policy runs, not how it trained, decides what compiles it.
    if hasattr(config, "compile_model"):
        config.compile_model = False
    policy = load_pretrained_policy(config)
    if hasattr(policy, "merge_and_unload"):
        # A PEFT adapter: compile the base weights with the adapter folded in.
        policy = policy.merge_and_unload()
    policy.eval()
    config.device = cfg.backend.device
    overrides = {"device_processor": {"device": config.device}}
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=config.pretrained_path,
        pretrained_revision=config.pretrained_revision,
        preprocessor_overrides=overrides,
        postprocessor_overrides=overrides,
    )
    return policy, preprocessor, postprocessor
