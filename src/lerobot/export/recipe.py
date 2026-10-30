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
from collections.abc import Callable, Iterable, Iterator
from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.overrides import TorchFunctionMode

from lerobot.bundle.manifest import NOISE, ProgramRun
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig, is_depth_map
from lerobot.policies import PreTrainedPolicy, prepare_observation_for_inference
from lerobot.processor import (
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    policy_output_to_transition,
    transition_to_policy_action,
)
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
_IMAGE_CELLS = (6, 8)
# Two tasks whose first words differ, so they stay different when a tokenizer cuts them short.
_PROBE_TASKS = ("pick up the red cube", "stop")
_RANDOM_NORMAL = {torch.randn, torch.randn_like, torch.normal, Tensor.normal_}
_MULTINOMIAL = {torch.multinomial, Tensor.multinomial}


@dataclass
class Part:
    """One program of a policy.

    Args:
        name (`str`):
            The program's name, such as `model`. Not `noise`, the name of the noise draw on CUDA.
        module (`nn.Module`):
            What the program computes, on the backend's device.
        example_inputs (`tuple[Tensor, ...]`):
            Inputs to trace `module` with, on the backend's device.
        inputs (`list[str]`):
            The names of the inputs: keys of the robot's frame, `noise`, or outputs of earlier parts.
        outputs (`list[str]`):
            The names of the outputs. The last part outputs `action`.
    """

    name: str
    module: nn.Module
    example_inputs: tuple[Tensor, ...]
    inputs: list[str]
    outputs: list[str]

    def __post_init__(self) -> None:
        """Refuse a reserved name, inputs without one name each or one tensor given for two.

        Keeps the example inputs contiguous.
        """
        if self.name == NOISE:
            raise ValueError(
                f"A part cannot be named {NOISE}: on CUDA, a bundle draws its noise with a program so named."
            )
        if len(self.inputs) != len(self.example_inputs):
            raise ValueError(
                f"The part {self.name} has a different number of input names ({len(self.inputs)}) and "
                f"example inputs ({len(self.example_inputs)}): give each input one name."
            )
        # torch.export traces two inputs given the same tensor as one, so the program reads the last for both.
        names: dict[int, str] = {}
        for name, x in zip(self.inputs, self.example_inputs, strict=True):
            if id(x) in names:
                raise ValueError(
                    f"The part {self.name} gives one tensor for its inputs {names[id(x)]} and {name}: give each "
                    "its own, or the traced program reads one of them for both."
                )
            names[id(x)] = name
        # A traced program keeps its example inputs' memory layout, and a bundle passes every input contiguous.
        self.example_inputs = tuple(x.contiguous() for x in self.example_inputs)


@dataclass
class TestCase:
    """A frame, the starting noise, and the actions PyTorch computes for them.

    Args:
        frame (`dict[str, np.ndarray]`):
            The frame as the robot sends it: uint8 HWC images and float32 arrays.
        noise (`np.ndarray`, *optional*):
            The starting noise, for a policy that takes one.
        actions (`np.ndarray`):
            The float32 actions to play, in robot units, shaped `[execution_steps, action_dim]`.
        action_scale (`np.ndarray`):
            For each action component, the robot units of one step of the policy's normalized actions.
    """

    frame: dict[str, np.ndarray]
    noise: np.ndarray | None
    actions: np.ndarray
    action_scale: np.ndarray


@dataclass
class RunPlan:
    """How the compiled parts of a bundle run, for a recipe whose parts do not simply run once in order.

    No run is named `noise` or runs a program so named: on CUDA, that is the noise draw.

    Args:
        runs (`list[ProgramRun]`, *optional*):
            The runs in order, each naming the part it runs as its program. A part may run several times, for
            example a denoising step. `None` runs each part once, in the order it was yielded, with its own
            names.
        constants (`dict[str, np.ndarray]`):
            Named arrays the runs can read, such as each step's timestep.
    """

    runs: list[ProgramRun] | None = None
    constants: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a run named like the one that draws a bundle's noise on CUDA, or a run of its program."""
        if any(NOISE in (run.name, run.program) for run in self.runs or []):
            raise ValueError(
                f"A run cannot be named {NOISE} or run a program so named: on CUDA, a bundle draws its noise "
                "with a run and a program so named."
            )


class ExportRecipe(abc.ABC):
    """Knows one kind of policy, never a compiler: turns it into parts and makes the test case.

    A policy gets its own recipe only when the default one, [`ActionChunkRecipe`], does not fit. It lives
    beside the policy's configuration, in `export_<type>.py`, as a class named like the configuration class
    with `ExportRecipe` in place of `Config`, such as `PI05ExportRecipe` for `PI05Config`.

    Args:
        policy (`PreTrainedPolicy`):
            The trained policy, in eval mode on the backend's device.
        preprocessor (`PolicyProcessorPipeline`):
            Its saved preprocessor, set to run on the backend's device.
        postprocessor (`PolicyProcessorPipeline`):
            Its saved postprocessor.
        cfg (`ExportConfig`):
            The `lerobot-export` configuration.

    Attributes:
        config (`PreTrainedConfig`):
            The policy config the bundle records, naming the frame it takes.
        task (`str`, *optional*):
            The task the parts hold, or `None` when none was given and the policy reads none.
        tolerance (`float`):
            The largest allowed action difference on the test case, in normalized action units.
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
    `PreTrainedPolicy.chunk_inference_spec`, draw a starting noise they take in exactly one random normal
    draw and compute the same chunk when given the standard normal draw behind it, and compute the same chunk
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
        self._metadata = dataset_metadata(cfg)
        self._features = robot_features(policy.config, preprocessor, self._metadata)
        self._visual = {name for name, f in self._features.items() if f.type is FeatureType.VISUAL}
        self.config = copy(policy.config)
        self.config.input_features = self._features
        frame = random_frame(self._features, self._generator)
        self.task = self._task(frame)
        self._noise_shape = self._sampled_noise_shape(frame)
        noise = self._random_noise()
        actions = self._reference(frame, noise)
        width = actions.shape[-1]
        action_scale = measure_action_scale(
            lambda value: self.postprocessor(torch.full((1, 1, width), value, device=self._device))[0, 0]
        )
        if not np.array_equal(actions, self._reference(frame, noise)):
            raise ValueError(
                f"This {policy_type} policy computes another chunk from the same frame and noise each time, "
                "so a compiled copy cannot be checked against it."
            )
        self._case = TestCase(frame, noise, actions, action_scale)

    def parts(self) -> Iterator[Part]:
        """The one part, after checking that it computes what PyTorch does for two different frames."""
        module = _ActionChunk(
            self.predict_chunk,
            self.policy,
            self.preprocessor,
            self.postprocessor,
            list(self._features),
            self._visual,
            self.task,
            self._steps,
        )
        frame, noise = random_frame(self._features, self._generator), self._random_noise()
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
        inputs = [*self._features, *([NOISE] if self._noise_shape else [])]
        yield Part("model", module, self._inputs(self._case.frame, self._case.noise), inputs, [ACTION])

    def test_case(self) -> TestCase:
        """The random frame and noise, and the actions PyTorch computes for them as `lerobot-rollout` does."""
        return self._case

    def predict_chunk(self, policy: PreTrainedPolicy, batch: dict[str, Any], noise: Tensor | None) -> Tensor:
        """The `[1, chunk, action]` chunk the policy predicts for a preprocessed batch, as rollout gets it.

        Without `noise` the policy draws its own. A recipe for a policy that predicts its actions another way
        overrides this. The program and the test case both call it, so they compute the same thing.
        """
        output = policy.predict_action_chunk(batch, **({} if noise is None else {NOISE: noise}))
        return transition_to_policy_action(policy_output_to_transition(output))

    def _task(self, frame: dict[str, np.ndarray]) -> str | None:
        """The task the program holds: the one given, or the dataset's first for a policy that reads text.

        `_reads_task` sees only the tensors the preprocessor makes. A policy that reads the task as text
        in its own code overrides it in its recipe.
        """
        if not self.cfg.task and not self._reads_task(frame):
            return None
        return export_task(self.cfg, self._metadata, self.policy.config.type)

    def _reads_task(self, frame: dict[str, np.ndarray]) -> bool:
        with torch.no_grad():
            first, second = (self.preprocessor(self._observation(frame, task)) for task in _PROBE_TASKS)
        return any(
            not isinstance(second.get(name), Tensor) or not torch.equal(value, second[name])
            for name, value in first.items()
            if isinstance(value, Tensor)
        )

    def _sampled_noise_shape(self, frame: dict[str, np.ndarray]) -> tuple[int, ...] | None:
        """The shape of the one starting noise the policy draws for a chunk, or `None` if it takes none.

        The policy is run once without a noise and its random normal draws are recorded, so the shape is the
        one the policy itself uses. It is then run on one standard normal noise twice, once drawing it and
        once given it, which must compute the same chunk.
        """
        if NOISE not in inspect.signature(self.policy.predict_action_chunk).parameters:
            return None
        self._reset()
        with torch.no_grad(), ReplayDraws() as draws:
            self.predict_chunk(self.policy, self.preprocessor(self._observation(frame, self.task)), None)
        shapes = draws.shapes
        # A noise with no axis besides the sample's would be one number, which a bundle's arrays cannot hold.
        if len(shapes) != 1 or len(shapes[0]) < 2 or shapes[0][0] != 1:
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
        noise_tensor = None if noise is None else torch.from_numpy(noise).to(self._device)[None]
        with torch.no_grad():
            batch = self.preprocessor(self._observation(frame, self.task))
            predicted = self.predict_chunk(self.policy, batch, noise_tensor)[:, : self._steps]
            actions = self.postprocessor(predicted)
        return actions[0].float().cpu().numpy()

    def _inputs(self, frame: dict[str, np.ndarray], noise: np.ndarray | None) -> tuple[Tensor, ...]:
        arrays = [*frame.values(), *([] if noise is None else [noise])]
        return tuple(torch.from_numpy(x).to(self._device) for x in arrays)


class _ActionChunk(nn.Module):
    """The program of [`ActionChunkRecipe`]: the robot's frame, and the noise if any, in; the actions out."""

    def __init__(
        self,
        predict_chunk: Callable[[PreTrainedPolicy, dict[str, Any], Tensor | None], Tensor],
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        frame_names: list[str],
        visual: set[str],
        task: str | None,
        execution_steps: int,
    ) -> None:
        super().__init__()
        self.predict_chunk = predict_chunk
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.frame_names = frame_names
        self.visual = visual
        self.task = task or ""
        self.execution_steps = execution_steps

    def forward(self, *inputs: Tensor) -> Tensor:
        observation: dict[str, Any] = {
            name: to_policy_input(x, name in self.visual)
            for name, x in zip(self.frame_names, inputs, strict=False)
        }
        observation["task"] = self.task
        observation["robot_type"] = ""
        noise = inputs[-1][None] if len(inputs) > len(self.frame_names) else None
        actions = self.predict_chunk(self.policy, self.preprocessor(observation), noise)
        return self.postprocessor(actions[:, : self.execution_steps])[0].float()


class ReplayDraws(TorchFunctionMode):
    """Makes the random draws of what runs while it is active read a given noise, one draw after another.

    A compiled program cannot draw random numbers the way PyTorch does, so a policy's randomness becomes the
    program's `noise` input, which the bundle draws from a standard normal. The k-th draw reads `noise[k]`:

    - a random normal tensor is `noise[k]` itself, or `mean + std * noise[k]` for `torch.normal(mean, std)` and
      `Tensor.normal_(mean, std)`, which also writes it into its tensor;
    - a multinomial draw of one category per row is made the way PyTorch makes it, `argmax(p / q)` with `q`
      exponentially distributed, and takes `q = (a**2 + b**2) / 2` from the two halves of `noise[k]`, shaped
      `[2, *p.shape]`.

    Without a noise, the draws are made as usual and `shapes` records the shape each would read.

    Args:
        noise (`Tensor`, *optional*):
            The draws, stacked on the first axis.
    """

    def __init__(self, noise: Tensor | None = None) -> None:
        """Read the draws from `noise`, or make them as usual and record their shapes without one."""
        super().__init__()
        self.noise = noise
        self.shapes: list[tuple[int, ...]] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Run `func`, reading it from the noise if it is a random draw."""
        kwargs = kwargs or {}
        if func is Tensor.normal_ and self.noise is not None:
            # Not drawn first: writing into a drawn tensor keeps a random operation ExecuTorch cannot lower.
            tensor = args[0]
            mean, std = self._mean_and_std(args[1:], kwargs)
            return tensor.copy_(mean + std * self._next(tuple(tensor.shape)).to(tensor))
        if func in _MULTINOMIAL:
            probs = args[0]
            if (args[1] if len(args) > 1 else kwargs.get("num_samples")) != 1:
                raise ValueError("Only a multinomial draw of one category per row can read the noise.")
            if self.noise is None:
                self.shapes.append((2, *probs.shape))
                return func(*args, **kwargs)
            first, second = self._next((2, *probs.shape))
            return torch.argmax(probs / ((first * first + second * second) / 2), dim=-1, keepdim=True)
        result = func(*args, **kwargs)
        if func in _RANDOM_NORMAL:
            if self.noise is None:
                self.shapes.append(tuple(result.shape))
            else:
                result = self._next(tuple(result.shape)).to(result)
                if func is torch.normal:
                    mean, std = self._mean_and_std(args, kwargs)
                    result = mean + std * result
        return result

    @staticmethod
    def _mean_and_std(args: tuple, kwargs: dict[str, Any]) -> tuple[Any, Any]:
        """The `mean` and `std` of `torch.normal(*args, **kwargs)` or of `tensor.normal_(*args, **kwargs)`."""
        given = {"mean": 0.0, "std": 1.0, **dict(zip(("mean", "std"), args, strict=False)), **kwargs}
        return given["mean"], given["std"]

    def _next(self, shape: tuple[int, ...]) -> Tensor:
        drawn = len(self.shapes)
        if self.noise is None or drawn == len(self.noise) or tuple(self.noise.shape[1:]) != shape:
            raise ValueError(
                f"Draw {drawn} reads noise shaped {list(shape)}, which the noise "
                f"{None if self.noise is None else list(self.noise.shape)} does not hold."
            )
        self.shapes.append(shape)
        return self.noise[drawn]


def _default_tolerance(policy: PreTrainedPolicy) -> float:
    if any(p.dtype == torch.bfloat16 for p in policy.parameters()):
        return BFLOAT16_TOLERANCE
    return FLOAT32_TOLERANCE


def preprocessor_renames(preprocessor: PolicyProcessorPipeline) -> dict[str, str]:
    """The observation renames the saved preprocessor makes, old name to the name its last rename gives."""
    renames: dict[str, str] = {}
    for step in preprocessor.steps:
        if isinstance(step, RenameObservationsProcessorStep):
            # A later rename reads the names the earlier ones made, as the pipeline runs them.
            renames = {old: step.rename_map.get(new, new) for old, new in renames.items()} | {
                old: new for old, new in step.rename_map.items() if old not in renames
            }
    return renames


def robot_features(
    config: PreTrainedConfig,
    preprocessor: PolicyProcessorPipeline,
    metadata: LeRobotDatasetMetadata | None = None,
) -> dict[str, PolicyFeature]:
    """The arrays of the robot's frame: the policy's inputs, or the dataset's observations that become them."""
    # The policy fills its empty camera slots itself, so the robot does not send them.
    empty = {f"{OBS_IMAGES}.empty_camera_{i}" for i in range(getattr(config, "empty_cameras", 0))}
    inputs = {name: f for name, f in (config.input_features or {}).items() if name not in empty}
    if metadata is None:
        return inputs
    renames = preprocessor_renames(preprocessor)
    features = dataset_to_policy_features(metadata.features)
    kept = {
        name: f
        for name, f in features.items()
        if name.startswith(f"{OBS_STR}.") and renames.get(name, name) in inputs
    }
    missing = sorted(set(inputs) - {renames.get(name, name) for name in kept})
    if missing:
        raise ValueError(
            f"The dataset gives nothing the policy reads as {missing}, even after its preprocessor's renames "
            f"({renames}). The dataset has {sorted(features)}."
        )
    refuse_depth_maps(kept, metadata)
    return kept


def refuse_depth_maps(names: Iterable[str], metadata: LeRobotDatasetMetadata) -> None:
    """Refuse a frame whose arrays `names` include a camera the dataset marks as a depth map."""
    depth = sorted(name for name in names if is_depth_map(metadata.features[name]))
    if depth:
        raise ValueError(
            f"The policy reads the depth maps {depth}, which an export cannot take: its programs read "
            "every image as 8-bit and divide it by 255, while a robot sends depth as 16-bit millimeters."
        )


def dataset_metadata(cfg: ExportConfig) -> LeRobotDatasetMetadata | None:
    """The metadata of the dataset `--dataset.repo_id` names, if any."""
    if cfg.dataset is None:
        return None
    from lerobot.datasets.storage import is_bucket_root, load_dataset_metadata

    dataset = cfg.dataset
    # As training loads it, so a dataset in a storage bucket or under a remote root is found too.
    root, repo_type = dataset.root, dataset.repo_type
    if dataset.streaming and is_bucket_root(dataset.repo_id, root):
        root, repo_type = None, "bucket"
    return load_dataset_metadata(
        dataset.repo_id,
        root=root,
        revision=dataset.revision,
        repo_type=cast(Literal["dataset", "bucket"], repo_type),
    )


def export_task(cfg: ExportConfig, metadata: LeRobotDatasetMetadata | None, policy_type: str) -> str:
    """The task to export for a policy that reads one: the one given, or the dataset's first."""
    if cfg.task:
        return cfg.task
    if metadata is None:
        raise ValueError(
            f"This {policy_type} policy reads the task: give the one to export with --task, or its dataset "
            "with --dataset.repo_id."
        )
    # A dataset without episodes has no tasks.
    tasks = [] if metadata.tasks is None else [str(task) for task in metadata.tasks.index]
    if not tasks:
        raise ValueError(
            f"This {policy_type} policy reads the task, and its dataset has none: give the one to export "
            "with --task."
        )
    if len(tasks) > 1:
        logger.warning(
            "No --task given: exporting the dataset's first task, %r, of %d.", tasks[0], len(tasks)
        )
    return tasks[0]


def frame_shapes(features: dict[str, PolicyFeature]) -> dict[str, tuple[int, ...]]:
    """The shape of each array of the robot's frame: height, width and channels for an image.

    An image feature is channels first, except in the checkpoints that record it channels last, as its shape
    shows.
    """
    return {
        name: _image_shape(f.shape) if f.type is FeatureType.VISUAL else tuple(f.shape)
        for name, f in features.items()
    }


def _image_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if shape[-1] in (1, 3) and shape[0] not in (1, 3):
        return tuple(shape)
    channels, height, width = shape
    return height, width, channels


def random_frame(features: dict[str, PolicyFeature], generator: torch.Generator) -> dict[str, np.ndarray]:
    """A random frame as the robot sends it: uint8 HWC images and float32 arrays."""
    frame = {}
    for name, shape in frame_shapes(features).items():
        if features[name].type is FeatureType.VISUAL:
            height, width, channels = shape
            coarse = torch.rand(1, channels, *_IMAGE_CELLS, generator=generator) * 255
            image = nn.functional.interpolate(coarse, size=(height, width), mode="bilinear")
            frame[name] = image[0].permute(1, 2, 0).round().to(torch.uint8).contiguous().numpy()
        else:
            frame[name] = torch.rand(shape, generator=generator).numpy()
    return frame


def to_policy_input(x: Tensor, visual: bool) -> Tensor:
    """What `prepare_observation_for_inference` does to one array of the robot's frame, as tensor operations."""
    return ((x.float() / 255).permute(2, 0, 1).contiguous() if visual else x).unsqueeze(0)


def measure_action_scale(actions_at: Callable[[float], Tensor]) -> np.ndarray:
    """Robot units per normalized unit for each action.

    `actions_at(value)` gives the robot actions for normalized actions that all equal `value`; it is asked for 0
    and 1.
    """
    with torch.no_grad():
        low, high = (actions_at(value).float().cpu().numpy() for value in (0.0, 1.0))
    # A joint that never moved in the training data has almost no spread, and float32 rounding of its
    # value would then read as a large difference.
    return np.maximum(np.abs(high - low), 1e-4 * np.abs(low) + 1e-6).astype(np.float32)


def load_policy(
    cfg: ExportConfig,
) -> tuple[PreTrainedPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """The whole trained policy, in eval mode on the backend's device, and its processors set to that device."""
    from lerobot.policies.factory import load_pretrained_policy

    config = cast(PreTrainedConfig, cfg.policy)
    config.device = cfg.backend.device
    # A checkpoint can save torch.compile on; the export traces the policy's own code and compiles it with
    # its backend.
    if hasattr(config, "compile_model"):
        config.compile_model = False
    # Built on the device, so the weights are copied there once from the checkpoint, with no second copy of
    # the model on the CPU.
    with torch.device(cfg.backend.device):
        # Strict: a weight the checkpoint does not fill would stay random, and no later check could tell.
        policy = load_pretrained_policy(config, strict=True)
    if hasattr(policy, "merge_and_unload"):
        # A PEFT adapter: compile the base weights with the adapter folded in.
        policy = policy.merge_and_unload()
    policy.eval()
    return policy, *load_processors(config)


def load_processors(config: PreTrainedConfig) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """The checkpoint's saved processors, with their device steps on `config.device`."""
    from lerobot.policies.factory import make_pre_post_processors

    overrides = {"device_processor": {"device": config.device}}
    return make_pre_post_processors(
        config,
        pretrained_path=config.pretrained_path,
        pretrained_revision=config.pretrained_revision,
        preprocessor_overrides=overrides,
        postprocessor_overrides=overrides,
    )
