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
import contextlib
import functools
import inspect
import json
import logging
import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.overrides import TorchFunctionMode

from lerobot.bundle.manifest import NOISE, ActionTokens, ProgramRun
from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig, is_depth_map
from lerobot.policies import PreTrainedPolicy, prepare_observation_for_inference
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    DeviceProcessorStep,
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
# Ten bfloat16 denoising steps amplify rounding: pi0.5's LIBERO checkpoint's TensorRT bundle differed from its test
# case by 0.11 normalized, where eager PyTorch with another attention kernel moved the same frame by 0.03. EO-1's
# TensorRT bundle differed from its test case by 0.079 on an H100, where another kernel moved it by 0.076. Another
# attention kernel moves Wall-OSS 0.5's actions by up to 0.16.
BFLOAT16_STEPS_TOLERANCE = 0.2
# A part checked on its own (`ProgramCheck`) may differ from PyTorch by this many times what rounding alone moves
# it. Correct LingBot-VA programs from TensorRT and ExecuTorch's CUDA backend measured up to 1.85 times; a wrong
# attention head size, swapped keys and values, a skipped layer or a zeroed output measured 9 to 830 times.
PROGRAM_FLOOR_FACTOR = 3.0
# A part checked on its own must compute a floating point output again, with nothing changed, within
# `RERUN_STEPS` rounding steps of the least precise float type it reads, holds or returns, or
# `RERUN_TOLERANCE` of its largest value, whichever is larger. The fraction covers float32 sums in an order
# that varies between runs: GPU atomic adds moved a float32 output by 9e-7 on an A100, 7.5 steps. It sits
# halfway, on a log scale, between that and random numbers of 1e-2 of a part's size, which must be refused.
RERUN_STEPS = 3
RERUN_TOLERANCE = 1e-4
# The test case takes the first random frame whose actions PyTorch plays again within the tolerance after the frame's
# floating point values and the noise move by one rounding step, up or down at random, `TEST_FRAME_TRIES` times. On a
# frame where two choices nearly tie, rounding alone picks one or the other: EVO1's gripper on one LIBERO frame moved
# its bfloat16 actions by 1.5 on 7 of 32 such steps, as much as a broken program moves them, so no compiled copy
# could be checked there. 16 tries miss such a frame about 2 times in 100.
TEST_FRAME_DRAWS = 8
TEST_FRAME_TRIES = 16
# A random test image blends a coarse grid of random colors (rows, columns), so it changes smoothly like a
# camera image. On independent random pixels a vision-language policy's bfloat16 actions differ more between
# two correct compilations, so the check would measure noise.
_IMAGE_CELLS = (6, 8)
# Two tasks whose first words differ, so they stay different when a tokenizer cuts them short.
_PROBE_TASKS = ("pick up the red cube", "stop")
_RANDOM_NORMAL = {torch.randn, torch.randn_like, torch.normal, Tensor.normal_}
_MULTINOMIAL = {torch.multinomial, Tensor.multinomial}
_RANDOM_INTEGERS = {torch.randint}
_HALF = (torch.bfloat16, torch.float16)


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
        check (`ProgramCheck`, *optional*):
            What the compiled program must compute on its own, checked at export before the bundle's test
            case.
    """

    name: str
    module: nn.Module
    example_inputs: tuple[Tensor, ...]
    inputs: list[str]
    outputs: list[str]
    check: ProgramCheck | None = None

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
class ProgramCheck:
    """One call of a part, which its compiled program must reproduce on its own, within what rounding moves.

    A long chain of bfloat16 programs, such as many denoising steps with classifier-free guidance, can amplify
    rounding until two correct compilations play actions as different as a wrong one does, so the bundle's test
    case cannot tell them apart. Fed the inputs the eager chain gave the part, each program is compared with the
    part's own outputs before anything builds up. `measure` makes the check. It is meant for bfloat16 and float16
    parts: a float32 part has no wider rerun to measure its rounding against, so a compiler that orders its float32
    sums otherwise can miss the floor of one rounding step.

    Args:
        inputs (`tuple[Tensor, ...]`):
            The part's inputs, contiguous, as a bundle passes them.
        outputs (`tuple[Tensor, ...]`):
            What the part computes from them in PyTorch.
        floor (`list[tuple[float, float]]`):
            For each output, the largest and the mean relative difference that rounding alone makes (see
            `relative_errors`): the part with other attention kernels and matmul settings, and in float32, and at
            least one rounding step of the output's dtype for the largest. A floating point output that none of
            these moved, such as a copy, gets one rounding step for both; an integer output must match exactly.
    """

    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]
    floor: list[tuple[float, float]]

    @classmethod
    def measure(
        cls,
        module: nn.Module,
        inputs: Sequence[Tensor],
        float32: Callable[[], contextlib.AbstractContextManager] | None = None,
        *,
        with_float32: bool = True,
    ) -> ProgramCheck:
        """Run `module` on `inputs`, then again with each change that only rounds otherwise, and keep the floor.

        `float32` runs the module in float32, by default with its own weights in float32 (see `in_float32`). A
        recipe whose part also reads a dtype elsewhere, such as from its policy, passes one that sets that too.
        `with_float32=False` leaves the float32 run out, for a part whose float32 copy does not fit beside the
        policy. Only on CUDA do the other changes, other attention kernels and matmul settings, still measure its
        floor; elsewhere only the attention kernel changes, which moves no part without attention. Refuses a
        module whose first rerun, with nothing changed, moves an output more than rounding can.
        """
        inputs = tuple(x.contiguous() for x in inputs)
        wide = tuple(x.float() if x.dtype in _HALF else x for x in inputs)
        with torch.no_grad():
            # Copies, since a part may return a buffer of its own that the reruns write into.
            outputs = tuple(x.clone() for x in _call(module, inputs))
            changes = [
                (change, inputs) for change in [contextlib.nullcontext, *_rounding_changes(outputs[0].device)]
            ]
            if with_float32:
                changes.append((float32 or (lambda: in_float32(module)), wide))
            measured = []
            for change, change_inputs in changes:
                with change():
                    measured.append(relative_errors(_call(module, change_inputs), outputs))
        # The first rerun changes nothing, so it measures what kernels vary between runs, unless the part
        # draws random numbers, which every rerun would then fold into the floor. A part may compute in
        # bfloat16 and return float32, which then varies by bfloat16's steps.
        step = max(_step(t) for t in (*inputs, *module.parameters(), *module.buffers(), *outputs))
        for index, ((again, _), x) in enumerate(zip(measured[0], outputs, strict=True)):
            # An output that holds NaN, or an integer one a rerun moves, is refused below for that reason.
            if (
                x.is_floating_point()
                and again > max(RERUN_STEPS * step, RERUN_TOLERANCE)
                and not x.isnan().any()
            ):
                raise ValueError(
                    f"Output {index} of the part changes by more than rounding does when it runs again on "
                    "the same inputs, as when it draws random numbers or keeps dropout on, so a compiled "
                    "program cannot be checked against it."
                )
        floor = [(0.0, 0.0)] * len(outputs)
        for errors in measured:
            floor = [(max(a, c), max(b, d)) for (a, b), (c, d) in zip(floor, errors, strict=True)]
        # A compiler may also round the largest value the other way, or contract arithmetic that no rerun moved,
        # such as a multiply and an add into one, by a step.
        floor = [
            (max(a, _step(x)), b) if b > 0 else (_step(x), _step(x))
            for (a, b), x in zip(floor, outputs, strict=True)
        ]
        for index, pair in enumerate(floor):
            # An infinite floor would let any output pass.
            if not np.isfinite(pair).all():
                raise ValueError(
                    f"Output {index} of the part holds NaN in PyTorch, or a rerun that only rounds otherwise "
                    "changes it without bound, such as an integer it computes from rounded values, so a compiled "
                    "program cannot be checked against it."
                )
        return cls(inputs, outputs, floor)

    def failures(self, name: str, outputs: Sequence[Tensor]) -> list[str]:
        """How the compiled program's `outputs` miss this check, empty when they pass. Logs each output's error."""
        if len(outputs) != len(self.outputs):
            return [f"{name} returns {len(outputs)} outputs, not {len(self.outputs)}"]
        failures = []
        errors = relative_errors(outputs, self.outputs)
        for index, (error, floor) in enumerate(zip(errors, self.floor, strict=True)):
            logger.info(
                "Program %s output %d on its own: largest relative difference %.2e (rounding %.2e), mean %.2e "
                "(rounding %.2e)",
                name,
                index,
                error[0],
                floor[0],
                error[1],
                floor[1],
            )
            if not all(e <= PROGRAM_FLOOR_FACTOR * f for e, f in zip(error, floor, strict=True)):
                failures.append(
                    f"{name} output {index}: largest relative difference {error[0]:.3g} and mean {error[1]:.3g}, "
                    f"where rounding moves it by up to {floor[0]:.3g} and {floor[1]:.3g}"
                )
        return failures

    def save(self, path: Path) -> None:
        """Write the check to `path`, for the step that runs the compiled program."""
        # Contiguous copies, since safetensors refuses a tensor that is not contiguous or shares memory with
        # another.
        tensors = {
            f"input.{i}": x.clone(memory_format=torch.contiguous_format) for i, x in enumerate(self.inputs)
        }
        tensors.update(
            {
                f"output.{i}": x.clone(memory_format=torch.contiguous_format)
                for i, x in enumerate(self.outputs)
            }
        )
        save_file(tensors, path, metadata={"floor": json.dumps(self.floor)})

    @classmethod
    def load(cls, path: Path, device: str) -> ProgramCheck:
        """Read a check `save` wrote, its tensors on `device`."""
        with safe_open(path, framework="pt") as file:
            floor = [tuple(pair) for pair in json.loads(file.metadata()["floor"])]
        tensors = load_file(path, device=device)
        return cls(
            tuple(tensors[f"input.{i}"] for i in range(sum(k.startswith("input.") for k in tensors))),
            tuple(tensors[f"output.{i}"] for i in range(len(floor))),
            cast(list[tuple[float, float]], floor),
        )


def relative_errors(outputs: Sequence[Tensor], expected: Sequence[Tensor]) -> list[tuple[float, float]]:
    """For each output, how far it is from `expected`, as two ratios.

    The largest absolute difference over the largest absolute value of `expected`, and the mean absolute
    difference over the mean absolute value. An output of another shape than `expected` counts as infinitely
    far. Where `expected` is not floating point, the output counts as infinitely far unless it is equal. Where
    both hold the same infinity, such as an additive mask, they are equal, and the infinities of `expected`
    count as 0 in its size. Any other infinity or NaN makes the output infinitely far.
    """
    errors = []
    for output, reference in zip(outputs, expected, strict=True):
        output = output.to(reference.device)
        if output.shape != reference.shape:
            errors.append((float("inf"), float("inf")))
        elif not reference.is_floating_point():
            errors.append((0.0, 0.0) if torch.equal(output, reference) else (float("inf"), float("inf")))
        else:
            output, reference = output.double(), reference.double()
            difference = (output - reference).abs().masked_fill(output == reference, 0)
            magnitude = reference.abs().masked_fill(~reference.isfinite(), 0)
            errors.append(
                (
                    _ratio(difference.max(), magnitude.max()) if difference.numel() else 0.0,
                    _ratio(difference.mean(), magnitude.mean()) if difference.numel() else 0.0,
                )
            )
    return errors


def _ratio(difference: Tensor, magnitude: Tensor) -> float:
    if torch.isnan(difference):
        return float("inf")
    if difference == 0:
        return 0.0
    return float(difference / magnitude) if magnitude > 0 else float("inf")


def _call(module: nn.Module, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    # A clone each time, since a part may write into an input, such as a cache.
    outputs = module(*(x.clone() for x in inputs))
    return outputs if isinstance(outputs, tuple) else (outputs,)


def _rounding_changes(device: torch.device) -> list[Callable[[], contextlib.AbstractContextManager]]:
    """Settings that change only how a computation rounds: other attention kernels, and more on a GPU.

    On a GPU, besides the attention kernels: TF32, full precision reductions in bfloat16 matmuls, cuDNN's
    benchmarked convolutions and cuBLASLt's matmuls.
    """
    changes: list[Callable[[], contextlib.AbstractContextManager]] = [
        functools.partial(sdpa_kernel, SDPBackend.MATH)
    ]
    if device.type != "cuda":
        return changes
    for backend in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION):
        # A kernel that does not take these inputs leaves them to the math one.
        changes.append(functools.partial(sdpa_kernel, [backend, SDPBackend.MATH]))
    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    for owner, setting, value in (
        (matmul, "allow_tf32", True),
        (cudnn, "allow_tf32", True),
        (matmul, "allow_bf16_reduced_precision_reduction", False),
        (cudnn, "benchmark", True),
    ):
        changes.append(functools.partial(_setting, owner, setting, value))
    changes.append(_cublaslt)
    return changes


@contextlib.contextmanager
def _setting(owner: Any, name: str, value: Any) -> Iterator[None]:
    saved = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, saved)


@contextlib.contextmanager
def _cublaslt() -> Iterator[None]:
    saved = torch.backends.cuda.preferred_blas_library()
    torch.backends.cuda.preferred_blas_library("cublaslt")
    try:
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(saved)


@contextlib.contextmanager
def in_float32(module: nn.Module) -> Iterator[None]:
    """`module` with its bfloat16 and float16 weights in float32, which convert back exactly, and TF32 off."""
    tensors = [t for t in [*module.parameters(), *module.buffers()] if t.dtype in _HALF]
    dtypes = [t.dtype for t in tensors]
    with contextlib.ExitStack() as stack:
        stack.enter_context(_setting(torch.backends.cuda.matmul, "allow_tf32", False))
        stack.enter_context(_setting(torch.backends.cudnn, "allow_tf32", False))
        for tensor in tensors:
            tensor.data = tensor.data.float()
        try:
            yield
        finally:
            for tensor, dtype in zip(tensors, dtypes, strict=True):
                tensor.data = tensor.data.to(dtype)


def _step(x: Tensor) -> float:
    """One rounding step of `x`'s dtype, relative to its values; 0 for one that is not floating point."""
    return float(torch.finfo(x.dtype).eps) if x.is_floating_point() else 0.0


@dataclass
class TestCase:
    """A frame, the starting noise, and the actions PyTorch computes for them.

    Args:
        frame (`dict[str, np.ndarray]`):
            The frame as the robot sends it: uint8 HWC images and float32 arrays. For a recipe whose plan has a
            `history`, the frames of several control ticks, oldest first, stacked on a new first axis; the
            actions are those computed at the last one. For one whose plan has a `carry`, likewise the frames of
            several chunks, each one's first tick; the actions are those of the last chunk. For one whose plan
            has `first_runs`, the frames of every tick of several chunks from a reset.
        noise (`np.ndarray`):
            The starting noise, for a policy that takes one, or `None`. With a `carry` or `first_runs`, the noise of each
            chunk, stacked on a new first axis like the frames.
        actions (`np.ndarray`):
            The float32 actions to play, in robot units, shaped `[execution_steps, action_dim]`. With
            `first_runs`, the action played at each tick.
        action_scale (`np.ndarray`):
            For each action component, the robot units of one step of the policy's normalized actions.
    """

    frame: dict[str, np.ndarray]
    noise: np.ndarray | None
    actions: np.ndarray
    action_scale: np.ndarray


@dataclass
class TaskTokenData:
    """How a policy's task becomes the `task_embeddings` and `task_length` its parts take.

    The bundle runs any task this way, without PyTorch: see `lerobot.bundle.manifest.TaskTokens`, which also says
    what the parts take. The bundle makes both values, so the test case's frame holds neither, and it runs the
    exported task, which must be one of `known_tasks`.

    Args:
        prompt_ops (`list[list[str]]`):
            The ops that make the prompt text from the task, as `TaskTokens.prompt_ops` describes.
        tokenizer (`Path`):
            The policy's tokenizer file, in the format of the `tokenizers` library, copied as it is.
        embeddings (`Tensor`):
            The token embedding table, `[vocabulary, width]`, in float32 or bfloat16.
        known_tasks (`dict[str, list[int]]`):
            The token ids of the prompt of each task known at export, such as the dataset's tasks: what the
            tokenizer gives the text `prompt_ops` make of the task, with its special tokens, without padding
            or truncation.
        max_tokens (`int`):
            The most tokens the parts take.
    """

    prompt_ops: list[list[str]]
    tokenizer: Path
    embeddings: Tensor
    known_tasks: dict[str, list[int]]
    max_tokens: int


@dataclass
class ActionTokenData:
    """How the token ids a policy generates become its actions, which the bundle decodes without PyTorch.

    Args:
        tokenizer (`str`):
            The policy's FAST tokenizer, as the JSON the `tokenizers` library saves.
        spec (`ActionTokens`):
            The manifest's `action_tokens`, naming the file `tokenizer` is written to.
    """

    tokenizer: str
    spec: ActionTokens


@dataclass
class RunPlan:
    """How the compiled parts of a bundle run, for a recipe whose parts do not simply run once in order.

    No run, in `runs` or `first_runs`, is named `noise` or runs a program so named: on CUDA, that is the noise
    draw.

    Args:
        runs (`list[ProgramRun]`, *optional*):
            The runs in order, each naming the part it runs as its program. A part may run several times, for
            example a denoising step. `None` runs each part once, in the order it was yielded, with its own
            names.
        first_runs (`list[list[ProgramRun]]`, *optional*):
            The runs of each of the first chunks after a reset, for a policy whose first chunks compute
            otherwise (see `BundleManifest.first_runs`).
        constants (`dict[str, np.ndarray]`, *optional*):
            Named arrays the runs can read, such as each step's timestep.
        task_tokens (`TaskTokenData`, *optional*):
            How the task becomes the `task_embeddings` and `task_length` the parts take, for a policy that
            reads it.
        action_tokens (`ActionTokenData`, *optional*):
            How the token ids the runs generate become the actions later runs take, for a policy that writes its
            actions as tokens.
        history (`dict[str, int]`, *optional*):
            For each key of the frame the parts read with its past values, how many frames: the parts take it
            as the last ones, oldest first, stacked on a new first axis (see `BundleManifest.history`).
        carry (`dict[str, tuple[int, ...] | None]`, *optional*):
            The values the runs pass from one chunk to the next, with their shapes: float32 zeros after a reset,
            or, without a shape, what the first chunk's runs write (see `BundleManifest.carry`).
    """

    runs: list[ProgramRun] | None = None
    first_runs: list[list[ProgramRun]] = field(default_factory=list)
    constants: dict[str, np.ndarray] = field(default_factory=dict)
    task_tokens: TaskTokenData | None = None
    action_tokens: ActionTokenData | None = None
    history: dict[str, int] = field(default_factory=dict)
    carry: dict[str, tuple[int, ...] | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a run named like the one that draws a bundle's noise on CUDA, or a run of its program."""
        if any(
            NOISE in (run.name, run.program) for runs in (self.runs or [], *self.first_runs) for run in runs
        ):
            raise ValueError(
                f"A run cannot be named {NOISE} or run a program so named: on CUDA, a bundle draws its noise "
                "with a run and a program so named."
            )


def repeated_runs(
    program: str,
    inputs: list[str],
    outputs: list[str],
    steps: Sequence[dict[str, np.ndarray]],
    start: dict[str, str] | None = None,
) -> tuple[list[ProgramRun], dict[str, np.ndarray]]:
    """The runs of a part that runs once per step, such as a denoising step, and the constants they read.

    Run `i` reads and writes the part's own `inputs` and `outputs` names, so a value the part reads and writes
    under one name, such as the sample `x`, passes from each run to the next. The first run reads
    `start[name]` in place of such an input, such as the starting `noise`. Each name in `steps[i]` is a
    constant of run `i` only, such as its timestep: the run reads it as `f"{name}_{i}"`, and the returned
    constants hold it.
    """
    runs, constants = [], {}
    for i, row in enumerate(steps):
        names = dict(start or {}) if i == 0 else {}
        names.update((name, f"{name}_{i}") for name in row)
        runs.append(ProgramRun(f"{program}_{i}", program, [names.get(n, n) for n in inputs], list(outputs)))
        constants.update({f"{name}_{i}": value for name, value in row.items()})
    return runs, constants


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

    A recipe that overrides `load` and `__init__`, to load its weights one part at a time, sets the attributes
    below itself; `lerobot-export` reads no other.

    **Attributes**:
        - **config** (`PreTrainedConfig`) -- The policy config the bundle records, naming the frame it
          takes.
        - **task** (`str | None`) -- The task the parts hold, or `None` when none was given and the policy
          reads none.
        - **tolerance** (`float`) -- The largest allowed action difference on the test case, in normalized
          action units.
        - **bfloat16_tolerance** (`float`) -- The default tolerance of a bfloat16 or float16 checkpoint. A
          recipe whose policy runs ten bfloat16 denoising steps, which amplify rounding, raises it to
          `BFLOAT16_STEPS_TOLERANCE`.
    """

    bfloat16_tolerance: float = BFLOAT16_TOLERANCE

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
        """Keep the policy, its processors and the configuration; the tolerance is `cfg`'s or the dtype's."""
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.cfg = cfg
        self.config: PreTrainedConfig = policy.config
        self.task: str | None = None
        self.tolerance = (
            cfg.tolerance
            if cfg.tolerance is not None
            else _default_tolerance(policy, self.bfloat16_tolerance)
        )

    @abc.abstractmethod
    def parts(self) -> Iterator[Part]:
        """Yield the parts in the order they run. `lerobot-export` traces and frees each before the next."""

    @abc.abstractmethod
    def test_case(self) -> TestCase:
        """The frame, noise and actions the bundle's programs must reproduce.

        `lerobot-export` asks for it once `parts` is used up, after `plan`.
        """

    def plan(self) -> RunPlan:
        """How the compiled parts run, asked once `parts` is used up. By default each part runs once."""
        return RunPlan()


class ChainRecipe(ExportRecipe):
    """A recipe that exports its policy as a chain of programs, which pass their values to each other by name.

    `parts` runs each part in PyTorch to make the next part's inputs and records the part's runs in `_runs`. The
    test case is then compared with the actions of the policy's own `predict_action_chunk`, which
    `_policy_actions` computes.

    **Attributes**:
        - **policy_name** (`str`) -- The policy's name in messages.
        - **text_steps** (`tuple[type, ...]`) -- The preprocessor's steps that write the prompt and tokenize it,
          which the programs do not run, for a recipe that uses `_tensor_preprocessor`.
        - **policy_tolerance** (`float | None`) -- The largest allowed difference from `_policy_actions`, in
          normalized action units, for a recipe that holds the chain closer to the policy than `tolerance`,
          which allows for the compiler. `None` uses `tolerance`.
    """

    policy_name: str
    text_steps: tuple[type, ...]
    policy_tolerance: float | None = None
    _runs: list[ProgramRun]

    @abc.abstractmethod
    def _policy_actions(self, case: TestCase) -> np.ndarray | None:
        """The actions the policy's own `predict_action_chunk` plays for the test case's frame and noise.

        `None` skips the comparison, for a policy that does not fit in memory beside the export.
        """

    def _part(
        self,
        name: str,
        module: nn.Module,
        inputs: tuple[Tensor, ...],
        input_names: list[str],
        output_names: list[str],
    ) -> Part:
        """The part, which runs once per chunk with its own names."""
        self._runs.append(ProgramRun(name, name, input_names, output_names))
        return Part(name, module, inputs, input_names, output_names)

    def _tensor_preprocessor(self) -> PolicyProcessorPipeline:
        """The preprocessor without `text_steps`."""
        return replace(
            self.preprocessor,
            steps=[s for s in self.preprocessor.steps if not isinstance(s, self.text_steps)],
        )

    def _check_against_policy(self, case: TestCase) -> None:
        """Refuse a test case whose actions differ from `_policy_actions` by more than the tolerance."""
        expected = self._policy_actions(case)
        if expected is None:
            return
        tolerance = self.tolerance if self.policy_tolerance is None else self.policy_tolerance
        error = float((np.abs(case.actions - expected) / case.action_scale).max())
        logger.info(
            "Test case against the whole policy: largest difference %.2e (tolerance %g)", error, tolerance
        )
        if not error <= tolerance:
            raise ValueError(
                f"The chained parts compute other actions than {self.policy_name}'s predict_action_chunk "
                f"({error:.2e} > {tolerance} in normalized units), so they do not export this checkpoint "
                "correctly."
            )


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

    **Attributes**:
        - **exact** (`bool`) -- Whether the program must compute, in PyTorch, the actions lerobot-rollout plays
          bit for bit, both with TF32 off as the export runs (rollout turns TF32 on by default, which moves a
          float32 policy's actions on a GPU a little). A recipe whose program computes them in another
          order, for example over a prompt padded to a fixed length, sets it to False, and the program is held
          to the export's tolerance instead.
    """

    exact: bool = True

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        cfg: ExportConfig,
    ) -> None:
        """Check that the policy can be exported this way, and make the test case."""
        super().__init__(policy, preprocessor, postprocessor, cfg)
        self._steps = self._execution_steps()
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
        self._case = self._make_case(frame)

    def parts(self) -> Iterator[Part]:
        """The one part, after checking that it computes what PyTorch does for two different frames."""
        module = _ActionChunk(
            self.predict_chunk,
            self.policy,
            self.program_preprocessor(),
            self.postprocessor,
            list(self._features),
            self._visual,
            self.task,
            self._steps,
            self._held_noise,
        )
        frame, noise = random_frame(self._features, self._generator), self._random_noise()
        for case_frame, case_noise, expected in (
            (self._case.frame, self._case.noise, self._case.actions),
            (frame, noise, self._reference(frame, noise)),
        ):
            with torch.no_grad():
                actions = module(*self._inputs(case_frame, case_noise)).cpu().numpy()
            self._check_program(actions, expected)
        inputs = [*self._features, *([NOISE] if self._noise_shape else [])]
        yield Part("model", module, self._inputs(self._case.frame, self._case.noise), inputs, [ACTION])

    def test_case(self) -> TestCase:
        """The random frame and noise, and the actions PyTorch computes for them as `lerobot-rollout` does."""
        return self._case

    def program_preprocessor(self) -> PolicyProcessorPipeline:
        """The preprocessor the program runs: the saved one.

        A recipe for a policy whose preprocessor does Python work on the frame, such as building a prompt with
        a transformers processor, returns one that computes the same with tensor operations. The program is
        still checked against the saved preprocessor.
        """
        return self.preprocessor

    def _check_program(self, actions: np.ndarray, expected: np.ndarray) -> None:
        if self.exact:
            if not np.array_equal(actions, expected):
                raise ValueError(
                    f"The {self.policy.config.type} policy computes something else as one module than it "
                    "does in lerobot-rollout, for example because a preprocessor step reads the frame in "
                    "Python. It needs its own export recipe."
                )
            return
        difference = float((np.abs(actions - expected) / self._case.action_scale).max())
        # Not `>`: a NaN difference must fail too.
        if not difference <= self.tolerance:
            raise ValueError(
                f"The {self.policy.config.type} policy's program computes actions {difference:.3g} normalized "
                f"units from the ones lerobot-rollout plays, more than the tolerance {self.tolerance}."
            )

    def predict_chunk(self, policy: PreTrainedPolicy, batch: dict[str, Any], noise: Tensor | None) -> Tensor:
        """The `[1, chunk, action]` chunk the policy predicts for a preprocessed batch, as rollout gets it.

        Without `noise` the policy draws its own. A recipe for a policy that predicts its actions another way
        overrides this. The program and the test case both call it, so they compute the same thing.
        """
        output = policy.predict_action_chunk(batch, **({} if noise is None else {NOISE: noise}))
        return transition_to_policy_action(policy_output_to_transition(output))

    def _execution_steps(self) -> int:
        """How many actions of a chunk `lerobot-rollout` plays, once the policy is known to fit this recipe."""
        try:
            return self.policy.chunk_inference_spec().execution_steps
        except ValueError as error:
            raise ValueError(
                f"This {self.policy.config.type} checkpoint cannot be exported: {error}"
            ) from error

    def _make_case(self, frame: dict[str, np.ndarray]) -> TestCase:
        """The test case: a frame, a random noise if the policy takes one, and the actions PyTorch computes.

        The frame is `frame`, or the first next random frame on which one rounding step of the inputs, tried
        `TEST_FRAME_TRIES` times, moves the actions by no more than the tolerance. It also sets `_held_noise` and
        `_noise_shape`, which `parts` and `_random_noise` read, so an override sets them too.
        """
        self._held_noise = self._seeded_noise()
        self._noise_shape = None if self._held_noise is not None else self._sampled_noise_shape(frame)
        floats = [p.dtype for p in self.policy.parameters() if p.is_floating_point()]
        step = max(float(torch.finfo(dtype).eps) for dtype in floats or [torch.float32])
        for draw in range(TEST_FRAME_DRAWS):
            if draw:
                frame = random_frame(self._features, self._generator)
            noise = self._random_noise()
            actions = self._reference(frame, noise)
            if not np.isfinite(actions).all():
                raise ValueError(
                    f"This {self.policy.config.type} policy computes actions that are not finite for a random "
                    "frame, so a compiled copy cannot be checked against it."
                )
            # PyTorch may round a rerun otherwise, as a part's rerun may in `ProgramCheck.measure`: in bfloat16 on
            # an A100, LingBot-VLA v2 came back 0.0035 apart (relative) in 5 of about 390 calls, where
            # `RERUN_STEPS` steps allow 0.023.
            ((again, _),) = relative_errors(
                [torch.from_numpy(self._reference(frame, noise))], [torch.from_numpy(actions)]
            )
            if not again <= max(RERUN_STEPS * step, RERUN_TOLERANCE):
                raise ValueError(
                    f"This {self.policy.config.type} policy computes another chunk from the same frame and noise "
                    "each time, so a compiled copy cannot be checked against it."
                )
            scale = self._action_scale(self._predicted_width)
            difference = 0.0
            for _ in range(TEST_FRAME_TRIES):
                moved = self._reference(*self._moved(frame, noise, step))
                difference = max(difference, float((np.abs(moved - actions) / scale).max()))
                if difference > self.tolerance:
                    break
            else:
                return TestCase(frame, noise, actions, scale)
            logger.info(
                "Random frame %d plays actions %.3g away once its inputs move by one rounding step, more than the "
                "tolerance %g: taking the next one",
                draw,
                difference,
                self.tolerance,
            )
        raise ValueError(
            f"On each of {TEST_FRAME_DRAWS} random frames, this {self.policy.config.type} policy plays other actions "
            f"once its inputs move by one rounding step, by more than the tolerance {self.tolerance}, so a compiled "
            "copy cannot be checked against it. Give a larger --tolerance."
        )

    def _moved(
        self, frame: dict[str, np.ndarray], noise: np.ndarray | None, step: float
    ) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
        """`frame`'s floating point values and `noise`, each moved up or down by `step` of itself, at random."""

        def move(x: np.ndarray | None) -> np.ndarray | None:
            if x is None or not np.issubdtype(x.dtype, np.floating):
                return x
            signs = torch.randint(0, 2, x.shape, generator=self._generator).numpy() * 2 - 1
            return (x + signs * step * np.abs(x)).astype(x.dtype)

        return {name: move(x) for name, x in frame.items()}, move(noise)

    def _action_scale(self, width: int) -> np.ndarray:
        return measure_action_scale(
            lambda value: self.postprocessor(torch.full((1, 1, width), value, device=self._device))[0, 0]
        )

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

    def _seeded_noise(self) -> Tensor | None:
        """The noise of a policy that draws the same noise for every chunk, as `predict_chunk` takes it.

        Such a policy seeds its own generator each time it predicts a chunk, so it computes the same chunk from
        the same frame. Its program holds this noise instead of taking one, and its bundle does the same. `None`,
        the default, for a policy that draws other noise each time, which its bundle draws afresh for each chunk.
        """
        return None

    def _sampled_noise_shape(self, frame: dict[str, np.ndarray]) -> tuple[int, ...] | None:
        """The shape of the one starting noise the policy draws for a chunk, or `None` if it takes none.

        The policy is run once without a noise and its random normal draws are recorded, so the shape is the
        one the policy itself uses. It is then run on one standard normal noise twice, once drawing it and
        once given it, which must compute the same chunk.
        """
        if NOISE not in inspect.signature(self.policy.predict_action_chunk).parameters:
            return None
        shape = self._recorded_noise_shape(frame)
        # The bundle draws a standard normal noise and gives it to the program, so the policy must compute the
        # same chunk from that noise as when it draws it itself. One that draws `5 + 0.5 * randn` would not.
        noise = torch.randn((1, *shape), generator=torch.Generator().manual_seed(0)).to(self._device)
        chunks = []
        # Both runs in a function mode, which turns off fused kernels such as PyTorch's transformer layers',
        # since those round differently. The first reads a copy, or a policy that moves its draw in place
        # would move the noise the second is given too.
        for draws, argument in ((ReplayDraws(noise[None].clone()), None), (ReplayDraws(), noise)):
            self._reset()
            with torch.no_grad(), draws:
                batch = self.preprocessor(self._observation(frame, self.task))
                chunks.append(self.predict_chunk(self.policy, batch, argument)[:, : self._steps])
        # NaN counts as equal here: the finite check that follows refuses NaN in the actions rollout plays.
        drawn, given = (chunk.float() for chunk in chunks)
        if drawn.shape != given.shape or not torch.allclose(drawn, given, rtol=0, atol=0, equal_nan=True):
            raise ValueError(
                f"The {self.policy.config.type} policy computes another chunk when it is given its starting "
                "noise than when it draws that noise itself, as when it changes the noise it draws before "
                "using it, or computes another chunk from the same noise each time. Its bundle draws a "
                "standard normal noise and gives it to the program, so the policy needs its own export "
                "recipe."
            )
        return shape

    def _recorded_noise_shape(self, frame: dict[str, np.ndarray]) -> tuple[int, ...]:
        """The shape of the one random normal tensor `predict_chunk` draws for one sample."""
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
            # The action scale is measured at this width: a policy may pad its actions, as EVO1 does.
            self._predicted_width = predicted.shape[-1]
            actions = self.postprocessor(predicted)
        return actions[0].float().cpu().numpy()

    def _inputs(self, frame: dict[str, np.ndarray], noise: np.ndarray | None) -> tuple[Tensor, ...]:
        arrays = [*frame.values(), *([] if noise is None else [noise])]
        return tuple(torch.from_numpy(x).to(self._device) for x in arrays)


class VisionGridRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] for a policy whose vision tower exports on a fixed image grid.

    A Qwen-VL vision tower reads its image grid (`image_grid_thw`) on the host, which `torch.export` cannot
    trace, so the policy takes what the tower derives from the grid as `vision_grid`. The recipe builds it from
    the first preprocessed batch, in the eager runs before the export (the test case and the checks), and
    passes it on every call; the policy refuses a frame with another grid. The batch's `image_grid_thw` holds one
    row per image, or one per camera of each sample, as LingBot-VLA v2's does.
    """

    grid_type: ClassVar[Any]
    """The policy's `FixedImageGrid` subclass, such as `Qwen2_5VisionGrid`."""
    _vision_grid: Any = None

    @abc.abstractmethod
    def vision_tower(self, policy: PreTrainedPolicy) -> nn.Module:
        """The vision tower that `grid_type.build` reads."""

    def predict_chunk(self, policy: PreTrainedPolicy, batch: dict[str, Any], noise: Tensor | None) -> Tensor:
        """The default recipe's chunk, with the policy's vision tower on the grid of the first batch."""
        if self._vision_grid is None:
            self._vision_grid = self.grid_type.build(
                self.vision_tower(policy), batch["image_grid_thw"].flatten(0, -2)
            )
        options: dict[str, Any] = {
            "vision_grid": self._vision_grid,
            **({} if noise is None else {NOISE: noise}),
        }
        output = policy.predict_action_chunk(batch, **options)
        return transition_to_policy_action(policy_output_to_transition(output))


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
        noise: Tensor | None = None,
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
        self.noise = noise

    def forward(self, *inputs: Tensor) -> Tensor:
        observation: dict[str, Any] = {
            name: to_policy_input(x, name in self.visual)
            for name, x in zip(self.frame_names, inputs, strict=False)
        }
        observation["task"] = self.task
        observation["robot_type"] = ""
        noise = inputs[-1][None] if len(inputs) > len(self.frame_names) else self.noise
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
      `[2, *p.shape]`;
    - a random integer tensor `torch.randint(low, high, size)` draws each value like a category of
      `high - low` equally likely ones, reading `noise[k]` shaped `[2, *size, high - low]`.

    Without a noise, the draws are made as usual, and `shapes` records the shape each would read.

    Args:
        noise (`Tensor` or `Sequence[Tensor]`, *optional*):
            The draws, stacked on the first axis, or one tensor per draw for draws of different shapes.
    """

    def __init__(self, noise: Tensor | Sequence[Tensor] | None = None) -> None:
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
        if func in _RANDOM_INTEGERS:
            low, high, size = _integer_range(args, kwargs)
            if self.noise is None:
                self.shapes.append((2, *size, high - low))
                return func(*args, **kwargs)
            first, second = self._next((2, *size, high - low))
            # The multinomial draw above with equal probabilities: the smallest exponential wins.
            picked = torch.argmin(first * first + second * second, dim=-1)
            return (low + picked).to(kwargs.get("dtype") or torch.int64)
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
        if self.noise is None or drawn == len(self.noise):
            raise ValueError(
                f"Draw {drawn} reads noise shaped {list(shape)}, but the noise holds only "
                f"{0 if self.noise is None else len(self.noise)} draws."
            )
        if tuple(self.noise[drawn].shape) != shape:
            raise ValueError(
                f"Draw {drawn} reads noise shaped {list(shape)}, but the noise holds "
                f"{list(self.noise[drawn].shape)} for it."
            )
        self.shapes.append(shape)
        return self.noise[drawn]


def _integer_range(args: tuple, kwargs: dict[str, Any]) -> tuple[int, int, tuple[int, ...]]:
    """The `low`, `high` and `size` of a call `torch.randint(high, size)` or `torch.randint(low, high, size)`."""
    named = {"low", "high", "size"} & kwargs.keys()
    order = ("low", "high", "size") if len(args) + len(named) == 3 else ("high", "size")
    values = {"low": 0, **dict(zip(order, args, strict=False)), **{name: kwargs[name] for name in named}}
    return values["low"], values["high"], tuple(values["size"])


def _default_tolerance(policy: PreTrainedPolicy, bfloat16_tolerance: float) -> float:
    if any(p.dtype in _HALF for p in policy.parameters()):
        return bfloat16_tolerance
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


def split_absolute_step(
    postprocessor: PolicyProcessorPipeline, action_dim: int
) -> tuple[PolicyProcessorPipeline, list[bool] | None]:
    """The postprocessor's steps before its enabled relative-to-absolute step, and the mask of that step.

    That step adds back the state the preprocessor cached in Python, which a program would hold as a constant, so a
    recipe runs the steps before it and adds the state its program takes as an input, with `to_absolute_actions`
    and the mask of the `action_dim` actions. The mask is None for a postprocessor without that step.
    """
    steps = list(postprocessor.steps)
    index = next(
        (i for i, s in enumerate(steps) if isinstance(s, AbsoluteActionsProcessorStep) and s.enabled), None
    )
    if index is None:
        return postprocessor, None
    if not all(isinstance(s, DeviceProcessorStep) for s in steps[index + 1 :]):
        raise ValueError("The relative-to-absolute action step must be the postprocessor's last step.")
    relative_step = cast(AbsoluteActionsProcessorStep, steps[index]).relative_step
    if relative_step is None:
        raise ValueError("The postprocessor's relative-to-absolute step is linked to no relative step.")
    # The relative step has no public mask accessor; LeRobot's RTC and statistics code call this one too.
    return replace(postprocessor, steps=steps[:index]), relative_step._build_mask(action_dim)


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


_MEMINFO = Path("/proc/meminfo")
_CGROUP = Path("/proc/self/cgroup")
_CGROUPS = Path("/sys/fs/cgroup")


def _available_memory() -> int | None:
    """Bytes Linux expects new allocations can use without swapping (page cache counts), None elsewhere.

    In a container or another cgroup with a cgroup v2 memory limit, the room that limit leaves, if it is less.
    """
    if not _MEMINFO.exists():
        return None
    for line in _MEMINFO.read_text().splitlines():
        name, value, *_ = line.split()
        if name == "MemAvailable:":
            room = _cgroup_room()
            return int(value) * 1024 if room is None else min(int(value) * 1024, room)
    return None


def _cgroup_room() -> int | None:
    """Bytes left under the tightest cgroup v2 memory limit of this process and its parents, or None.

    Inactive page cache, which the cgroup counts as used and the kernel drops first, counts as room.
    """
    try:
        group = next(line[3:] for line in _CGROUP.read_text().splitlines() if line.startswith("0::"))
    except (OSError, StopIteration):
        return None
    rooms = []
    path = _CGROUPS / group.lstrip("/")
    while path.is_relative_to(_CGROUPS):
        try:
            limit = (path / "memory.max").read_text().strip()
            if limit != "max":
                stats = dict(line.split() for line in (path / "memory.stat").read_text().splitlines())
                used = int((path / "memory.current").read_text()) - int(stats.get("inactive_file", 0))
                rooms.append(max(int(limit) - used, 0))
        except (OSError, ValueError):
            pass
        path = path.parent
    return min(rooms, default=None)


def whole_policy_fits(model_file: Path, device: torch.device, policy_name: str, peak: float) -> bool:
    """Whether the whole policy fits in memory beside the export, to check the test case against it.

    It needs `peak` times its checkpoint's size, what the recipe measured for loading and running its policy,
    in the host's memory and, on a CUDA device, in the GPU's free memory too; a CUDA device whose host does not say how much memory it has is checked on the GPU alone.
    When it does not fit, or a CPU machine does not say how much memory it has, logs that the comparison is
    skipped; the bundle's own check of its test case still runs.
    """
    size = model_file.stat().st_size
    if device.type == "cuda":
        # Memory that PyTorch keeps cached for the parts already run is free to the policy too.
        torch.cuda.empty_cache()
        available, measured = torch.cuda.mem_get_info(device)[0], "free on the GPU"
        # A policy can be built in host memory before it moves to the GPU, and a Jetson's GPU uses the host's
        # memory.
        if (host := _available_memory()) is not None and host < available:
            available, measured = host, "available on the host"
    elif (linux_available := _available_memory()) is not None:
        available, measured = linux_available, "available"
    elif hasattr(os, "sysconf"):
        available, measured = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), "installed"
    else:
        # Windows has neither /proc/meminfo nor os.sysconf.
        logger.warning(
            "Skipped the comparison with the whole %s policy: this machine does not say how much memory it "
            "has. The bundle's own check against its test case still runs.",
            policy_name,
        )
        return False
    if peak * size <= available:
        return True
    logger.warning(
        "Skipped the comparison with the whole %s policy: it needs about %.1f GB, %.1f GB %s. The bundle's "
        "own check against its test case still runs.",
        policy_name,
        peak * size / 1e9,
        available / 1e9,
        measured,
    )
    return False


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


def refuse_peft_adapter(config: PreTrainedConfig, policy_name: str) -> None:
    """Refuse a PEFT adapter, for a recipe that reads each part's weights from the checkpoint's own file."""
    from lerobot.policies.factory import has_peft_adapter_config

    if (
        config.use_peft
        and config.pretrained_path is not None
        and has_peft_adapter_config(config.pretrained_path, config.pretrained_revision)
    ):
        raise ValueError(
            f"{config.pretrained_path} is a PEFT adapter, and {policy_name}'s export reads each part's weights "
            "from the checkpoint's model.safetensors, which an adapter does not hold. Merge the adapter into "
            "its base policy, save that, and export it. See https://huggingface.co/docs/lerobot/export."
        )


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
