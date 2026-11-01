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

"""Export TD-MPC as one program that plans each chunk with MPPI/CEM, reading its random draws from the noise.

At every chunk TD-MPC samples action sequences, scores them with its world model, and refines a Gaussian over
them for a fixed number of iterations, warm started from the mean its previous plan ended with. The program does
the same: it takes the robot's frame, the noise its draws read and that mean, and returns the actions
`select_action` queues from the plan and the mean for the next one. The bundle passes the mean from chunk to chunk
as its `carry`, and starts again from zeros after a reset, which the planner treats exactly as no previous plan.
A policy that does not plan (`use_mpc=False`) acts with its policy network alone, and its program takes only the
frame.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.bundle.manifest import NOISE
from lerobot.configs import PolicyFeature
from lerobot.export.recipe import (
    ActionChunkRecipe,
    Part,
    ReplayDraws,
    RunPlan,
    TestCase,
    random_frame,
    to_policy_input,
)
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION

from .modeling_tdmpc import TDMPCPolicy

# The mean the program reads and writes, as the bundle carries it from chunk to chunk.
PLAN_MEAN = "plan_mean"
# The first chunk of the test case plans from zeros, and each of the others from the mean the one before left.
TEST_CHUNKS = 3


class TDMPCExportRecipe(ActionChunkRecipe):
    """TD-MPC's rollout as one program: the frame, the noise and the previous plan's mean in; the chunk and the
    plan's mean out.

    When the policy plans, the test case plays a few chunks, each from its own frame and noise, with the policy's
    own `select_action`, so it covers the warm start; without planning it is one chunk with no noise. A plan
    draws random tensors of several shapes, so the noise holds them one after another, flat, and each is read in
    the order the planner draws it (see `ReplayDraws`).
    """

    # On a GPU the program and the policy may sum in other orders, so they are held to the export's tolerance.
    exact = False

    def _execution_steps(self) -> int:
        config = self.policy.config
        # What `select_action` queues from one plan: the first action repeated, or the first few.
        return config.n_action_repeats if config.n_action_repeats > 1 else config.n_action_steps

    def _make_case(self, frame: dict[str, np.ndarray]) -> TestCase:
        """A few chunks from their own frames and noises (one with no noise without planning), and the actions
        PyTorch plays at the last one."""
        planning = self.policy.config.use_mpc
        self._reset()
        with torch.no_grad(), ReplayDraws() as draws:
            self.policy.select_action(self.preprocessor(self._observation(frame, self.task)))
        self._draw_shapes = draws.shapes
        self._noise_shape = (sum(map(math.prod, self._draw_shapes)),) if self._draw_shapes else None
        chunks = TEST_CHUNKS if planning else 1
        self._case_frames = [
            frame,
            *(random_frame(self._features, self._generator) for _ in range(chunks - 1)),
        ]
        self._case_noises = [self._random_noise() for _ in self._case_frames]
        actions = self._rollout(self._case_frames, self._case_noises)
        if not np.array_equal(actions, self._rollout(self._case_frames, self._case_noises)):
            raise ValueError(
                f"This {self.policy.config.type} policy computes other chunks from the same frames and noises "
                "each time, so a compiled copy cannot be checked against it."
            )
        scale = self._action_scale(actions.shape[-1])
        if not planning:
            return TestCase(frame, None, actions, scale)
        frames = {name: np.stack([f[name] for f in self._case_frames]) for name in self._features}
        return TestCase(frames, np.stack(self._case_noises), actions, scale)

    def parts(self) -> Iterator[Part]:
        """The one part, after checking that it plays the policy's chunks for two sets of frames and noises."""
        module = _TDMPCChunk(
            self.policy,
            self.program_preprocessor(),
            self.postprocessor,
            list(self._features),
            self._visual,
            self._steps,
            self._draw_shapes,
        )
        self._check_chunks(module)
        carried = [PLAN_MEAN] if self.policy.config.use_mpc else []
        noise = [NOISE] if self.policy.config.use_mpc else []
        example = self._program_inputs(self._case_frames[-1], self._case_noises[-1], self._zero_mean())
        yield Part("model", module, example, [*self._features, *noise, *carried], [ACTION, *carried])

    def plan(self) -> RunPlan:
        """One run, which reads the mean of the previous plan when the policy plans."""
        if not self.policy.config.use_mpc:
            return RunPlan()
        return RunPlan(carry={PLAN_MEAN: tuple(self._zero_mean().shape)})

    def _rollout(self, frames: list[dict[str, np.ndarray]], noises: list[np.ndarray | None]) -> np.ndarray:
        """The actions `lerobot-rollout` plays at the last of `frames`, each the first tick of a chunk, with the
        random draws of each chunk read from its noise."""
        self._reset()
        with torch.no_grad():
            for frame, noise in zip(frames, noises, strict=True):
                batch = self.preprocessor(self._observation(frame, self.task))
                draws = (
                    None
                    if noise is None
                    else _split(torch.from_numpy(noise).to(self._device), self._draw_shapes)
                )
                with ReplayDraws(draws) as replay:
                    chunk = [self.policy.select_action(dict(batch)) for _ in range(self._steps)]
                if len(replay.shapes) != len(self._draw_shapes):
                    raise ValueError(
                        f"The {self.policy.config.type} policy drew {len(replay.shapes)} random tensors in a chunk, "
                        f"not the {len(self._draw_shapes)} of one plan."
                    )
                actions = self.postprocessor(torch.stack(chunk, dim=1))
        return actions[0].float().cpu().numpy()

    def _check_chunks(self, module: nn.Module) -> None:
        """Refuse a program that plays other chunks than the policy, on the test case and on other frames and noises."""
        frames = [random_frame(self._features, self._generator) for _ in self._case_frames]
        noises = [self._random_noise() for _ in frames]
        for case_frames, case_noises, expected in (
            (self._case_frames, self._case_noises, self._case.actions),
            (frames, noises, self._rollout(frames, noises)),
        ):
            mean = self._zero_mean()
            with torch.no_grad():
                for frame, noise in zip(case_frames, case_noises, strict=True):
                    outputs = module(*self._program_inputs(frame, noise, mean))
                    actions, mean = outputs if self.policy.config.use_mpc else (outputs, mean)
            self._check_program(actions.cpu().numpy(), expected)

    def _program_inputs(
        self, frame: dict[str, np.ndarray], noise: np.ndarray | None, mean: Tensor
    ) -> tuple[Tensor, ...]:
        inputs = self._inputs(frame, noise)
        return (*inputs, mean) if self.policy.config.use_mpc else inputs

    def _zero_mean(self) -> Tensor:
        """The mean before the first plan: zeros, which the planner treats as no previous plan."""
        config = self.policy.config
        # TD-MPC refuses to build without an action feature.
        action = cast(PolicyFeature, config.action_feature)
        return torch.zeros(config.horizon, 1, action.shape[0], device=self._device)


class _TDMPCChunk(nn.Module):
    """The program of [`TDMPCExportRecipe`]."""

    def __init__(
        self,
        policy: TDMPCPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        frame_names: list[str],
        visual: set[str],
        execution_steps: int,
        draw_shapes: list[tuple[int, ...]],
    ):
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.frame_names = frame_names
        self.visual = visual
        self.execution_steps = execution_steps
        self.draw_shapes = draw_shapes

    def forward(self, *inputs: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        config = self.policy.config
        observation: dict[str, Any] = {
            name: to_policy_input(x, name in self.visual)
            for name, x in zip(self.frame_names, inputs, strict=False)
        }
        batch = self.preprocessor({**observation, "task": "", "robot_type": ""})
        z = self.policy.model.encode(batch)
        mean = None
        if config.use_mpc:
            noise, previous = inputs[len(self.frame_names) :]
            with ReplayDraws(_split(noise, self.draw_shapes)):
                actions, mean = self.policy.plan_with_warm_start(z, previous)
        else:
            actions = self.policy.model.pi(z).unsqueeze(0)
        actions = torch.clamp(actions, -1, +1)
        if config.n_action_repeats > 1:
            chunk = actions[:1].expand(self.execution_steps, -1, -1)
        else:
            chunk = actions[: self.execution_steps]
        robot_actions = self.postprocessor(chunk.transpose(0, 1))[0].float()
        return robot_actions if mean is None else (robot_actions, mean)


def _split(noise: Tensor, shapes: list[tuple[int, ...]]) -> list[Tensor]:
    """The draws of a plan, one per shape of `shapes`, read one after another from the flat `noise`."""
    pieces = noise.split([math.prod(shape) for shape in shapes])
    return [piece.view(shape) for piece, shape in zip(pieces, shapes, strict=True)]
