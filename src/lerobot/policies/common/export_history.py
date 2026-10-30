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

"""Export recipes for the policies that read past frames: diffusion, Multi-Task DiT and VQ-BeT.

These policies keep their past frames in queues that `select_action` fills on every control tick. A bundle
keeps them instead (`RunPlan.history`), so its programs take each key of the frame as its last frames, stacked
oldest first, as many as the recipe's `_history()` names.
"""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.export.recipe import (
    ActionChunkRecipe,
    Part,
    ReplayDraws,
    RunPlan,
    TestCase,
    random_frame,
    to_policy_input,
)
from lerobot.policies.utils import populate_queues
from lerobot.processor import (
    PolicyProcessorPipeline,
    policy_output_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES


class HistoryChunkRecipe(ActionChunkRecipe):
    """The default recipe's frame, task and checks, for a policy that reads its past frames.

    The test case replays a few control ticks: at least two, so that their order counts, and fewer than the frames
    the policy keeps when it keeps more than two, so that the first frame stands in for the missing ones. Its actions are the policy's
    own `predict_action_chunk` on its queues, filled tick by tick as `select_action` fills them, with the
    policy's random draws read from the test case's noise (see `ReplayDraws`). A subclass makes the parts.
    """

    def _execution_steps(self) -> int:
        return self.policy.config.n_action_steps

    def _make_case(self, frame: dict[str, np.ndarray]) -> TestCase:
        ticks = [frame, *self._random_ticks()[1:]]
        _, shapes = self._history_reference(ticks, None)
        if len(set(shapes)) > 1:
            raise ValueError(
                f"The {self.policy.config.type} policy draws random tensors of different shapes, {shapes}, "
                "which one noise input cannot hold."
            )
        self._noise_shape = (len(shapes), *shapes[0]) if shapes else None
        noise = self._random_noise()
        actions, _ = self._history_reference(ticks, noise)
        if not np.isfinite(actions).all():
            raise ValueError(
                f"This {self.policy.config.type} policy computes actions that are not finite for random frames, "
                "so a compiled copy cannot be checked against it."
            )
        if not np.array_equal(actions, self._history_reference(ticks, noise)[0]):
            raise ValueError(
                f"This {self.policy.config.type} policy computes another chunk from the same frames and noise "
                "each time, so a compiled copy cannot be checked against it."
            )
        stacked = {name: np.stack([tick[name] for tick in ticks]) for name in self._features}
        return TestCase(stacked, noise, actions, self._action_scale(self._predicted_width))

    def plan(self) -> RunPlan:
        """Each part runs once, in order, and reads every key of the frame with its past values."""
        return RunPlan(history=self._history())

    def _history(self) -> dict[str, int]:
        """How many frames of each key the bundle keeps: `n_obs_steps` of every key.

        A recipe that keeps another number of some key cannot build its parts with `HistoryBatch`, which
        stacks the same number of every key.
        """
        return dict.fromkeys(self._features, self.policy.config.n_obs_steps)

    def _random_ticks(self) -> list[dict[str, np.ndarray]]:
        return [
            random_frame(self._features, self._generator)
            for _ in range(max(self.policy.config.n_obs_steps - 1, 2))
        ]

    def _history_inputs(self, ticks: dict[str, np.ndarray], noise: np.ndarray | None) -> tuple[Tensor, ...]:
        """What the programs take after `ticks`: each key's last frames, as many as `_history` keeps, the
        first standing in for any not seen yet, then the noise."""
        history = self._history()
        inputs = []
        for name, value in ticks.items():
            size = history[name]
            frames = [value[0]] * max(size - len(value), 0) + list(value[-size:])
            inputs.append(torch.from_numpy(np.stack(frames)).to(self._device))
        if noise is not None:
            inputs.append(torch.from_numpy(noise).to(self._device))
        return tuple(inputs)

    def _push(self, frame: dict[str, np.ndarray]) -> dict[str, Any]:
        """What `select_action` does with a frame before it predicts: preprocess, stack the cameras, queue."""
        batch = dict(self.preprocessor(self._observation(frame, self.task)))
        batch.pop(ACTION, None)
        image_features = list(self.policy.config.image_features)
        if image_features:
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in image_features], dim=-4)
        self.policy._queues = populate_queues(self.policy._queues, batch)
        return batch

    def _history_reference(
        self, ticks: list[dict[str, np.ndarray]], noise: np.ndarray | None
    ) -> tuple[np.ndarray, list[tuple[int, ...]]]:
        """The actions the policy computes at the last of `ticks`, and the shapes of its random draws.

        It also sets `_predicted_width`, the width the action scale is measured at, which `_make_case` reads.
        """
        self._reset()
        with torch.no_grad():
            for frame in ticks:
                batch = self._push(frame)
            draws = ReplayDraws(None if noise is None else torch.from_numpy(noise).to(self._device))
            with draws:
                output = self.policy.predict_action_chunk(batch)
            predicted = transition_to_policy_action(policy_output_to_transition(output))[:, : self._steps]
            # Before the postprocessor, which may crop a padded width once it unnormalized it.
            self._predicted_width = predicted.shape[-1]
            if noise is not None and len(draws.shapes) != len(noise):
                raise ValueError(f"The policy drew {len(draws.shapes)} random tensors, not {len(noise)}.")
            actions = self.postprocessor(predicted)
        return actions[0].float().cpu().numpy(), draws.shapes

    def _history_batch(self) -> HistoryBatch:
        queued = [key for key in self.policy._queues if key != ACTION]
        return HistoryBatch(
            self.preprocessor,
            list(self._features),
            self._visual,
            list(self.policy.config.image_features),
            queued,
            self.task,
        )

    def _check_chain(self, run: Callable[..., Tensor]) -> None:
        """Refuse parts whose chain `run`, given the programs' inputs, computes another chunk than the policy.

        It is checked on the test case and on another noise and more random ticks than the bundle keeps, so that
        each frame the programs take is another one, as on a running robot.
        """
        count = max(self._history().values()) + 1
        ticks = [random_frame(self._features, self._generator) for _ in range(count)]
        noise = self._random_noise()
        stacked = {name: np.stack([tick[name] for tick in ticks]) for name in self._features}
        for case_ticks, case_noise, expected in (
            (self._case.frame, self._case.noise, self._case.actions),
            (stacked, noise, self._history_reference(ticks, noise)[0]),
        ):
            with torch.no_grad():
                actions = run(*self._history_inputs(case_ticks, case_noise)).cpu().numpy()
            # Not bit for bit: on a GPU the parts and the policy sum in other orders, so they drift apart by
            # float32 rounding, as a compiled program does, and are held to the export's tolerance.
            difference = float((np.abs(actions - expected) / self._case.action_scale).max())
            # Not `>`: a NaN difference must fail too.
            if not difference <= self.tolerance:
                raise ValueError(
                    f"The exported parts of the {self.policy.config.type} policy compute another chunk than "
                    f"the policy itself (largest difference {difference:.3g} of the actions' spread)."
                )

    @abc.abstractmethod
    def parts(self) -> Iterator[Part]:
        """Yield the parts, which take each key of the frame as its last frames, as `_history` counts them."""


class HistoryBatch(nn.Module):
    """The robot's last frames in, each stacked `[ticks, *shape]`; the batch `predict_action_chunk` reads out.

    Each tick goes through the preprocessor on its own, and its cameras are stacked under `observation.images`,
    as `select_action` does before it queues a frame. The queued keys are then stacked on the second axis, as
    `predict_action_chunk` stacks its queues; the other keys are those of the last tick.

    Args:
        preprocessor (`PolicyProcessorPipeline`):
            The policy's preprocessor.
        frame_names (`list[str]`):
            The keys of the robot's frame, in the order of the inputs.
        visual (`set[str]`):
            The keys that are uint8 HWC camera images.
        image_features (`list[str]`):
            The policy's cameras, in the order it stacks them.
        queued (`list[str]`):
            The keys the policy keeps the past values of.
        task (`str`, *optional*):
            The task, for a preprocessor that reads it.
    """

    def __init__(
        self,
        preprocessor: PolicyProcessorPipeline,
        frame_names: list[str],
        visual: set[str],
        image_features: list[str],
        queued: list[str],
        task: str | None,
    ):
        super().__init__()
        self.preprocessor = preprocessor
        self.frame_names = frame_names
        self.visual = visual
        self.image_features = image_features
        self.queued = queued
        self.task = task or ""

    def forward(self, *frames: Tensor) -> dict[str, Tensor]:
        # TensorRT holds uint8 tensors only at an engine's inputs, so a tick taken from a uint8 frame would
        # run outside the engine. The cast in `to_policy_input` then changes nothing.
        frames = tuple(
            x.float() if name in self.visual else x for name, x in zip(self.frame_names, frames, strict=True)
        )
        ticks = []
        for tick in range(frames[0].shape[0]):
            observation: dict[str, Any] = {
                name: to_policy_input(x[tick], name in self.visual)
                for name, x in zip(self.frame_names, frames, strict=True)
            }
            batch = dict(self.preprocessor({**observation, "task": self.task, "robot_type": ""}))
            batch.pop(ACTION, None)
            if self.image_features:
                batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.image_features], dim=-4)
            ticks.append(batch)
        return {**ticks[-1], **{key: torch.stack([b[key] for b in ticks], dim=1) for key in self.queued}}
