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

"""The Long-WAM export recipe: one program that reads the last frames of each camera, as Long-WAM keeps them.

Long-WAM keeps every control tick's camera frame and predicts a chunk from the last `past_obs_steps + 1` of them,
every `video_stride`-th one, encoded by its video VAE. A bundle keeps those frames (`RunPlan.history`), so the
program takes each camera as its last `past_obs_steps + 1` frames and the state as the current one.

The program holds the task's text context, which Long-WAM's T5 encoder computes once per task: the encoder is far
larger than the rest of the policy. It also holds the starting noise, which
Long-WAM draws from a generator it seeds with `inference_seed` for each chunk.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor

from lerobot.export.recipe import BFLOAT16_STEPS_TOLERANCE, Part, random_frame, to_policy_input
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STATE

from ..common.export_history import HistoryChunkRecipe
from ..common.export_utils import PolicyPart
from .modeling_longwam import LongWAMPolicy

if TYPE_CHECKING:
    from lerobot.export.configs import ExportConfig

    from .configuration_longwam import LongWAMConfig


class LongWAMExportRecipe(HistoryChunkRecipe):
    """Long-WAM's rollout as one program: the cameras' last frames and the state in, the action chunk out."""

    policy: LongWAMPolicy
    # 20 bfloat16 action denoising steps, after 4 that imagine the future video.
    bfloat16_tolerance = BFLOAT16_STEPS_TOLERANCE

    @classmethod
    def load(cls, cfg: ExportConfig) -> LongWAMExportRecipe:
        """Refuse a checkpoint this recipe cannot export before any weight loads, then load the policy."""
        if cast("LongWAMConfig", cfg.policy).inference_seed is None:
            raise ValueError(
                "This Long-WAM checkpoint draws new noise for each chunk (inference_seed is null), which its export "
                "does not do yet. Export it with --policy.inference_seed=1, as the released checkpoints run."
            )
        return cast(LongWAMExportRecipe, super().load(cfg))

    def _history(self) -> dict[str, int]:
        return dict.fromkeys(sorted(self._visual), self.policy.config.past_obs_steps + 1)

    def _random_ticks(self) -> list[dict[str, np.ndarray]]:
        # Enough ticks that Long-WAM reads three of them, every `video_stride`-th, and fewer than it keeps, so
        # that the first stands in for the ones not seen yet.
        config = self.policy.config
        count = max(min(2 * config.video_stride + 1, config.past_obs_steps), 2)
        return [random_frame(self._features, self._generator) for _ in range(count)]

    def _history_inputs(self, ticks: dict[str, np.ndarray], noise: np.ndarray | None) -> tuple[Tensor, ...]:
        """Each camera's last frames, the first standing in for any not seen yet, and the current state."""
        size = self.policy.config.past_obs_steps + 1
        inputs = []
        for name, value in ticks.items():
            if name in self._visual:
                value = np.stack([value[0]] * max(size - len(value), 0) + list(value[-size:]))
            else:
                value = value[-1]
            inputs.append(torch.from_numpy(value).to(self._device))
        return tuple(inputs)

    def _history_reference(
        self, ticks: list[dict[str, np.ndarray]], noise: np.ndarray | None
    ) -> tuple[np.ndarray, list[tuple[int, ...]]]:
        """The actions Long-WAM plays at the last of `ticks`, having kept the frames of the ticks before it."""
        self._reset()
        with torch.no_grad():
            batches = [self.preprocessor(self._observation(frame, self.task)) for frame in ticks]
            for batch in batches[:-1]:
                self.policy._observe(batch)
            predicted = self.policy.predict_action_chunk(batches[-1])[:, : self._steps]
            actions = self.postprocessor(predicted)
        return actions[0].float().cpu().numpy(), []

    def parts(self) -> Iterator[Part]:
        """The one part, after checking that it computes the policy's chunk on two sets of frames."""
        policy = self.policy
        inputs = self._history_inputs(self._case.frame, None)
        last = {name: value[-1] for name, value in self._case.frame.items()}
        with torch.no_grad():
            text = policy._text(self.preprocessor(self._observation(last, self.task)))
            module = _LongWAMChunk(
                policy, self.preprocessor, list(self._features), self._visual, self.postprocessor, self.task
            )
            clean, _ = module.encode(*inputs)
            # Long-WAM seeds its generator for each chunk, so every chunk starts from this noise.
            module.hold(text, *policy._initial_noise(clean), self._steps)
        self._check_chain(module)
        yield Part("model", module, inputs, list(self._features), [ACTION])


class _LongWAMChunk(PolicyPart):
    """The cameras' last frames and the current state in; the actions to play, in robot units, out."""

    def __init__(
        self,
        policy: LongWAMPolicy,
        preprocessor: PolicyProcessorPipeline,
        frame_names: list[str],
        visual: set[str],
        postprocessor: PolicyProcessorPipeline,
        task: str | None,
    ):
        # Not the text encoder nor the video decoder, which the program never runs.
        super().__init__(policy, ("dit", "vae.encoder", "vae.quant_conv", "proprio_encoder"))
        self.preprocessor = preprocessor
        self.frame_names = frame_names
        self.visual = visual
        self.postprocessor = postprocessor
        self.task = task

    def hold(self, text: Tensor, future: Tensor, action: Tensor, execution_steps: int) -> None:
        """Hold the task's text context, the starting noise, and how many actions of a chunk rollout plays."""
        self.register_buffer("text", text.detach().clone())
        self.register_buffer("future", future.detach().clone())
        self.register_buffer("action", action.detach().clone())
        self.execution_steps = execution_steps

    def encode(self, *inputs: Tensor) -> tuple[Tensor, dict[str, Any]]:
        """The encoded past video, every `video_stride`-th frame as Long-WAM reads it, and the last tick's batch."""
        policy, config = self.policy, self.policy.config
        frames = []
        for tick in range(0, config.past_obs_steps + 1, config.video_stride):
            observation: dict[str, Any] = {
                name: to_policy_input(x[tick], True) if name in self.visual else to_policy_input(x, False)
                for name, x in zip(self.frame_names, inputs, strict=True)
            }
            batch = self.preprocessor({**observation, "task": self.task, "robot_type": ""})
            frames.append(policy._images(batch))
        return policy._encode_video(torch.stack(frames, dim=2)), batch

    def forward(self, *inputs: Tensor) -> Tensor:
        policy = self.policy
        clean, batch = self.encode(*inputs)
        actions = policy._denoise(
            clean, policy._with_state(self.text, batch[OBS_STATE]), self.future, self.action
        )
        return self.postprocessor(actions[:, : self.execution_steps])[0].float()
