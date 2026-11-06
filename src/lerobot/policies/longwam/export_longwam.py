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

Long-WAM keeps every control tick's camera frame and predicts a chunk from the last `past_obs_steps + 1` of
them, every `video_stride`-th one, encoded by its video VAE. A bundle keeps those frames (`RunPlan.history`),
so the program takes each camera as its last `past_obs_steps + 1` frames and the state as its last frame.

The program holds the task's text context, computed at export. It depends only on the task: the PyTorch policy
computes it once per episode with Long-WAM's T5 encoder, which holds about half of the policy's weights. The
program also holds the starting noise, which Long-WAM draws from a generator it seeds with `inference_seed`
for each chunk.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor

from lerobot.export.recipe import Part, random_frame, to_policy_input
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

    @classmethod
    def load(cls, cfg: ExportConfig) -> LongWAMExportRecipe:
        """Refuse a checkpoint this recipe cannot export before any weight loads, then load the policy."""
        if cast("LongWAMConfig", cfg.policy).inference_seed is None:
            raise ValueError(
                "This Long-WAM checkpoint draws new noise for each chunk (inference_seed is null), and its "
                "program holds a fixed starting noise. Export it with --policy.inference_seed=1, as the "
                "released checkpoints run."
            )
        return cast(LongWAMExportRecipe, super().load(cfg))

    def _history(self) -> dict[str, int]:
        count = self.policy.config.past_obs_steps + 1
        return {name: count if name in self._visual else 1 for name in self._features}

    def _random_ticks(self) -> list[dict[str, np.ndarray]]:
        # For any `past_obs_steps` but 0, enough ticks that Long-WAM reads three of them, every
        # `video_stride`-th, and fewer than it keeps, so that the first stands in for the ones not seen yet.
        count = 2 * self.policy.config.video_stride + 1
        return [random_frame(self._features, self._generator) for _ in range(count)]

    def _history_reference(
        self, ticks: list[dict[str, np.ndarray]], noise: np.ndarray | None
    ) -> tuple[np.ndarray, list[tuple[int, ...]]]:
        """The actions Long-WAM plays at the last of `ticks`, having kept the frames of the ticks before it.

        No shapes of random draws: the program holds the noise Long-WAM seeds for every chunk.
        """
        self._reset()
        with torch.no_grad():
            batches = [self.preprocessor(self._observation(frame, self.task)) for frame in ticks]
            for batch in batches[:-1]:
                self.policy._observe(batch)
            predicted = self.policy.predict_action_chunk(batches[-1])[:, : self._steps]
            self._predicted_width = predicted.shape[-1]
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
                policy,
                self.preprocessor,
                list(self._features),
                self._visual,
                self.postprocessor,
                self.task,
                self._steps,
            )
            clean, _ = module.encode(*inputs)
            # Long-WAM seeds its generator for each chunk, so every chunk starts from this noise.
            module.hold(text, *policy._initial_noise(clean))
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
        execution_steps: int,
    ):
        # Not the text encoder, nor the VAE's decoder and its post_quant_conv, which the program never runs.
        super().__init__(policy, ("dit", "vae.encoder", "vae.quant_conv", "proprio_encoder"))
        self.preprocessor = preprocessor
        self.frame_names = frame_names
        self.visual = visual
        self.postprocessor = postprocessor
        self.task = task or ""
        self.execution_steps = execution_steps

    def hold(self, text: Tensor, future: Tensor, action: Tensor) -> None:
        """Hold the task's text context and the starting noise."""
        self.register_buffer("text", text)
        self.register_buffer("future", future)
        self.register_buffer("action", action)

    def encode(self, *inputs: Tensor) -> tuple[Tensor, dict[str, Any]]:
        """The past video, encoded every `video_stride`-th frame as Long-WAM reads it, and the last batch."""
        policy, config = self.policy, self.policy.config
        frames = []
        for tick in range(0, config.past_obs_steps + 1, config.video_stride):
            observation: dict[str, Any] = {
                name: to_policy_input(x[tick], True) if name in self.visual else to_policy_input(x[-1], False)
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
