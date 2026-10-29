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

"""Export a policy that reads past frames and denoises its action chunk, as small programs run in a loop.

One action chunk runs as these programs, which pass their values to each other by name:

::

    encoder     the last frames and the noise -> the conditioning and the first sample
    step x K    the sample, the conditioning and this step's constants -> the next sample
    actions     the last sample -> the actions in robot units

Unrolled into one graph, K runs of the denoising network hold K copies of its code, which takes too much
memory to build on a small board. A sampler's arithmetic depends on the timestep only through a few numbers,
which are computed once, from the policy's own scheduler, and given to each run of `step` as constants.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.export.recipe import Part, ProgramCall, RunPlan
from lerobot.inference.bundle.manifest import NOISE
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION

from .export_history import HistoryBatch, HistoryChunkRecipe

DDPM = "ddpm"
DDIM = "ddim"
EULER = "euler"


@dataclass
class Schedule:
    """What each run of the denoising step takes, besides the sample and the conditioning.

    Args:
        kind (`str`):
            The sampler: `ddpm`, `ddim` or `euler`.
        timesteps (`list[np.ndarray]`):
            The timestep the network takes on each run.
        coefficients (`list[np.ndarray]`):
            The sampler's numbers on each run: for DDPM and DDIM `sqrt(1 - a_t)` and `sqrt(a_t)`, then
            DDPM's coefficients of the clean sample and of the sample and its noise's spread, or DDIM's
            `sqrt(a_prev)` and `sqrt(1 - a_prev)`; for Euler the time step.
        noise_index (`list[int]`, *optional*):
            For DDPM, the draw of the noise each run adds; the starting sample is draw 0.
        prediction (`str`):
            What the network predicts, `epsilon` or `sample`.
        clip (`float`, *optional*):
            The range the clean sample is clipped to.
    """

    kind: str
    timesteps: list[np.ndarray]
    coefficients: list[np.ndarray]
    noise_index: list[int] | None = None
    prediction: str = "epsilon"
    clip: float | None = None


def scheduler_schedule(scheduler: Any) -> Schedule:
    """The schedule of a diffusers DDPM or DDIM scheduler, with its timesteps already set.

    The numbers are computed the way the scheduler's `step` computes them, so the step program does the same
    arithmetic as the policy.
    """
    from diffusers import DDIMScheduler, DDPMScheduler

    config = scheduler.config
    if config.thresholding or config.prediction_type not in ("epsilon", "sample"):
        raise ValueError(
            "Only a scheduler predicting `epsilon` or `sample`, without thresholding, can be exported; this "
            f"one predicts {config.prediction_type!r} with thresholding {config.thresholding}."
        )
    clip = config.clip_sample_range if config.clip_sample else None
    timesteps, coefficients, noise_index = [], [], []
    if isinstance(scheduler, DDPMScheduler):
        if scheduler.variance_type != "fixed_small":
            raise ValueError(
                f"Only DDPM's `fixed_small` variance can be exported, not {scheduler.variance_type!r}."
            )
        draw = 1
        for t in scheduler.timesteps.tolist():
            prev_t = int(scheduler.previous_timestep(t))
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            current_alpha_t = alpha_prod_t / alpha_prod_t_prev
            current_beta_t = 1 - current_alpha_t
            # DDPM adds no noise at the last timestep, 0.
            spread = scheduler._get_variance(t) ** 0.5 if t > 0 else torch.zeros(())
            noise_index.append(draw if t > 0 else 0)
            draw += t > 0
            values = (
                beta_prod_t**0.5,
                alpha_prod_t**0.5,
                (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t,
                current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t,
                spread,
            )
            timesteps.append(np.array([t], np.int64))
            coefficients.append(torch.stack(values).float().numpy())
        return Schedule(DDPM, timesteps, coefficients, noise_index, config.prediction_type, clip)
    if isinstance(scheduler, DDIMScheduler):
        for t in scheduler.timesteps.tolist():
            prev_t = t - config.num_train_timesteps // scheduler.num_inference_steps
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.final_alpha_cumprod
            )
            beta_prod_t = 1 - alpha_prod_t
            roots = (
                beta_prod_t**0.5,
                alpha_prod_t**0.5,
                alpha_prod_t_prev**0.5,
                (1 - alpha_prod_t_prev) ** 0.5,
            )
            timesteps.append(np.array([t], np.int64))
            coefficients.append(torch.stack(roots).float().numpy())
        return Schedule(DDIM, timesteps, coefficients, None, config.prediction_type, clip)
    raise ValueError(f"A {type(scheduler).__name__} cannot be exported; DDPM and DDIM can.")


class DenoisingRecipe(HistoryChunkRecipe):
    """A policy that reads its last frames, encodes them once, and denoises its chunk with one network.

    The parts are `encoder`, `step` (run once per denoising step) and `actions`; see this module's
    documentation. A subclass gives the policy's encoder, its denoising network and its schedule.
    """

    @abc.abstractmethod
    def _encoder(self) -> nn.Module:
        """The module that turns the policy's batch into the conditioning of the denoising network."""

    @abc.abstractmethod
    def _network(self) -> nn.Module:
        """The denoising network, called as `network(sample, timestep, conditioning)`."""

    @abc.abstractmethod
    def _schedule(self) -> Schedule:
        """The sampler's schedule."""

    def parts(self) -> Iterator[Part]:
        """The three parts, after checking that their chain computes the policy's chunk on two sets of frames."""
        schedule = self._schedule()
        draws = 1 + sum(index > 0 for index in schedule.noise_index or [])
        if self._case.noise is None or len(self._case.noise) != draws:
            raise ValueError(
                f"The {self.policy.config.type} policy's sampler adds {draws} noises, but the policy drew "
                f"{0 if self._case.noise is None else len(self._case.noise)}."
            )
        config = self.policy.config
        start = config.n_obs_steps - 1
        encoder = _Encoder(self._history_batch(), self._encoder())
        step = _Step(self._network(), schedule)
        actions = _Actions(self.postprocessor, start, start + self._steps)

        def chain(*inputs: Tensor) -> Tensor:
            conditioning, sample = encoder(*inputs)
            for i in range(len(schedule.timesteps)):
                sample = step(sample, conditioning, *self._step_values(schedule, i, inputs[-1]))
            return actions(sample)

        self._check_chain(chain)
        inputs = self._history_inputs(self._case.frame, self._case.noise)
        with torch.no_grad():
            conditioning, sample = encoder(*inputs)
        frame_names = list(self._frame_shapes)
        yield Part("encoder", encoder, inputs, [*frame_names, NOISE], ["conditioning", "sample_0"])
        step_inputs = (sample, conditioning, *self._step_values(schedule, 0, inputs[-1]))
        yield Part("step", step, step_inputs, self._step_inputs(schedule, 0), ["sample_1"])
        yield Part("actions", actions, (sample,), ["sample"], [ACTION])

    def plan(self) -> RunPlan:
        """The encoder, the step once per denoising step with its constants, and the actions."""
        schedule = self._schedule()
        count = len(schedule.timesteps)
        calls = [ProgramCall("encoder", [*self._frame_shapes, NOISE], ["conditioning", "sample_0"])]
        calls += [
            ProgramCall("step", self._step_inputs(schedule, i), [f"sample_{i + 1}"]) for i in range(count)
        ]
        calls.append(ProgramCall("actions", [f"sample_{count}"], [ACTION]))
        constants: dict[str, np.ndarray] = {}
        for i in range(count):
            constants[f"timestep_{i}"] = schedule.timesteps[i]
            constants[f"coefficients_{i}"] = schedule.coefficients[i]
            if schedule.noise_index is not None:
                constants[f"noise_index_{i}"] = np.array([schedule.noise_index[i]], np.int64)
        return RunPlan(calls=calls, constants=constants, history=self._history())

    def _step_inputs(self, schedule: Schedule, i: int) -> list[str]:
        names = [f"sample_{i}", "conditioning", f"timestep_{i}", f"coefficients_{i}"]
        return [*names, NOISE, f"noise_index_{i}"] if schedule.noise_index is not None else names

    def _step_values(self, schedule: Schedule, i: int, noise: Tensor) -> tuple[Tensor, ...]:
        """What run `i` of the step takes after the sample and the conditioning."""
        values = [torch.from_numpy(schedule.timesteps[i]), torch.from_numpy(schedule.coefficients[i])]
        if schedule.noise_index is not None:
            values += [noise, torch.tensor([schedule.noise_index[i]])]
        return tuple(value.to(self._device) for value in values)


class _Encoder(nn.Module):
    def __init__(self, batch: HistoryBatch, encoder: nn.Module):
        super().__init__()
        self.batch = batch
        self.encoder = encoder

    def forward(self, *inputs: Tensor) -> tuple[Tensor, Tensor]:
        *frames, noise = inputs
        # The first draw of the noise is the starting sample.
        return self.encoder(self.batch(*frames)), noise[0].clone()


class _Step(nn.Module):
    """One denoising step, with the arithmetic of the scheduler's `step` in the same order."""

    def __init__(self, network: nn.Module, schedule: Schedule):
        super().__init__()
        self.network = network
        self.kind = schedule.kind
        self.prediction = schedule.prediction
        self.clip = schedule.clip

    def forward(
        self,
        sample: Tensor,
        conditioning: Tensor,
        timestep: Tensor,
        coefficients: Tensor,
        noise: Tensor | None = None,
        noise_index: Tensor | None = None,
    ) -> Tensor:
        output = self.network(sample, timestep, conditioning)
        if self.kind == EULER:
            return sample + coefficients[0] * output
        sqrt_beta, sqrt_alpha, first, second = coefficients[:4]
        if self.prediction == "epsilon":
            original, epsilon = (sample - sqrt_beta * output) / sqrt_alpha, output
        else:
            original = output
            epsilon = (sample - sqrt_alpha * original) / sqrt_beta
        if self.clip is not None:
            original = original.clamp(-self.clip, self.clip)
        if self.kind == DDIM:
            return first * original + second * epsilon
        if noise is None or noise_index is None:
            raise ValueError("A DDPM step takes the noise and the index of the draw it adds.")
        return first * original + second * sample + coefficients[4] * noise.index_select(0, noise_index)[0]


class _Actions(nn.Module):
    def __init__(self, postprocessor: PolicyProcessorPipeline, start: int, end: int):
        super().__init__()
        self.postprocessor = postprocessor
        self.start = start
        self.end = end

    def forward(self, sample: Tensor) -> Tensor:
        return self.postprocessor(sample[:, self.start : self.end])[0].float()
