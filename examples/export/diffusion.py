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

"""The Diffusion Policy recipe: one action chunk of LeRobot's own Diffusion Policy, as three programs.

The first encodes the robot's last `n_obs_steps` frames, which the rollout keeps for it, once per chunk.
The second runs the U-Net and one step of the noise scheduler, once per scheduler step. The last turns the
denoised actions into robot units. The scheduler's random draws are the noise input: the starting sample,
then, for DDPM, the noise each step adds, so the programs themselves are deterministic.
"""

import argparse
from collections.abc import Iterator

import numpy as np
import torch
from common import NOISE, Part, device, frame_from_policy, processors, write_folder
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

TOLERANCE = 0.5
# Seeds the scheduler's draws for the test case, so the PyTorch policy draws the same noise as the programs.
SEED = 0


def scheduler_draws(count: int, config) -> list[Tensor]:
    """The first `count` noisy-action draws the policy makes after `SEED`, one call each, as it makes them."""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    shape = (config.horizon, config.action_feature.shape[0])
    return [torch.randn(shape, device=device()) for _ in range(count)]


class DiffusionObservation(nn.Module):
    """The last `n_obs_steps` frames in, oldest first, the U-Net's conditioning and the first noisy actions out."""

    def __init__(self, policy: DiffusionPolicy, preprocessor: PolicyProcessorPipeline):
        super().__init__()
        self.model = policy.diffusion
        self.preprocessor = preprocessor
        self.frame_names = list(policy.config.input_features)
        self.image_names = list(policy.config.image_features)

    def forward(self, *inputs: Tensor) -> tuple[Tensor, Tensor]:
        *frames, noise = inputs
        # The frames of each step stand in the batch dimension, which the preprocessor's steps keep.
        observation = {
            name: (x.float() / 255).permute(0, 3, 1, 2) if name in self.image_names else x
            for name, x in zip(self.frame_names, frames, strict=True)
        }
        observation = self.preprocessor(observation)
        batch = {OBS_STATE: observation[OBS_STATE][None]}
        if self.image_names:
            images = torch.stack([observation[name] for name in self.image_names], dim=-4)
            batch[OBS_IMAGES] = images[None]
        return self.model._prepare_global_conditioning(batch), noise[0]


class DiffusionStep(nn.Module):
    """`per_program` scheduler steps: the U-Net's prediction, then the scheduler's update, from tables indexed
    by timestep.

    The tables hold each timestep's coefficients of the scheduler's own update, so the program needs no
    scheduler object and no random generator. Several steps per program means fewer program calls per chunk.
    """

    def __init__(self, model: DiffusionModel, per_program: int = 1):
        super().__init__()
        self.unet = model.unet
        scheduler = model.noise_scheduler
        config = scheduler.config
        if config.prediction_type != "epsilon":
            raise SystemExit(
                f"Only epsilon-predicting checkpoints are supported, not {config.prediction_type!r}."
            )
        self.clip = config.clip_sample_range if config.clip_sample else None
        scheduler.set_timesteps(model.num_inference_steps)
        timesteps = scheduler.timesteps.tolist()
        alphas = scheduler.alphas_cumprod.double()
        size = config.num_train_timesteps
        tables = {name: torch.zeros(size, dtype=torch.float64) for name in ("signal", "x0", "x", "eps", "z")}
        noise_row = torch.zeros(size, dtype=torch.long)
        ddpm = type(scheduler).__name__ == "DDPMScheduler"
        if ddpm and config.variance_type != "fixed_small":
            raise SystemExit(
                f"Only the DDPM variance 'fixed_small' is supported, not {config.variance_type!r}."
            )
        for step, t in enumerate(timesteps):
            prev = t - size // model.num_inference_steps
            alpha = alphas[t]
            if prev >= 0:
                alpha_prev = alphas[prev]
            else:
                alpha_prev = torch.tensor(
                    1.0 if ddpm else float(scheduler.final_alpha_cumprod), dtype=torch.float64
                )
            tables["signal"][t] = alpha.sqrt()
            if ddpm:
                beta = 1 - alpha / alpha_prev
                tables["x0"][t] = alpha_prev.sqrt() * beta / (1 - alpha)
                tables["x"][t] = (1 - beta).sqrt() * (1 - alpha_prev) / (1 - alpha)
                if t > 0:
                    variance = ((1 - alpha_prev) / (1 - alpha) * beta).clamp(min=1e-20)
                    tables["z"][t] = variance.sqrt()
            else:
                # DDIM with eta 0, as LeRobot calls it: no noise after the starting sample.
                tables["x0"][t] = alpha_prev.sqrt()
                tables["eps"][t] = (1 - alpha_prev).sqrt()
            # Row 0 of the noise is the starting sample; each DDPM step but the last adds the next row.
            noise_row[t] = step + 1 if ddpm and t > 0 else 0
        for name, table in tables.items():
            self.register_buffer(name, table.float())
        self.register_buffer("noise_row", noise_row)
        if len(timesteps) % per_program:
            raise SystemExit(f"--steps_per_program must divide the {len(timesteps)} scheduler steps.")
        self.per_program = per_program
        self.dtype = torch.float32
        self.stride = size // model.num_inference_steps
        # The timestep each program call starts from.
        self.timesteps = timesteps[::per_program]

    def noise_rows(self) -> int:
        return int(self.noise_row.max()) + 1

    def forward(self, x_t: Tensor, timestep: Tensor, global_cond: Tensor, noise: Tensor) -> Tensor:
        for step in range(self.per_program):
            x_t = self.update(x_t, timestep.long() - step * self.stride, global_cond, noise)
        return x_t

    def update(self, x_t: Tensor, t: Tensor, global_cond: Tensor, noise: Tensor) -> Tensor:
        with torch.autocast(x_t.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            eps = self.unet(x_t[None], t, global_cond=global_cond)[0].float()
        signal = self.signal[t]
        x0 = (x_t - (1 - signal**2).sqrt() * eps) / signal
        if self.clip is not None:
            x0 = x0.clamp(-self.clip, self.clip)
        z = noise[self.noise_row[t]][0]
        return self.x0[t] * x0 + self.x[t] * x_t + self.eps[t] * eps + self.z[t] * z


class DiffusionActions(nn.Module):
    """The denoised actions in, the `n_action_steps` from the current frame on, in robot units."""

    def __init__(self, policy: DiffusionPolicy, postprocessor: PolicyProcessorPipeline):
        super().__init__()
        self.start = policy.config.n_obs_steps - 1
        self.end = self.start + policy.config.n_action_steps
        self.postprocessor = postprocessor

    def forward(self, x_t: Tensor) -> Tensor:
        return self.postprocessor(x_t[None, self.start : self.end])[0]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--steps_per_program", type=int, default=1, help="Scheduler steps each denoising program runs."
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Run the U-Net under this autocast dtype.",
    )


def part_names(args: argparse.Namespace) -> list[str]:
    return ["observation", "step", "actions"]


class Exporter:
    """Loads a trained Diffusion Policy checkpoint, and writes the folder around its compiled programs."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.policy = DiffusionPolicy.from_pretrained(args.policy_path).to(device()).eval()
        config = self.policy.config
        self.step = DiffusionStep(self.policy.diffusion, args.steps_per_program).to(device())
        frame = frame_from_policy(config)
        # The test case starts an episode, where the history holds copies of the first frame.
        self.frame = {name: np.stack([x] * config.n_obs_steps) for name, x in frame.items()}
        self.noise = torch.stack(scheduler_draws(self.step.noise_rows(), config))
        self.expected_actions = self.rollout_actions(frame)
        # After the test case, which the float32 policy computes, as `lerobot-rollout` would run it.
        self.step.dtype = getattr(torch, args.dtype)

    def parts(self) -> Iterator[Part]:
        preprocessor, postprocessor = processors(self.policy.config, self.args.policy_path)
        observation = DiffusionObservation(self.policy, preprocessor)
        inputs = (*(torch.from_numpy(x).to(device()) for x in self.frame.values()), self.noise)
        yield Part("observation", observation, inputs, [*self.frame, NOISE], ["global_cond", "x_t"])
        with torch.no_grad():
            global_cond, x_t = observation(*inputs)
        timestep = torch.tensor([float(self.step.timesteps[0])], device=device())
        yield Part(
            "step",
            self.step,
            (x_t, timestep, global_cond, self.noise),
            ["x_t", "timestep", "global_cond", NOISE],
            ["x_next"],
        )
        yield Part("actions", DiffusionActions(self.policy, postprocessor), (x_t,), ["x_t"], [ACTION])

    def rollout_actions(self, frame: dict[str, np.ndarray]) -> np.ndarray:
        """The chunk the PyTorch policy plays for the test frame, drawing the same noise as the programs.

        The policy draws its starting sample and then each DDPM step's noise from the default generator, in
        that order, so seeding it as `self.noise` was seeded reproduces those draws.
        """
        preprocessor, postprocessor = processors(self.policy.config, self.args.policy_path)
        self.policy.reset()
        observation = {name: torch.from_numpy(x).to(device()).unsqueeze(0) for name, x in frame.items()}
        for name in self.policy.config.image_features:
            observation[name] = (observation[name].float() / 255).permute(0, 3, 1, 2)
        (first,) = scheduler_draws(1, self.policy.config)
        with torch.inference_mode():
            batch = preprocessor(observation)
            actions = [self.policy.select_action(batch, noise=first[None])]
            actions += [
                self.policy.select_action(batch) for _ in range(self.policy.config.n_action_steps - 1)
            ]
            actions = postprocessor(torch.stack(actions, dim=1))
        return actions[0].cpu().numpy()

    def write(self, programs: dict[str, dict]) -> None:
        write_folder(
            self.args,
            self.policy.config,
            self.frame,
            self.noise,
            self.expected_actions,
            programs,
            n_obs_steps=self.policy.config.n_obs_steps,
            steps=["step"],
            timesteps=[float(t) for t in self.step.timesteps],
        )
