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

"""Export Cosmos3 as a chain of programs: an encoder, a solver step run once per timestep, and the actions.

One action chunk runs as these programs, which pass their values to each other by name:

::

    encode        the frame and the flat noise -> the starting state: the noisy video latents and actions, flat
    step x30      the state and the solver's history -> the next state and history
    actions       the denoised state -> the actions to play, in robot units

Cosmos3 denoises the latents of a future video and an action chunk together. The two noises it draws for a chunk
are one flat noise input. Its prompt holds the task and the canvas size, not the frame, so the programs hold the
prompt's token ids and the M-RoPE positions and sequence plan the policy computes from its latent grid on the host.
Each step of its UniPC solver is linear in the state, the network's velocity, the solver's earlier model outputs and
its last sample, with coefficients that depend only on the step. The recipe measures them once on the policy's own
scheduler, and the step runs pass the solver's history to each other. The video is never decoded, so the VAE
decoder is left out, and only the guidance-free sampling exports.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from copy import copy
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.bundle.manifest import NOISE, ProgramRun
from lerobot.configs import FeatureType
from lerobot.export.recipe import (
    BFLOAT16_STEPS_TOLERANCE,
    ChainRecipe,
    Part,
    ReplayDraws,
    RunPlan,
    TestCase,
    dataset_metadata,
    export_task,
    measure_action_scale,
    random_frame,
    repeated_runs,
    robot_features,
    to_policy_input,
)
from lerobot.policies import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline

from .modeling_cosmos3 import Cosmos3Policy, DomainAwareLinear, FlowState, SequencePlan
from .processor_cosmos3 import (
    Cosmos3IdleFrameProcessorStep,
    Cosmos3PromptProcessorStep,
    Cosmos3TokenizerProcessorStep,
)

if TYPE_CHECKING:
    from diffusers import UniPCMultistepScheduler

    from lerobot.export.configs import ExportConfig

# Elements of each random trajectory the solver's coefficients are measured on.
PROBE_ELEMENTS = 16


def _plan_buffers(module: nn.Module, plan: SequencePlan) -> None:
    module.register_buffer("video_condition_mask", plan.video_condition_mask)
    module.register_buffer("action_condition_mask", plan.action_condition_mask)
    module.register_buffer("action_start_frame_offset", plan.action_start_frame_offset)


def _plan(module: nn.Module) -> SequencePlan:
    return SequencePlan(
        cast(Tensor, module.video_condition_mask),
        cast(Tensor, module.action_condition_mask),
        cast(Tensor, module.action_start_frame_offset),
    )


class Cosmos3Encode(nn.Module):
    """The frame and the flat noise in, the flat starting state out: the noisy video latents, then the actions.

    It runs what `OmniMoTModel.generate` runs before its solver, with the latent grid and the sequence plan held.
    """

    def __init__(
        self,
        policy: Cosmos3Policy,
        preprocessor: PolicyProcessorPipeline,
        frame_names: list[str],
        visual: set[str],
        plan: SequencePlan,
        latent_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
    ):
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.frame_names = frame_names
        self.visual = visual
        self.latent_shape = latent_shape
        self.action_shape = action_shape
        _plan_buffers(self, plan)

    def forward(self, *inputs: Tensor) -> Tensor:
        *frame, noise = inputs
        batch = self.preprocessor(
            {
                name: to_policy_input(x, name in self.visual)
                for name, x in zip(self.frame_names, frame, strict=True)
            }
        )
        model = self.policy.model
        vae = model.network.vae
        # `Cosmos3VFMNetwork.encode_video` with the grid held: it reads the grid back to Python.
        latents = vae.encode(batch["pixel_values_videos_gen"].to(vae.dtype)).latent_dist.mode()
        mean = latents.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        inv_std = 1.0 / latents.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        _, frames, height, width = self.latent_shape
        latents = ((latents - mean) * inv_std)[..., :height, :width].float()
        latents = nn.functional.pad(latents, (0, 0, 0, 0, 0, frames - latents.shape[2]))
        placeholder = latents.new_zeros(1, self.policy.config.chunk_size, self.policy.config.max_action_dim)
        actions = model.pack_actions(placeholder, self.policy.prepare_state(batch))
        video_noise, action_noise = noise.split([math.prod(self.latent_shape), math.prod(self.action_shape)])
        ones = actions.new_ones(1)
        noisy = model.add_noise(
            FlowState(latents, actions),
            FlowState(video_noise.view(1, *self.latent_shape), action_noise.view(1, *self.action_shape)),
            ones,
            ones,
            _plan(self),
        )
        return torch.cat([noisy.video_latents.flatten(), noisy.actions.flatten()])


class Cosmos3Step(nn.Module):
    """One step of `OmniMoTModel.sample`: the flat state, the solver's history, the timestep and the step's
    coefficients in, the next state and history out.

    The coefficients map the state, the network's velocity and the history to the next state and history.
    """

    def __init__(
        self,
        policy: Cosmos3Policy,
        sequence: dict[str, Tensor],
        plan: SequencePlan,
        latent_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
    ):
        super().__init__()
        self.model = policy.model
        self.latent_shape = latent_shape
        self.action_shape = action_shape
        for name, value in sequence.items():
            self.register_buffer(name, value)
        _plan_buffers(self, plan)

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        state, *history, timestep, coefficients = inputs
        video, actions = state.split([math.prod(self.latent_shape), math.prod(self.action_shape)])
        plan = _plan(self)
        velocity = self.model.mask_velocity(
            self.velocity(
                video.view(1, *self.latent_shape), actions.view(1, *self.action_shape), timestep, plan
            ),
            plan,
        )
        flat_velocity = torch.cat([velocity.video_latents.flatten(), velocity.actions.flatten()])
        terms = torch.stack([state, flat_velocity, *history])
        # Products summed one by one, not a matrix product, which a backend may round to TF32.
        return tuple((row[:, None] * terms).sum(0) for row in coefficients.unbind(0))

    def velocity(self, video: Tensor, actions: Tensor, timestep: Tensor, plan: SequencePlan) -> FlowState:
        """`OmniMoTModel.predict_velocity` on the held prompt, with the action projections of the held domain."""
        network = self.model.network
        domain = network.config.domain_id
        video_embeds = network.embed_video(
            video, timestep[:, None].expand(-1, video.shape[2]), plan.video_condition_mask
        )
        action_embeds = network.action_modality_embed(domain_projection(network.action2llm, actions, domain))
        action_times = network.embed_timesteps(timestep[:, None].expand(-1, actions.shape[1]))
        action_times = action_times.to(action_embeds.dtype).masked_fill(
            plan.action_condition_mask.unsqueeze(-1), 0.0
        )
        outputs = network.mot_backbone.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            inputs_embeds_gen=network.merge_gen_stream(video_embeds, action_embeds + action_times),
        )
        video_hidden, action_hidden = network.split_gen_stream(
            outputs.last_hidden_state_gen, video_embeds.shape[1]
        )
        video_velocity = network.unpatchify(network.llm2vae(video_hidden), video.shape)
        action_velocity = domain_projection(network.llm2action, cast(Tensor, action_hidden), domain)
        return FlowState(video_velocity.float(), action_velocity.float())


def domain_projection(layer: DomainAwareLinear, x: Tensor, domain: int) -> Tensor:
    """`DomainAwareLinear` for one domain. Its grouped matrix product sorts and counts the rows by domain at run
    time, which the backends cannot compile, while the domain of an export is fixed."""
    weight = layer.fc.weight[domain].view(layer.input_size, layer.output_size)
    return x.to(weight.dtype) @ weight + layer.bias.weight[domain]


class Cosmos3Actions(nn.Module):
    """The denoised flat state in, the actions to play out, in robot units."""

    def __init__(
        self, policy: Cosmos3Policy, postprocessor: PolicyProcessorPipeline, action_shape: tuple[int, ...]
    ):
        super().__init__()
        self.policy = policy
        self.postprocessor = postprocessor
        self.action_shape = action_shape
        self.execution_steps = policy.chunk_inference_spec().execution_steps

    def forward(self, state: Tensor) -> Tensor:
        actions = state[state.shape[0] - math.prod(self.action_shape) :].view(1, *self.action_shape)
        actions = self.policy.model.unpack_actions(actions)[:, :, : self.policy.config.original_action_dim]
        return self.postprocessor(actions)[0, : self.execution_steps].float()


def solver_coefficients(
    make_scheduler: Callable[[], UniPCMultistepScheduler],
) -> tuple[list[Tensor], list[Tensor]]:
    """Each step's timestep, and the float64 matrix that maps `(state, velocity, *history)` to `(next state, *history)`.

    The history is the scheduler's model outputs, then its last sample. Both are measured on new schedulers from
    `make_scheduler`, on random trajectories in float64, and refused unless the matrices reproduce another
    trajectory exactly.
    """
    trajectories = []
    for seed in (0, 1):
        generator = torch.Generator().manual_seed(seed)
        scheduler = make_scheduler()
        # Shaped like a batch of latents, [batch, channels, elements], as the scheduler's thresholding needs.
        state = torch.randn(1, PROBE_ELEMENTS, 1, generator=generator, dtype=torch.float64)
        steps = []
        for timestep in scheduler.timesteps:
            zeros = torch.zeros_like(state)
            history = [zeros if output is None else output for output in scheduler.model_outputs]
            history.append(zeros if scheduler.last_sample is None else scheduler.last_sample)
            velocity = torch.randn(state.shape, generator=generator, dtype=torch.float64)
            following = scheduler.step(velocity, timestep, state).prev_sample
            outputs = [
                following,
                *(zeros if o is None else o for o in scheduler.model_outputs),
                scheduler.last_sample,
            ]
            steps.append(
                (timestep, torch.cat([state, velocity, *history]).flatten(1), torch.cat(outputs).flatten(1))
            )
            state = following
        trajectories.append(steps)

    timesteps, matrices = [], []
    for timestep, terms, outputs in trajectories[0]:
        solution = np.linalg.lstsq(terms.T.numpy(), outputs.T.numpy(), rcond=None)[0]
        timesteps.append(torch.tensor([float(timestep)], dtype=torch.float32))
        matrices.append(torch.from_numpy(solution.T))
    for (_, terms, outputs), matrix in zip(trajectories[1], matrices, strict=True):
        error = float((matrix @ terms - outputs).abs().max() / outputs.abs().max().clamp(min=1e-12))
        if not error < 1e-9:
            raise ValueError(
                f"Cosmos3's solver step is not linear in its inputs ({error:.1e} from the measured map), so its "
                "steps cannot be exported as one program. Its scheduler's thresholding or clipping can cause this."
            )
    return timesteps, matrices


class Cosmos3ExportRecipe(ChainRecipe):
    """Cosmos3 as a chain of programs: see this module's documentation.

    Args:
        policy (`Cosmos3Policy`):
            The trained policy, in eval mode on the backend's device.
        preprocessor (`PolicyProcessorPipeline`):
            Its saved preprocessor.
        postprocessor (`PolicyProcessorPipeline`):
            Its saved postprocessor.
        cfg (`ExportConfig`):
            The `lerobot-export` configuration.
    """

    # The solver runs 30 steps by default.
    bfloat16_tolerance = BFLOAT16_STEPS_TOLERANCE
    policy_name = "Cosmos3"
    text_steps = (Cosmos3IdleFrameProcessorStep, Cosmos3PromptProcessorStep, Cosmos3TokenizerProcessorStep)
    policy: Cosmos3Policy
    task: str

    def __init__(
        self,
        policy: Cosmos3Policy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        cfg: ExportConfig,
    ) -> None:  # noqa: D107
        self._device = torch.device(cfg.backend.device)
        policy.to(self._device)
        super().__init__(policy, preprocessor, postprocessor, cfg)
        config = policy.config
        if config.guidance_scale != 1.0:
            raise ValueError(
                "Cosmos3 exports without classifier-free guidance: use a checkpoint with guidance_scale=1, not "
                f"{config.guidance_scale}."
            )
        if config.n_obs_steps != 1:
            raise ValueError(
                "Cosmos3 exports a checkpoint that reads the current frame only (n_obs_steps=1)."
            )
        self._generator = torch.Generator().manual_seed(cfg.seed)
        self._metadata = dataset_metadata(cfg)
        self._features = robot_features(config, preprocessor, self._metadata)
        self._visual = {name for name, f in self._features.items() if f.type is FeatureType.VISUAL}
        self.config = copy(config)
        self.config.input_features = self._features
        self.task = export_task(cfg, self._metadata, config.type)
        self._frame = random_frame(self._features, self._generator)
        self._sequence, self._plan, self._latent_shape, self._action_shape = self._prompt()
        size = math.prod(self._latent_shape) + math.prod(self._action_shape)
        self._noise = torch.randn(size, generator=self._generator)
        self._timesteps, matrices = solver_coefficients(policy.model.make_scheduler)
        self._coefficients = [matrix.float() for matrix in matrices]
        self._runs: list[ProgramRun] = []
        self._constants: dict[str, np.ndarray] = {}
        self._actions: np.ndarray | None = None

    def _prompt(self) -> tuple[dict[str, Tensor], SequencePlan, tuple[int, ...], tuple[int, ...]]:
        """The prompt's token ids, mask and M-RoPE positions, the sequence plan, and the latents' and actions' shapes.

        They are computed from two random frames, and refused unless both give the same.
        """
        model = self.policy.model
        second = random_frame(self._features, self._generator)
        batches = []
        with torch.no_grad():
            for frame in (self._frame, second):
                self.preprocessor.reset()
                batches.append(
                    self.preprocessor(prepare_observation_for_inference(dict(frame), self._device, self.task))
                )
        first, other = batches
        for name in ("input_ids", "attention_mask", "video_grid_thw_gen", "second_per_grid_ts_gen"):
            if not torch.equal(first[name], other[name]):
                raise ValueError(
                    f"Cosmos3's {name} differs between two frames, so the programs cannot hold it."
                )
        grid = first["video_grid_thw_gen"]
        num_actions = self.policy.config.chunk_size + int(self.policy.config.use_state)
        plan = model.build_sequence_plan(grid, num_actions)
        position_ids = model.network.get_rope_index(
            first["input_ids"],
            grid,
            first["second_per_grid_ts_gen"],
            num_actions=num_actions,
            action_start_frame_offset=plan.action_start_frame_offset,
            attention_mask=first["attention_mask"],
            mm_token_type_ids=first.get("mm_token_type_ids"),
        )
        frames, height, width = (int(n) for n in grid[0].tolist())
        latent_shape = (model.network.vae.config.z_dim, frames, height, width)
        action_shape = (num_actions, self.policy.config.max_action_dim)
        sequence = {
            "input_ids": first["input_ids"],
            "attention_mask": first["attention_mask"],
            "position_ids": position_ids,
        }
        return sequence, plan, latent_shape, action_shape

    def parts(self) -> Iterator[Part]:
        """The chain's parts, each run in PyTorch to make the next part's inputs."""
        frame_names = list(self._features)
        frame = tuple(torch.from_numpy(x).to(self._device) for x in self._frame.values())
        noise = self._noise.to(self._device)
        encode = Cosmos3Encode(
            self.policy,
            self._tensor_preprocessor(),
            frame_names,
            self._visual,
            self._plan,
            self._latent_shape,
            self._action_shape,
        )
        with torch.no_grad():
            state = encode(*frame, noise)
        yield self._part("encode", encode, (*frame, noise), [*frame_names, NOISE], ["state"])

        step = Cosmos3Step(self.policy, self._sequence, self._plan, self._latent_shape, self._action_shape)
        depth = self._coefficients[0].shape[0] - 1
        # Separate tensors: inputs traced from one tensor would become one input of the program.
        history = [torch.zeros_like(state) for _ in range(depth)]
        example = None
        with torch.no_grad():
            for timestep, coefficients in zip(self._timesteps, self._coefficients, strict=True):
                inputs = (state, *history, timestep.to(self._device), coefficients.to(self._device))
                if example is None:
                    example = inputs
                state, *history = step(*inputs)
        history_names = [f"history_{j}" for j in range(depth)]
        step_names = ["state", *history_names, "timestep", "coefficients"]
        step_outputs = ["state", *history_names]
        yield Part("step", step, cast(tuple[Tensor, ...], example), step_names, step_outputs)
        step_constants = [
            {"timestep": timestep.numpy(), "coefficients": coefficients.numpy()}
            for timestep, coefficients in zip(self._timesteps, self._coefficients, strict=True)
        ]
        step_runs, self._constants = repeated_runs(
            "step", step_names, step_outputs, step_constants, start=dict.fromkeys(history_names, "zeros")
        )
        self._runs += step_runs

        actions = Cosmos3Actions(self.policy, self.postprocessor, self._action_shape)
        with torch.no_grad():
            self._actions = actions(state).cpu().numpy()
        yield self._part("actions", actions, (state,), ["state"], ["action"])

    def test_case(self) -> TestCase:
        """The random frame and noise, and the actions the parts compute for them, checked against the policy."""
        if self._actions is None:
            raise RuntimeError("The test case is made by `parts`, which has not run yet.")
        if not np.isfinite(self._actions).all():
            raise ValueError(
                "The chained parts compute actions that are not finite, so they cannot be checked."
            )
        width = self.policy.config.original_action_dim
        action_scale = measure_action_scale(
            lambda value: self.postprocessor(torch.full((1, 1, width), value, device=self._device))[0, 0]
        )
        case = TestCase(self._frame, self._noise.numpy(), self._actions, action_scale)
        self._check_against_policy(case)
        return case

    def plan(self) -> RunPlan:
        """The parts in order with the step run once per timestep, and the timesteps and coefficients."""
        size = math.prod(self._latent_shape) + math.prod(self._action_shape)
        return RunPlan(
            runs=self._runs, constants={"zeros": np.zeros(size, dtype=np.float32), **self._constants}
        )

    def _policy_actions(self, case: TestCase) -> np.ndarray:
        """The policy's own actions for the test case's frame and noise."""
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.policy.reset()
        observation = prepare_observation_for_inference(dict(case.frame), self._device, self.task)
        assert case.noise is not None
        video, actions = (
            torch.from_numpy(case.noise)
            .to(self._device)
            .split([math.prod(self._latent_shape), math.prod(self._action_shape)])
        )
        draws = [video.view(1, *self._latent_shape), actions.view(1, *self._action_shape)]
        with torch.no_grad(), ReplayDraws(draws):
            predicted = self.policy.predict_action_chunk(self.preprocessor(observation))
        with torch.no_grad():
            return self.postprocessor(predicted)[0, : self.policy.config.n_action_steps].float().cpu().numpy()


Cosmos3NanoExportRecipe = Cosmos3ExportRecipe
Cosmos3EdgeExportRecipe = Cosmos3ExportRecipe
