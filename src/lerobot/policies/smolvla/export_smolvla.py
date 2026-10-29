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

"""Export SmolVLA as a chain of small programs, so that it builds on a board with 8 GB of memory.

One action chunk runs as these programs, which pass their values to each other by name:

::

    embed           the frame -> the prefix's embeddings and masks
    language_model  the prefix -> the keys and values of every layer, stacked
    step x10        the padding mask, the keys and values, the noisy actions and a timestep -> the next ones
    actions         the denoised actions -> the actions in robot units

Compiled as one program, the denoising loop is unrolled ten times, and building that graph takes most of an
8 GB board's memory. Here the step is compiled once and run once per timestep, and each program is small enough to
build with room to spare. As with the default recipe, the task is tokenized at export and the bundle runs only
that task.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.export.recipe import ActionChunkRecipe, Part, ProgramCall, RunPlan
from lerobot.inference.bundle.manifest import NOISE
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import _transformers_available

from .modeling_smolvla import SmolVLAPolicy

if TYPE_CHECKING or _transformers_available:
    from transformers import DynamicCache


EMBED_OUTPUTS = ("prefix_embs", "prefix_pad_masks", "prefix_att_masks")
# What the steps read besides the noisy actions and the timestep.
CACHE = ("prefix_pad_masks", "past_keys", "past_values")
# The modules each part runs. A part's program holds only their weights.
VLM = "model.vlm_with_expert.vlm.model"
EMBED_MODULES = (
    f"{VLM}.vision_model",
    f"{VLM}.connector",
    f"{VLM}.text_model.embed_tokens",
    "model.state_proj",
)
LANGUAGE_MODEL_MODULES = (f"{VLM}.text_model.layers", f"{VLM}.text_model.norm")
STEP_MODULES = (
    "model.vlm_with_expert.lm_expert",
    "model.action_in_proj",
    "model.action_out_proj",
    "model.action_time_mlp_in",
    "model.action_time_mlp_out",
)


class _PolicyPart(nn.Module):
    """A part that runs some of the policy's own code.

    `torch.export` keeps every weight of the module it exports, used or not. So the policy is held as a plain
    reference, not a submodule, and only the modules the part runs are registered: the program holds their
    weights, not the whole policy's.
    """

    def __init__(self, policy: SmolVLAPolicy, modules: tuple[str, ...] = ()):
        super().__init__()
        object.__setattr__(self, "policy", policy)
        self.weights = nn.ModuleList(policy.get_submodule(name) for name in modules)


class SmolVLAEmbed(_PolicyPart):
    """The raw frame in; the prefix's embeddings, padding mask and attention mask out.

    It is what `predict_action_chunk` does before the language model, from the frame as the robot sends it.
    """

    def __init__(
        self,
        policy: SmolVLAPolicy,
        preprocessor: PolicyProcessorPipeline,
        frame_names: list[str],
        visual: set[str],
        task: str,
    ):
        super().__init__(policy, EMBED_MODULES)
        self.preprocessor = preprocessor
        self.frame_names = frame_names
        self.visual = visual
        self.task = task

    def forward(self, *frame: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # What prepare_observation_for_inference does to each array of the frame, as tensor operations.
        observation: dict = {
            name: ((x.float() / 255).permute(2, 0, 1).contiguous() if name in self.visual else x).unsqueeze(0)
            for name, x in zip(self.frame_names, frame, strict=True)
        }
        observation["task"] = self.task
        observation["robot_type"] = ""
        policy = self.policy
        batch = policy._prepare_batch(self.preprocessor(observation))
        images, img_masks = policy.prepare_images(batch)
        return policy.model.embed_prefix(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state=policy.prepare_state(batch),
        )


class SmolVLALanguageModel(_PolicyPart):
    """The prefix's embeddings and masks in; the keys and values of every layer out.

    The keys of all layers come out as one tensor, and the values as another: each input of a program costs time
    at every run, and the step reads them once per timestep.
    """

    def __init__(self, policy: SmolVLAPolicy):
        super().__init__(policy, LANGUAGE_MODEL_MODULES)

    def forward(
        self, prefix_embs: Tensor, prefix_pad_masks: Tensor, prefix_att_masks: Tensor
    ) -> tuple[Tensor, Tensor]:
        _, cache = self.policy.model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(prefix_pad_masks, prefix_att_masks),
            position_ids=torch.cumsum(prefix_pad_masks, dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        keys = torch.stack([layer.keys for layer in cache.layers])
        return keys, torch.stack([layer.values for layer in cache.layers])


class SmolVLAStep(_PolicyPart):
    """One Euler step of `sample_actions`: the prefix's padding mask and KV cache, the noisy actions and the
    timestep in, the next noisy actions out."""

    def __init__(self, policy: SmolVLAPolicy):
        super().__init__(policy, STEP_MODULES)
        self.dt = -1.0 / policy.config.num_steps

    def forward(
        self, prefix_pad_masks: Tensor, keys: Tensor, values: Tensor, x_t: Tensor, timestep: Tensor
    ) -> Tensor:
        past_key_values = DynamicCache(
            tuple((k, v, None) for k, v in zip(keys.unbind(0), values.unbind(0), strict=True))
        )
        velocity = self.policy.model.denoise_step(prefix_pad_masks, past_key_values, x_t[None], timestep)
        return x_t + self.dt * velocity[0]


class SmolVLAActions(_PolicyPart):
    """The denoised actions in, the actions to play out, in robot units."""

    def __init__(self, policy: SmolVLAPolicy, postprocessor: PolicyProcessorPipeline, steps: int):
        super().__init__(policy)
        self.postprocessor = postprocessor
        self.steps = steps

    def forward(self, actions: Tensor) -> Tensor:
        config = self.policy.config
        actions = actions[None, :, : config.action_feature.shape[0]]
        if config.adapt_to_pi_aloha:
            actions = self.policy._pi_aloha_encode_actions(actions.clone())
        return self.postprocessor(actions[:, : self.steps])[0].float()


class SmolVLAExportRecipe(ActionChunkRecipe):
    """SmolVLA as a chain of small programs: see this module's documentation.

    The checks and the test case are the default recipe's, made with the whole policy. Before anything is
    exported, the chain must compute exactly what the policy does, in PyTorch, for two different frames.
    """

    def parts(self) -> Iterator[Part]:
        """The embedding, the language model, the step and the actions, in the order they run."""
        policy = cast(SmolVLAPolicy, self.policy)
        embed = SmolVLAEmbed(
            policy, self.preprocessor, list(self._frame_shapes), self._visual, self.task or ""
        )
        language_model = SmolVLALanguageModel(policy)
        step = SmolVLAStep(policy)
        actions = SmolVLAActions(policy, self.postprocessor, self._steps)
        chain = (embed, language_model, step, actions)
        # The schedule of `euler_integrate`.
        self._timesteps = [1.0 + i * step.dt for i in range(policy.config.num_steps)]
        frame, noise = self._random_frame(), self._random_noise()
        for case_frame, case_noise, expected in (
            (self._case.frame, self._case.noise, self._case.actions),
            (frame, noise, self._reference(frame, noise)),
        ):
            if not np.array_equal(self._run(chain, case_frame, case_noise), expected):
                raise ValueError(
                    "The SmolVLA policy computes something else as a chain of parts than it does in "
                    "lerobot-rollout, so it cannot be exported this way."
                )

        *frame_inputs, noise_input = self._inputs(self._case.frame, self._case.noise)
        with torch.no_grad():
            prefix = embed(*frame_inputs)
            keys_values = language_model(*prefix)
        yield Part("embed", embed, tuple(frame_inputs), list(self._frame_shapes), list(EMBED_OUTPUTS))
        yield Part("language_model", language_model, prefix, list(EMBED_OUTPUTS), list(CACHE[1:]))
        timestep = torch.tensor(self._timesteps[:1], device=self._device)
        step_inputs = (prefix[1], *keys_values, noise_input, timestep)
        yield Part("step", step, step_inputs, [*CACHE, "x_t", "timestep"], ["x_next"])
        del prefix, keys_values
        yield Part("actions", actions, (noise_input,), ["x_t"], [ACTION])

    def plan(self) -> RunPlan:
        """The parts in order with the step once per timestep, and the timesteps as constants."""
        x_names = [NOISE, *(f"x_{i + 1}" for i in range(len(self._timesteps)))]
        return RunPlan(
            calls=[
                ProgramCall("embed", list(self._frame_shapes), list(EMBED_OUTPUTS)),
                ProgramCall("language_model", list(EMBED_OUTPUTS), list(CACHE[1:])),
                *(
                    ProgramCall("step", [*CACHE, x_names[i], f"timestep_{i}"], [x_names[i + 1]])
                    for i in range(len(self._timesteps))
                ),
                ProgramCall("actions", [x_names[-1]], [ACTION]),
            ],
            constants={f"timestep_{i}": np.array([t], np.float32) for i, t in enumerate(self._timesteps)},
        )

    def _run(
        self,
        chain: tuple[SmolVLAEmbed, SmolVLALanguageModel, SmolVLAStep, SmolVLAActions],
        frame: dict[str, np.ndarray],
        noise: np.ndarray | None,
    ) -> np.ndarray:
        """The chain's actions in PyTorch, for one frame and noise."""
        embed, language_model, step, actions = chain
        *frame_inputs, x_t = self._inputs(frame, noise)
        with torch.no_grad():
            prefix = embed(*frame_inputs)
            keys_values = language_model(*prefix)
            for t in self._timesteps:
                x_t = step(prefix[1], *keys_values, x_t, torch.tensor([t], device=self._device))
            return actions(x_t).cpu().numpy()
