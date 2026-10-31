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

"""Export VLA-JEPA as a chain of small programs, so that it builds on a board with 8 GB of memory.

One action chunk runs as these programs, which pass their values to each other by name:

::

    embed               the frame -> the prompt's embeddings with the image features, the DeepStack features
                        and the state
    language_model_0..  the hidden states -> the hidden states after six more Qwen3-VL layers; the last part
                        gathers the embodied-action tokens' states
    actions             those states, the state and the noise -> the actions in robot units

Compiled as one program, its 5 GiB of weights need more memory to build than an 8 GB board has. Here each
program holds only the weights it runs.

VLA-JEPA makes its Qwen3-VL inputs inside `predict_action_chunk`, with the transformers processor, which
renders a chat template and tokenizes it in Python, so a program cannot run it. The `embed` program's
preprocessor makes those inputs in a last step, with the policy's own `qwen_inputs`, run in [`FixedPromptStep`],
with the pixel values, computed by the processor's own image processor, as the only per-frame input. The checks
compare the chain with the saved preprocessor and the policy making its own inputs, as rollout runs them. The
vision tower runs on the image grid of the export's frames ([`VisionGridRecipe`]).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.bundle.manifest import NOISE, ProgramRun
from lerobot.export.recipe import Part, RunPlan, VisionGridRecipe, random_frame, to_policy_input
from lerobot.policies import PreTrainedPolicy
from lerobot.processor import (
    EnvTransition,
    PolicyProcessorPipeline,
    ProcessorStep,
    TransitionKey,
    policy_output_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import get_autocast_context

from ..common.export_utils import PolicyPart
from ..common.fixed_prompt import FixedPromptStep
from ..common.qwen3_vision import Qwen3VisionGrid, qwen3_text_layers, qwen3_vl_image_embeds
from ..common.qwen_vision import check_visible_images, qwen_image_rope_index
from .modeling_vla_jepa import VLAJEPAPolicy

QWEN = "model.qwen.model.model"
LANGUAGE_LAYERS_PER_PART = 6


class VLAJEPAExportRecipe(VisionGridRecipe):
    """[`VisionGridRecipe`] with VLA-JEPA's Qwen inputs made in the preprocessor, the vision tower on a fixed grid."""

    grid_type = Qwen3VisionGrid

    def program_preprocessor(self) -> PolicyProcessorPipeline:
        """The saved preprocessor with VLA-JEPA's Qwen inputs made in a last step, in [`FixedPromptStep`]."""
        step = _QwenInputsStep(cast(VLAJEPAPolicy, self.policy))
        steps = [*self.preprocessor.steps, FixedPromptStep(step, step.pixel_values)]
        return replace(self.preprocessor, steps=steps)

    def predict_chunk(self, policy: PreTrainedPolicy, batch: dict[str, Any], noise: Tensor | None) -> Tensor:
        """The default recipe's chunk; the grid comes from the Qwen inputs the policy makes for the batch."""
        if self._vision_grid is None and "image_grid_thw" not in batch:
            grid_thw = cast(VLAJEPAPolicy, policy).qwen_inputs(batch)["image_grid_thw"]
            self._vision_grid = self.grid_type.build(self.vision_tower(policy), grid_thw)
        return super().predict_chunk(policy, batch, noise)

    def parts(self) -> Iterator[Part]:
        """The embedding, the language model's layers six at a time, and the actions, in the order they run."""
        policy = cast(VLAJEPAPolicy, self.policy)
        frame_names = list(self._features)
        preprocessor = self.program_preprocessor()
        embed = VLAJEPAEmbed(
            policy, preprocessor, frame_names, self._visual, self.task or "", self._vision_grid
        )
        prompt = self._prompt(preprocessor)
        depth = len(policy.model.qwen.model.model.language_model.layers)
        layers = [
            VLAJEPALanguageModel(policy, range(first, min(first + LANGUAGE_LAYERS_PER_PART, depth)), prompt)
            for first in range(0, depth, LANGUAGE_LAYERS_PER_PART)
        ]
        actions = VLAJEPAActions(policy, self.postprocessor, self._steps)
        chain = (embed, *layers, actions)
        frame, noise = random_frame(self._features, self._generator), self._random_noise()
        for case_frame, case_noise, expected in (
            (self._case.frame, self._case.noise, self._case.actions),
            (frame, noise, self._reference(frame, noise)),
        ):
            if not np.array_equal(self._run(chain, case_frame, case_noise)[-1], expected):
                raise ValueError(
                    "The VLA-JEPA policy computes something else as a chain of parts than it does in "
                    "lerobot-rollout, so it cannot be exported this way."
                )

        values = self._run(chain, self._case.frame, self._case.noise)
        deepstack_names = [f"deepstack_{i}" for i in range(embed.num_deepstack)]
        state_names = ["state"] if embed.reads_state else []
        hidden_names = [f"language_hidden_{i}" for i in range(len(layers) - 1)]
        self._runs = []
        yield self._part(
            "embed", embed, values[0], frame_names, ["inputs_embeds", *deepstack_names, *state_names]
        )
        for group, (part, inputs, input_name, output_name) in enumerate(
            zip(
                layers,
                values[1:-2],
                ["inputs_embeds", *hidden_names],
                [*hidden_names, "embodied_tokens"],
                strict=True,
            )
        ):
            names = [input_name, *deepstack_names[: len(inputs) - 1]]
            yield self._part(f"language_model_{group}", part, inputs, names, [output_name])
        yield self._part("actions", actions, values[-2], ["embodied_tokens", *state_names, NOISE], [ACTION])

    def plan(self) -> RunPlan:
        """The parts in order, each once."""
        return RunPlan(runs=self._runs)

    def _part(
        self,
        name: str,
        module: nn.Module,
        inputs: tuple[Tensor, ...],
        input_names: list[str],
        outputs: list[str],
    ) -> Part:
        self._runs.append(ProgramRun(name, name, input_names, outputs))
        return Part(name, module, inputs, input_names, outputs)

    def _prompt(self, preprocessor: PolicyProcessorPipeline) -> dict[str, Tensor]:
        """The prompt's tensors, which the preprocessor's `FixedPromptStep` makes the same for every frame."""
        self._reset()
        with torch.no_grad():
            batch = preprocessor(self._observation(self._case.frame, self.task))
        qwen = cast(VLAJEPAPolicy, self.policy).model.qwen.model.model
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        position_ids = qwen_image_rope_index(
            input_ids,
            attention_mask,
            self._vision_grid.grid_thw,
            qwen.visual.spatial_merge_size,
            qwen.config.image_token_id,
        )
        image_mask = input_ids == qwen.config.image_token_id
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_mask": image_mask,
            "position_ids": position_ids,
        }

    def _run(
        self, chain: tuple[nn.Module, ...], frame: dict[str, np.ndarray], noise: np.ndarray | None
    ) -> list[Any]:
        """Each part's inputs, in order, then the chain's actions in PyTorch, for one frame and noise."""
        embed, *layers, actions = chain
        *frame_inputs, noise_input = self._inputs(frame, noise)
        values: list[Any] = [tuple(frame_inputs)]
        with torch.no_grad():
            hidden, *rest = embed(*frame_inputs)
            deepstack, state = rest[: embed.num_deepstack], tuple(rest[embed.num_deepstack :])
            for part in layers:
                inputs = (hidden, *(deepstack if part.layers.start < len(deepstack) else ()))
                values.append(inputs)
                hidden = part(*inputs)
            values.append((hidden, *state, noise_input))
            values.append(actions(*values[-1]).cpu().numpy())
        return values

    def _reads_task(self, frame: dict[str, np.ndarray]) -> bool:
        # VLA-JEPA writes the task into its Qwen prompt itself, which the saved preprocessor cannot show.
        return True

    def vision_tower(self, policy: PreTrainedPolicy) -> nn.Module:
        """The policy's Qwen vision tower."""
        return policy.model.qwen.model.model.visual


class VLAJEPAEmbed(PolicyPart):
    """The raw frame in; the prompt's embeddings with the image features, the DeepStack features and the state
    out, as `predict_action_chunk` computes them before Qwen3-VL's text model."""

    def __init__(
        self,
        policy: VLAJEPAPolicy,
        preprocessor: PolicyProcessorPipeline,
        frame_names: list[str],
        visual: set[str],
        task: str,
        vision_grid: Qwen3VisionGrid,
    ):
        super().__init__(policy, (f"{QWEN}.visual", f"{QWEN}.language_model.embed_tokens"))
        self.preprocessor = preprocessor
        self.frame_names = frame_names
        self.visual = visual
        self.task = task
        self.vision_grid = vision_grid
        self.num_deepstack = len(policy.model.qwen.model.model.visual.deepstack_visual_indexes)
        self.reads_state = policy.config.robot_state_feature is not None

    def forward(self, *frame: Tensor) -> tuple[Tensor, ...]:
        observation: dict[str, Any] = {
            name: to_policy_input(x, name in self.visual)
            for name, x in zip(self.frame_names, frame, strict=True)
        }
        observation["task"] = self.task
        observation["robot_type"] = ""
        batch = self.preprocessor(observation)
        policy = self.policy
        state = policy._prepare_model_inputs(batch, training=False).get("state")
        qwen = policy.model.qwen.model.model
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        with get_autocast_context(input_ids.device.type, torch.bfloat16):
            check_visible_images(input_ids, attention_mask, qwen.config)
            hidden, deepstack = qwen3_vl_image_embeds(
                qwen,
                qwen.get_input_embeddings()(input_ids),
                batch["pixel_values"],
                batch["image_grid_thw"],
                self.vision_grid,
                input_ids == qwen.config.image_token_id,
            )
        return hidden, *deepstack, *((state,) if self.reads_state else ())


class VLAJEPALanguageModel(PolicyPart):
    """The hidden states in, and the DeepStack features its layers add; the hidden states after `layers` out.

    The last part returns the embodied-action tokens' states, read before the text model's final norm, as
    VLA-JEPA reads them. `prompt` is the prompt's tensors, the same for every frame.
    """

    def __init__(self, policy: VLAJEPAPolicy, layers: range, prompt: dict[str, Tensor]):
        text = policy.model.qwen.model.model.language_model
        super().__init__(policy, tuple(f"{QWEN}.language_model.layers.{i}" for i in layers))
        self.layers = layers
        self.last = layers.stop == len(text.layers)
        for name, value in prompt.items():
            self.register_buffer(name, value)

    def forward(self, hidden: Tensor, *deepstack: Tensor) -> Tensor:
        policy = self.policy
        text = policy.model.qwen.model.model.language_model
        with get_autocast_context(hidden.device.type, torch.bfloat16):
            hidden = qwen3_text_layers(
                text,
                hidden,
                self.attention_mask,
                self.image_mask,
                list(deepstack),
                self.layers,
                self.position_ids,
            )
            if self.last:
                return policy.model._embodied_token_states(hidden, self.input_ids)
        return hidden


class VLAJEPAActions(PolicyPart):
    """The embodied-action tokens' states, the state and the noise in; the actions to play out, in robot units."""

    def __init__(self, policy: VLAJEPAPolicy, postprocessor: PolicyProcessorPipeline, execution_steps: int):
        super().__init__(policy, ("model.action_model",))
        self.postprocessor = postprocessor
        self.execution_steps = execution_steps

    def forward(self, tokens: Tensor, *inputs: Tensor) -> Tensor:
        *state, noise = inputs
        actions = self.policy.model.action_model.predict_action(
            tokens.float(), state[0].float() if state else None, noise=noise[None]
        )
        actions = transition_to_policy_action(policy_output_to_transition(actions.to(torch.float32)))
        return self.postprocessor(actions[:, : self.execution_steps])[0].float()


class _QwenInputsStep(ProcessorStep):
    """The Qwen3-VL processor tensors that `predict_action_chunk` would make, added to the complementary data."""

    def __init__(self, policy: VLAJEPAPolicy) -> None:
        self.policy = policy

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        comp = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        comp.update(self.policy.qwen_inputs(transition_to_batch(transition)))
        return {**transition, TransitionKey.COMPLEMENTARY_DATA: comp}

    def pixel_values(self, transition: EnvTransition) -> dict[str, torch.Tensor]:
        """The pixel values of the step's tensors, from the processor's image processor alone."""
        inputs = self.policy._prepare_model_inputs(transition_to_batch(transition), training=False)
        images = [image for views in inputs["images"] for image in views]
        qwen = self.policy.model.qwen
        processed = qwen.processor.image_processor(
            images=images, do_rescale=False, device=qwen.model.device, return_tensors="pt"
        )
        return {"pixel_values": processed["pixel_values"]}

    def transform_features(self, features):
        return features
