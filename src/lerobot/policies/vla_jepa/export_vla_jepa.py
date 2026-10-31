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
preprocessor makes those inputs in a last step, with the model's own `qwen_inputs`, run in [`FixedPromptStep`],
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

from lerobot.bundle.manifest import NOISE
from lerobot.export.recipe import Part, PostprocessedActions, VisionGridRecipe, to_policy_input
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
        """The policy's chunk for a preprocessed batch, its vision tower on the grid of the first batch.

        As `predict_action_chunk` computes it, which takes no grid: the model's own `predict_action` does.
        """
        policy = cast(VLAJEPAPolicy, policy)
        inputs = policy._prepare_model_inputs(batch, training=False)
        qwen_inputs = policy.model.qwen_inputs(inputs["images"], inputs["instructions"])
        if self._vision_grid is None:
            self._vision_grid = self.grid_type.build(self.vision_tower(policy), qwen_inputs["image_grid_thw"])
        actions = policy.model.predict_action(
            qwen_inputs, inputs.get("state"), noise=noise, vision_grid=self._vision_grid
        )
        return transition_to_policy_action(policy_output_to_transition(actions.to(torch.float32)))

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
        self._check_chain(lambda *inputs: self._run(chain, *inputs)[-1])
        values = self._run(chain, *self._inputs(self._case.frame, self._case.noise))
        deepstack_names = [f"deepstack_{i}" for i in range(embed.num_deepstack)]
        state_names = ["state"] if embed.reads_state else []
        hidden_names = [f"language_hidden_{i}" for i in range(len(layers) - 1)]
        yield Part("embed", embed, values[0], frame_names, ["inputs_embeds", *deepstack_names, *state_names])
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
            yield Part(f"language_model_{group}", part, inputs, names, [output_name])
        yield Part(
            "actions",
            actions,
            values[-2],
            ["embodied_tokens", *state_names, NOISE],
            actions.postprocess.outputs,
        )

    def _prompt(self, preprocessor: PolicyProcessorPipeline) -> dict[str, Tensor]:
        """The prompt's tensors, which the preprocessor's `FixedPromptStep` makes the same for every frame."""
        self._reset()
        with torch.no_grad():
            batch = preprocessor(self._observation(self._case.frame, self.task))
        qwen = cast(VLAJEPAPolicy, self.policy).model.qwen.model.model
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        position_ids = qwen_image_rope_index(
            input_ids, attention_mask, self._vision_grid.grid_thw, qwen.config
        )
        image_mask = input_ids == qwen.config.image_token_id
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_mask": image_mask,
            "position_ids": position_ids,
        }

    def _run(self, chain: tuple[nn.Module, ...], *inputs: Tensor) -> list[Any]:
        """Each part's inputs, in order, then the chain's actions in PyTorch, for the program's inputs."""
        embed, *layers, actions = chain
        *frame_inputs, noise_input = inputs
        values: list[Any] = [tuple(frame_inputs)]
        with torch.no_grad():
            hidden, *rest = embed(*frame_inputs)
            deepstack, state = rest[: embed.num_deepstack], tuple(rest[embed.num_deepstack :])
            for part in layers:
                inputs = (hidden, *(deepstack if part.layers.start < len(deepstack) else ()))
                values.append(inputs)
                hidden = part(*inputs)
            values.append((hidden, *state, noise_input))
            values.append(actions(*values[-1]))
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
        self.postprocess = PostprocessedActions(postprocessor)
        self.execution_steps = execution_steps

    def forward(self, tokens: Tensor, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        *state, noise = inputs
        actions = self.policy.model.action_model.predict_action(
            tokens.float(), state[0].float() if state else None, noise=noise[None]
        )
        actions = transition_to_policy_action(policy_output_to_transition(actions.to(torch.float32)))
        return self.postprocess(actions[:, : self.execution_steps])


def _qwen_inputs(policy: VLAJEPAPolicy, batch: dict[str, Any]) -> dict[str, Tensor]:
    """The Qwen3-VL processor tensors `predict_action_chunk` makes for a preprocessed batch."""
    inputs = policy._prepare_model_inputs(batch, training=False)
    return policy.model.qwen_inputs(inputs["images"], inputs["instructions"])


class _QwenInputsStep(ProcessorStep):
    """The Qwen3-VL processor tensors that `predict_action_chunk` would make, added to the complementary data."""

    def __init__(self, policy: VLAJEPAPolicy) -> None:
        self.policy = policy

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        comp = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        comp.update(_qwen_inputs(self.policy, transition_to_batch(transition)))
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
