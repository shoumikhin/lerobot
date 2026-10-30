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

"""Export LaWAM as a chain of small programs, so that it builds on a board with 8 GB of memory.

One action chunk runs as these programs, which pass their values to each other by name:

::

    embed               the frame -> the prompt's embeddings with the queries and image features, the DeepStack
                        features, the latent action model's image and, when the flow head reads it, the state
    language_model_0..  the hidden states -> the hidden states after six more Qwen3-VL layers; the last part
                        returns the final hidden states and the predicted latent action
    image_features      the latent action model's image -> its image features
    actions             those, the latent action, the hidden states, the state and the noise -> the next image
                        features the latent action predicts, then the actions in robot units

Compiled as one program, its 5 GiB of weights need more memory to build than an 8 GB board has. Here each
program holds only the weights it runs, which leaves out the parts of the latent action model that only
training runs.

LaWAM's preprocessor builds the Qwen3-VL inputs with the transformers processor, which a program cannot run.
The recipe wraps that step in [`FixedPromptStep`], with the pixel values computed by the saved processor's own
image processor as the only per-frame input. The vision tower runs on the image grid of the export's frames
([`VisionGridRecipe`]).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_file
from torch import Tensor, nn

from lerobot.bundle.manifest import NOISE, ProgramRun
from lerobot.export.recipe import (
    Part,
    RunPlan,
    VisionGridRecipe,
    load_processors,
    random_frame,
    refuse_peft_adapter,
    to_policy_input,
)
from lerobot.policies import PreTrainedPolicy
from lerobot.processor import (
    EnvTransition,
    PolicyProcessorPipeline,
    TransitionKey,
    policy_output_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import ACTION

from ..common.export_utils import PolicyPart
from ..common.fixed_prompt import FixedPromptStep
from ..common.openpi_checkpoint import load_modules, parameters_on_meta
from ..common.qwen3_vision import Qwen3VisionGrid, qwen3_text_layers, qwen3_vl_image_embeds
from .configuration_lawam import LaWAMConfig
from .modeling_lawam import LaWAMModel, LaWAMPolicy
from .processor_lawam import LaWAMQwenInputsProcessorStep
from .vlas.lawam import _cuda_autocast

if TYPE_CHECKING:
    from lerobot.export.configs import ExportConfig

BACKEND = "model.policy_backend"
QWEN = f"{BACKEND}.vlm.model"
LANGUAGE_LAYERS_PER_PART = 6
# The prompt's tensors, which `FixedPromptStep` makes the same for every frame, and the flow's inputs that are too.
PROMPT = ("attention_mask", "act_placeholder_mask", "state_mask", "action_hz", "embodiment_id")


class LaWAMExportRecipe(VisionGridRecipe):
    """[`VisionGridRecipe`] with LaWAM's Qwen step made traceable."""

    grid_type = Qwen3VisionGrid

    @classmethod
    def load(cls, cfg: ExportConfig) -> LaWAMExportRecipe:
        """The recipe, on the policy that `inference_policy` loads."""
        config = cast(LaWAMConfig, cfg.policy)
        refuse_peft_adapter(config, "LaWAM")
        config.device = cfg.backend.device
        return cls(cls.inference_policy(config), *load_processors(config), cfg)

    @staticmethod
    def inference_policy(config: LaWAMConfig) -> LaWAMPolicy:
        """The trained policy without the parts of its latent action model that only training runs.

        Prediction reads the latent action model's vision encoder and decoder; its encoder and state decoder only
        teach the policy in training. They are built without weights and dropped, so their 1.5 GiB is never read:
        an 8 GB board runs out of memory with them. The rest is built on the device, as `load_policy` builds a
        policy, and filled from the checkpoint.
        """
        from transformers.utils import cached_file

        with torch.device(config.device), parameters_on_meta():
            model = LaWAMModel(config)
        model.policy_backend.lam.encoder = None
        model.policy_backend.lam.state_decoder = None
        model_file = cached_file(
            config.pretrained_path, SAFETENSORS_SINGLE_FILE, revision=config.pretrained_revision
        )
        weights = {name.removeprefix("model."): tensor for name, tensor in load_file(model_file).items()}
        load_modules(model, weights, ["policy_backend"], config.device)
        return LaWAMPolicy(config, native_model=model).eval()

    def program_preprocessor(self) -> PolicyProcessorPipeline:
        """The saved preprocessor with its Qwen step wrapped in [`FixedPromptStep`]."""
        steps = [
            FixedPromptStep(step, _PixelValues(step))
            if isinstance(step, LaWAMQwenInputsProcessorStep)
            else step
            for step in self.preprocessor.steps
        ]
        return replace(self.preprocessor, steps=steps)

    def parts(self) -> Iterator[Part]:
        """The embedding, the language model's layers six at a time, the image features and the actions, in order."""
        policy = cast(LaWAMPolicy, self.policy)
        frame_names = list(self._features)
        preprocessor = self.program_preprocessor()
        embed = LaWAMEmbed(
            policy, preprocessor, frame_names, self._visual, self.task or "", self._vision_grid
        )
        prompt = self._prompt(preprocessor)
        depth = len(policy.model.policy_backend.vlm.model.language_model.layers)
        layers = [
            LaWAMLanguageModel(policy, range(first, min(first + LANGUAGE_LAYERS_PER_PART, depth)), prompt)
            for first in range(0, depth, LANGUAGE_LAYERS_PER_PART)
        ]
        image_features = LaWAMImageFeatures(policy)
        actions = LaWAMActions(policy, self.postprocessor, self._steps, prompt)
        chain = (embed, *layers, image_features, actions)
        frame, noise = random_frame(self._features, self._generator), self._random_noise()
        for case_frame, case_noise, expected in (
            (self._case.frame, self._case.noise, self._case.actions),
            (frame, noise, self._reference(frame, noise)),
        ):
            if not np.array_equal(self._run(chain, case_frame, case_noise)[-1], expected):
                raise ValueError(
                    "The LaWAM policy computes something else as a chain of parts than it does in "
                    "lerobot-rollout, so it cannot be exported this way."
                )

        values = self._run(chain, self._case.frame, self._case.noise)
        deepstack_names = [f"deepstack_{i}" for i in range(embed.num_deepstack)]
        state_names = ["state"] if embed.reads_state else []
        hidden_names = [f"language_hidden_{i}" for i in range(len(layers) - 1)]
        self._runs = []
        yield self._part(
            "embed",
            embed,
            values[0],
            frame_names,
            ["inputs_embeds", *deepstack_names, "lam_image", *state_names],
        )
        for group, (part, inputs, input_name, output_names) in enumerate(
            zip(
                layers,
                values[1 : len(layers) + 1],
                ["inputs_embeds", *hidden_names],
                [*([name] for name in hidden_names), ["vlm_hidden", "latent_action"]],
                strict=True,
            )
        ):
            names = [input_name, *deepstack_names[: len(inputs) - 1]]
            yield self._part(f"language_model_{group}", part, inputs, names, output_names)
        yield self._part("image_features", image_features, values[-3], ["lam_image"], ["image_features"])
        yield self._part(
            "actions",
            actions,
            values[-2],
            ["image_features", "latent_action", "vlm_hidden", *state_names, NOISE],
            [ACTION],
        )

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
        """The tensors of the batch that are the same for every frame (`PROMPT`), and the prompt's image mask."""
        self._reset()
        with torch.no_grad():
            batch = preprocessor(self._observation(self._case.frame, self.task))
        image_token_id = cast(LaWAMPolicy, self.policy).model.policy_backend.vlm.config.image_token_id
        return {name: batch[name] for name in PROMPT} | {"image_mask": batch["input_ids"] == image_token_id}

    def _run(
        self, chain: tuple[nn.Module, ...], frame: dict[str, np.ndarray], noise: np.ndarray | None
    ) -> list[Any]:
        """Each part's inputs, in order, then the chain's actions in PyTorch, for one frame and noise."""
        embed, *layers, image_features, actions = chain
        *frame_inputs, noise_input = self._inputs(frame, noise)
        values: list[Any] = [tuple(frame_inputs)]
        with torch.no_grad():
            hidden, *rest = embed(*frame_inputs)
            deepstack, (lam_image, *state) = rest[: embed.num_deepstack], rest[embed.num_deepstack :]
            for part in layers:
                inputs = (hidden, *(deepstack if part.layers.start < len(deepstack) else ()))
                values.append(inputs)
                outputs = part(*inputs)
                hidden = outputs[0] if part.last else outputs
            vlm_hidden, latent_action = outputs
            values.append((lam_image,))
            features = image_features(lam_image)
            values.append((features, latent_action, vlm_hidden, *state, noise_input))
            values.append(actions(*values[-1]).cpu().numpy())
        return values

    def vision_tower(self, policy: PreTrainedPolicy) -> nn.Module:
        """The policy's Qwen vision tower."""
        return policy.model.policy_backend.vlm.model.visual


class LaWAMEmbed(PolicyPart):
    """The raw frame in; the prompt's embeddings with the queries and the image features, the DeepStack features,
    the latent action model's image and the state out, as LaWAM computes them before Qwen3-VL's text model."""

    def __init__(
        self,
        policy: LaWAMPolicy,
        preprocessor: PolicyProcessorPipeline,
        frame_names: list[str],
        visual: set[str],
        task: str,
        vision_grid: Qwen3VisionGrid,
    ):
        super().__init__(policy, (f"{QWEN}.visual", f"{QWEN}.language_model.embed_tokens"))
        backend = policy.model.policy_backend
        # Not in a module of their own.
        self.queries = nn.ParameterList([backend.act_query, backend.flow_action_query])
        self.preprocessor = preprocessor
        self.frame_names = frame_names
        self.visual = visual
        self.task = task
        self.vision_grid = vision_grid
        self.num_deepstack = len(backend.vlm.model.visual.deepstack_visual_indexes)
        self.reads_state = backend.flow.config.use_state

    def forward(self, *frame: Tensor) -> tuple[Tensor, ...]:
        observation: dict[str, Any] = {
            name: to_policy_input(x, name in self.visual)
            for name, x in zip(self.frame_names, frame, strict=True)
        }
        observation["task"] = self.task
        observation["robot_type"] = ""
        batch = self.preprocessor(observation)
        backend = self.policy.model.policy_backend
        input_ids = batch["input_ids"]
        vlm_dtype = backend.model_cfg.vlm_dtype
        act_query, flow_query = backend._prepare_queries(device=input_ids.device, vlm_stage_dtype=vlm_dtype)
        with _cuda_autocast(vlm_dtype):
            embeds = backend._query_embeds(
                input_ids,
                batch["act_placeholder_mask"],
                batch["flow_placeholder_mask"],
                act_query,
                flow_query,
            )
            hidden, deepstack = qwen3_vl_image_embeds(
                backend.vlm.model,
                embeds,
                batch["pixel_values"],
                batch["image_grid_thw"],
                self.vision_grid,
                input_ids == backend.vlm.config.image_token_id,
            )
        return hidden, *deepstack, batch["primary_image"], *((batch["state"],) if self.reads_state else ())


class LaWAMLanguageModel(PolicyPart):
    """The hidden states in, and the DeepStack features its layers add; the hidden states after `layers` out.

    The last part returns the text model's final hidden states and the latent action predicted from the action
    queries' states. `prompt` is the batch's tensors that are the same for every frame.
    """

    def __init__(self, policy: LaWAMPolicy, layers: range, prompt: dict[str, Tensor]):
        last = layers.stop == len(policy.model.policy_backend.vlm.model.language_model.layers)
        last_modules = (f"{QWEN}.language_model.norm", f"{BACKEND}.vlm_to_lam") if last else ()
        super().__init__(policy, (*(f"{QWEN}.language_model.layers.{i}" for i in layers), *last_modules))
        self.layers = layers
        self.last = last
        for name in ("attention_mask", "image_mask", "act_placeholder_mask"):
            self.register_buffer(name, prompt[name])

    def forward(self, hidden: Tensor, *deepstack: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        backend = self.policy.model.policy_backend
        text = backend.vlm.model.language_model
        with _cuda_autocast(backend.model_cfg.vlm_dtype):
            hidden = qwen3_text_layers(
                text, hidden, self.attention_mask, self.image_mask, list(deepstack), self.layers
            )
            if not self.last:
                return hidden
            hidden = text.norm(hidden)
            return hidden, backend.vlm_to_lam(backend._action_query_states(hidden, self.act_placeholder_mask))


class LaWAMImageFeatures(PolicyPart):
    """The latent action model's image in; its image features out."""

    def __init__(self, policy: LaWAMPolicy):
        super().__init__(policy, (f"{BACKEND}.lam.vision_encoder",))

    def forward(self, image: Tensor) -> Tensor:
        with _cuda_autocast(torch.bfloat16):
            return self.policy.model.policy_backend.lam.extract_vision_features(image)[:, 0, :, :]


class LaWAMActions(PolicyPart):
    """The latent action model's image features, the latent action, the text model's hidden states, the state when
    the flow head reads it, and the noise in; the actions to play out, in robot units."""

    def __init__(
        self,
        policy: LaWAMPolicy,
        postprocessor: PolicyProcessorPipeline,
        execution_steps: int,
        prompt: dict[str, Tensor],
    ):
        backend = policy.model.policy_backend
        decoder = (f"{BACKEND}.lam.decoder",) if backend.model_cfg.future_prediction else ()
        super().__init__(policy, (*decoder, f"{BACKEND}.flow"))
        self.postprocessor = postprocessor
        self.execution_steps = execution_steps
        for name in ("attention_mask", "state_mask", "action_hz", "embodiment_id"):
            self.register_buffer(name, prompt[name])

    def forward(self, now: Tensor, latent_action: Tensor, hidden: Tensor, *inputs: Tensor) -> Tensor:
        *state, noise = inputs
        policy = self.policy
        backend = policy.model.policy_backend
        next_ = now
        if backend.model_cfg.future_prediction:
            with _cuda_autocast(torch.bfloat16):
                next_ = backend._decode_future_tokens_strict_single_query(
                    h_t=now, pred_action_emb=latent_action, source="LaWAMActions"
                )
        with _cuda_autocast(torch.float32):
            actions = backend.flow.sample_actions_cfg(
                h_t=now,
                h_t1_star=next_,
                h_vlm=hidden,
                # A flow head without a state token never reads it.
                state=state[0] if state else None,
                state_mask=self.state_mask,
                action_hz=self.action_hz,
                embodiment_id=self.embodiment_id,
                cfg_scale=float(backend.flow.config.cfg_guidance_scale),
                num_inference_steps=int(backend.flow.config.num_inference_steps),
                attention_mask=self.attention_mask == 1,
                noise=noise[None],
            )
        actions = transition_to_policy_action(policy_output_to_transition(policy._action_chunk(actions)))
        return self.postprocessor(actions[:, : self.execution_steps])[0].float()


class _PixelValues:
    """The pixel values LaWAM's Qwen step computes, from its processor's image processor alone."""

    def __init__(self, step: LaWAMQwenInputsProcessorStep) -> None:
        self.step = step

    def __call__(self, transition: EnvTransition) -> dict[str, torch.Tensor]:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        image_views, wrist_image_views = self.step.camera_views(observation)
        images = [
            image
            for primary, wrist in zip(image_views, wrist_image_views, strict=True)
            for image in (*primary, *wrist)
        ]
        processor, _ = self.step._ensure_processor()
        processed = processor.image_processor(
            images=images, do_rescale=False, device=images[0].device, return_tensors="pt"
        )
        return {"pixel_values": processed["pixel_values"]}
