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

"""Export pi0 as a chain of small programs, so that it builds and runs on a board with 8 GB of memory.

As one program, pi0 needs all its weights in memory at once, and a compiler builds its ten denoising steps as one
graph that reads the action expert's weights ten times. One action chunk runs as these programs instead, which pass
their values to each other by name, as pi0.5's do (see `export_pi05`):

::

    embed                the frame -> the prefix's embeddings and padding mask, and the padded state
    language_model_0..k  3 layers of the language model each -> the next hidden states and their KV cache
    step x10             the KV cache, the state, the noisy actions and a timestep -> the next noisy actions
    actions              the denoised actions -> the actions in robot units

Each part is built with only its own weights in memory. pi0's prompt is the task alone, so the `embed` program holds
the embedding rows of the exported task's tokens, made by the policy's own processor, and the bundle runs that task,
as the default recipe's does.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from copy import copy
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.bundle.manifest import NOISE, ProgramRun
from lerobot.configs import FeatureType
from lerobot.export.recipe import (
    BFLOAT16_STEPS_TOLERANCE,
    FLOAT32_TOLERANCE,
    ExportRecipe,
    Part,
    RunPlan,
    TestCase,
    dataset_metadata,
    export_task,
    load_processors,
    measure_action_scale,
    preprocessor_renames,
    random_frame,
    refuse_peft_adapter,
    robot_features,
    split_absolute_step,
    to_policy_input,
    whole_policy_fits,
)
from lerobot.policies import prepare_observation_for_inference
from lerobot.policies.common.meta_load import load_checkpoint_modules
from lerobot.processor import PolicyProcessorPipeline, TokenizerProcessorStep
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import _transformers_available

from ..pi05.export_pi05 import (
    LANGUAGE_LAYERS_PER_PART,
    LookedUpEmbedding,
    PI05Actions,
    PI05LanguageModelLayers,
    hold_linear_weights_in_bfloat16,
)
from . import modeling_pi0
from .modeling_pi0 import PI0Policy, PI0Pytorch
from .processor_pi0 import Pi0NewLineProcessor

if TYPE_CHECKING or _transformers_available:
    from transformers import DynamicCache

if TYPE_CHECKING:
    from lerobot.configs import PolicyFeature
    from lerobot.export.configs import ExportConfig

    from .configuration_pi0 import PI0Config

logger = logging.getLogger(__name__)

LANGUAGE_MODEL = "model.paligemma_with_expert.paligemma.model.language_model"
EMBED_TOKENS = f"{LANGUAGE_MODEL}.embed_tokens"
# The vision encoder, its projector and the token embeddings; the language model's lm_head is never needed.
EMBED_MODULES = (
    "model.paligemma_with_expert.paligemma.model.vision_tower",
    "model.paligemma_with_expert.paligemma.model.multi_modal_projector",
    EMBED_TOKENS,
)
# The action expert without its lm_head, and the layers around it that `denoise_step` runs.
STEP_MODULES = (
    "model.paligemma_with_expert.gemma_expert.model",
    "model.state_proj",
    "model.action_in_proj",
    "model.action_out_proj",
    "model.action_time_mlp_in",
    "model.action_time_mlp_out",
)
PADDED_STATE = "padded_state"


class PI0Embed(nn.Module):
    """The raw frame in; the prefix's embeddings and padding mask, and the state padded as the policy reads it, out.

    The task's tokens are the rows `task_rows`, which the language model's embedding scales as it scales the rows
    of the token ids (see `LookedUpEmbedding`). Only the tokens the policy attends to are kept: padding takes no
    part in attention and shifts no position. The policy fills a camera the frame does not hold with an image of -1
    that it leaves out of attention, and puts it after the others, so the program leaves it out too.
    """

    def __init__(
        self,
        policy: PI0Policy,
        preprocessor: PolicyProcessorPipeline,
        task_rows: Tensor,
        frame_names: list[str],
        visual: set[str],
    ):
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.register_buffer("task_rows", task_rows)
        self.register_buffer(
            "task_mask", torch.ones(task_rows.shape[:2], dtype=torch.bool, device=task_rows.device)
        )
        self.frame_names = frame_names
        self.visual = visual

    def forward(self, *frame: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch = self.preprocessor(
            {
                name: to_policy_input(x, name in self.visual)
                for name, x in zip(self.frame_names, frame, strict=True)
            }
        )
        # The policy's own image resize and padding, which it exposes under no public name.
        images, img_masks = self.policy._preprocess_images(batch)
        cameras = len(self.visual)
        prefix_embs, prefix_pad_masks, _ = self.policy.model.embed_prefix(
            images[:cameras], img_masks[:cameras], self.task_rows, self.task_mask
        )
        return prefix_embs, prefix_pad_masks, self.policy.prepare_state(batch)


class PI0Step(nn.Module):
    """One step of `denoise_step` with its Euler update: the prefix's mask and KV cache, the padded state, the noisy
    actions and the timestep in, the next noisy actions out.

    It runs the policy's own `denoise_step`, with the weights in the dtypes the checkpoint gives them.
    """

    def __init__(self, model: PI0Pytorch, dt: float):
        super().__init__()
        self.model = model
        self.dt = dt

    def forward(self, prefix_pad_masks: Tensor, state: Tensor, *inputs: Tensor) -> Tensor:
        *cache, x_t, timestep = inputs
        past_key_values = DynamicCache(
            tuple((keys, values, None) for keys, values in zip(cache[::2], cache[1::2], strict=True))
        )
        velocity = self.model.denoise_step(state, prefix_pad_masks, past_key_values, x_t[None], timestep)
        return x_t + self.dt * velocity[0]


class PI0ExportRecipe(ExportRecipe):
    """pi0 as a chain of small programs: see this module's documentation.

    It builds the policy on the meta device and loads each part's weights when the part is made, so the trace
    holds one part's weights at a time. The test case's actions are the parts run one after another in PyTorch.
    When the whole policy fits in memory, they are also compared with the policy's own `predict_action_chunk`.

    Args:
        cfg (`ExportConfig`):
            The `lerobot-export` configuration.
    """

    # pi0 runs ten bfloat16 denoising steps.
    bfloat16_tolerance = BFLOAT16_STEPS_TOLERANCE

    @classmethod
    def load(cls, cfg: ExportConfig) -> PI0ExportRecipe:
        """The recipe, with no weight loaded yet."""
        return cls(cfg)

    def __init__(self, cfg: ExportConfig) -> None:  # noqa: D107
        from transformers.utils import cached_file

        policy_config = cast("PI0Config", cfg.policy)
        refuse_peft_adapter(policy_config, "pi0")
        self.cfg = cfg
        self._device = torch.device(cfg.backend.device)
        policy_config.device = cfg.backend.device
        # How the policy runs, not how it trained, decides what compiles it.
        policy_config.compile_model = False
        self._policy_config = policy_config
        self._model_file = Path(
            cached_file(
                policy_config.pretrained_path, "model.safetensors", revision=policy_config.pretrained_revision
            )
        )
        # The policy without weights checks that it can run one chunk from the current frame.
        self._steps = self._load().chunk_inference_spec().execution_steps
        self.preprocessor, self.postprocessor = load_processors(policy_config)
        self.tolerance = (
            cfg.tolerance
            if cfg.tolerance is not None
            else self.bfloat16_tolerance
            if policy_config.dtype == torch.bfloat16
            else FLOAT32_TOLERANCE
        )
        self._generator = torch.Generator().manual_seed(cfg.seed)
        self._metadata = dataset_metadata(cfg)
        self._features = robot_features(policy_config, self.preprocessor, self._metadata)
        self._visual = {name for name, f in self._features.items() if f.type is FeatureType.VISUAL}
        self.config = copy(policy_config)
        self.config.input_features = self._features
        renames = preprocessor_renames(self.preprocessor)
        self._state_name = next(name for name in self._features if renames.get(name, name) == OBS_STATE)
        self.task = export_task(cfg, self._metadata, "pi0")
        self._frame = random_frame(self._features, self._generator)
        self._noise = torch.randn(
            policy_config.chunk_size, policy_config.max_action_dim, generator=self._generator
        )
        self._dt = -1.0 / policy_config.num_inference_steps
        # The schedule of `euler_integrate`.
        self._timesteps = [1.0 + step * self._dt for step in range(policy_config.num_inference_steps)]
        self._task_ids = self._task_token_ids()
        self._runs: list[ProgramRun] = []
        self._actions: np.ndarray | None = None

    def parts(self) -> Iterator[Part]:
        """The chain's parts, each run in PyTorch to make the next part's inputs, one in memory at a time."""
        config = self._policy_config
        frame_names = list(self._features)
        frame = tuple(torch.from_numpy(x).to(self._device) for x in self._frame.values())

        policy = self._load(*EMBED_MODULES)
        if config.dtype == torch.bfloat16 and self._device.type == "cuda":
            # The policy runs its float32 vision tower under bfloat16 autocast on a GPU (`_vision_autocast`).
            vlm = policy.model.paligemma_with_expert.paligemma.model
            hold_linear_weights_in_bfloat16(vlm.vision_tower)
            hold_linear_weights_in_bfloat16(vlm.multi_modal_projector)
        language_model = policy.model.paligemma_with_expert.paligemma.model.language_model
        embedding = language_model.embed_tokens
        task_rows = embedding.weight.detach()[self._task_ids][None]
        # The program holds the task's rows, so it drops the table.
        language_model.embed_tokens = LookedUpEmbedding(embedding)
        del embedding
        embed = PI0Embed(policy, self._tensor_preprocessor(), task_rows, frame_names, self._visual)
        with torch.no_grad():
            hidden, prefix_pad_masks, state = embed(*frame)
        yield self._part(
            "embed", embed, frame, frame_names, ["prefix_embs", "prefix_pad_masks", PADDED_STATE]
        )
        del policy, language_model, embed, task_rows

        hidden_name = "prefix_embs"
        cache: dict[str, Tensor] = {}
        depth = modeling_pi0.get_gemma_config(config.paligemma_variant).depth
        for group, first in enumerate(range(0, depth, LANGUAGE_LAYERS_PER_PART)):
            layers = range(first, min(first + LANGUAGE_LAYERS_PER_PART, depth))
            module = PI05LanguageModelLayers(
                self._load(*(f"{LANGUAGE_MODEL}.layers.{i}" for i in layers)).model, layers
            )
            inputs: tuple[Tensor, ...] = (hidden, prefix_pad_masks)
            with torch.no_grad():
                outputs = module(*inputs)
            cache_names = [f"past_{kind}_{i}" for i in layers for kind in ("key", "value")]
            output_names = cache_names if module.last else [f"language_hidden_{group}", *cache_names]
            if not module.last:
                hidden, *outputs = outputs
            cache.update(zip(cache_names, outputs, strict=True))
            yield self._part(
                f"language_model_{group}", module, inputs, [hidden_name, "prefix_pad_masks"], output_names
            )
            del module, inputs, outputs
            hidden_name = output_names[0]

        step = PI0Step(self._load(*STEP_MODULES).model, self._dt)
        cache_inputs = (prefix_pad_masks, state, *cache.values())
        cache_names = ["prefix_pad_masks", PADDED_STATE, *cache]
        noise = self._noise.to(self._device)
        x_t = noise
        with torch.no_grad():
            for timestep in self._timesteps:
                x_t = step(*cache_inputs, x_t, torch.tensor([timestep], device=self._device))
        step_inputs = (*cache_inputs, noise, torch.tensor(self._timesteps[:1], device=self._device))
        yield Part("step", step, step_inputs, [*cache_names, "x_t", "timestep"], ["x_next"])
        x_names = [NOISE, *(f"x_{i + 1}" for i in range(len(self._timesteps)))]
        self._runs += [
            ProgramRun(f"step_{i}", "step", [*cache_names, x_names[i], f"timestep_{i}"], [x_names[i + 1]])
            for i in range(len(self._timesteps))
        ]
        del step, step_inputs, cache, cache_inputs

        actions = PI05Actions(
            self.postprocessor,
            self._steps,
            cast("dict[str, PolicyFeature]", config.output_features)[ACTION].shape[0],
        )
        raw_state = torch.from_numpy(self._frame[self._state_name]).to(self._device)
        inputs, names = ((x_t, raw_state), [x_names[-1], self._state_name])
        if actions.mask is None:
            # A program takes only the inputs it reads.
            inputs, names = inputs[:1], names[:1]
        with torch.no_grad():
            self._actions = actions(*inputs).cpu().numpy()
        yield self._part("actions", actions, inputs, names, [ACTION])

    def test_case(self) -> TestCase:
        """The random frame and noise, and the actions the parts compute for them in PyTorch."""
        if self._actions is None:
            raise RuntimeError("The test case is made by `parts`, which has not run yet.")
        width = self._actions.shape[-1]
        postprocessor, _ = split_absolute_step(self.postprocessor, width)
        action_scale = measure_action_scale(
            lambda value: postprocessor(torch.full((1, 1, width), value, device=self._device))[0, 0]
        )
        case = TestCase(self._frame, self._noise.numpy(), self._actions, action_scale)
        self._check_against_policy(case)
        return case

    def plan(self) -> RunPlan:
        """The parts in order with the step run once per timestep, and the timesteps."""
        return RunPlan(
            runs=self._runs,
            constants={f"timestep_{i}": np.array([t], np.float32) for i, t in enumerate(self._timesteps)},
        )

    def _part(
        self,
        name: str,
        module: nn.Module,
        inputs: tuple[Tensor, ...],
        input_names: list[str],
        output_names: list[str],
    ) -> Part:
        self._runs.append(ProgramRun(name, name, input_names, output_names))
        return Part(name, module, inputs, input_names, output_names)

    def _load(self, *modules: str) -> PI0Policy:
        """The policy with only `modules`' weights read from the checkpoint; every other module is None."""
        return load_checkpoint_modules(PI0Policy, str(self._model_file), self._policy_config, list(modules))

    def _task_token_ids(self) -> Tensor:
        """The ids of the task's tokens that the policy attends to, as its own processor makes them."""
        observation = prepare_observation_for_inference(dict(self._frame), self._device, self.task)
        with torch.no_grad():
            batch = self.preprocessor(observation)
        self.preprocessor.reset()
        return batch[OBS_LANGUAGE_TOKENS][0][batch[OBS_LANGUAGE_ATTENTION_MASK][0].bool()]

    def _tensor_preprocessor(self) -> PolicyProcessorPipeline:
        """The preprocessor without the steps that make the prompt text and tokenize it, done once at export."""
        text_steps = (Pi0NewLineProcessor, TokenizerProcessorStep)
        return replace(
            self.preprocessor, steps=[s for s in self.preprocessor.steps if not isinstance(s, text_steps)]
        )

    def _check_against_policy(self, case: TestCase) -> None:
        """Compare the test case with the whole policy's `predict_action_chunk`, when the policy fits in memory."""
        if not whole_policy_fits(self._model_file, self._device, "pi0"):
            return
        policy = PI0Policy.from_pretrained(
            str(self._policy_config.pretrained_path),
            config=self._policy_config,
            revision=self._policy_config.pretrained_revision,
        ).eval()
        self.preprocessor.reset()
        self.postprocessor.reset()
        observation = prepare_observation_for_inference(dict(case.frame), self._device, self.task)
        with torch.no_grad():
            batch = self.preprocessor(observation)
            noise = torch.from_numpy(case.noise).to(self._device)[None]
            predicted = policy.predict_action_chunk(batch, noise=noise)[:, : self._steps]
            expected = self.postprocessor(predicted)[0].float().cpu().numpy()
        del policy
        error = float((np.abs(case.actions - expected) / case.action_scale).max())
        logger.info(
            "Test case against the whole policy: largest difference %.2e (tolerance %g)",
            error,
            self.tolerance,
        )
        if error > self.tolerance:
            raise ValueError(
                f"The chained parts compute other actions than pi0's predict_action_chunk ({error:.2e} > "
                f"{self.tolerance} in normalized units), so they do not export this checkpoint correctly."
            )
