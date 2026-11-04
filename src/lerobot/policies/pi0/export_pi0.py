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

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor, nn

from lerobot.export.recipe import to_policy_input
from lerobot.policies import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, TokenizerProcessorStep
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import _transformers_available

from ..pi05.export_pi05 import PaliGemmaChainRecipe
from . import modeling_pi0
from .modeling_pi0 import PI0Policy, PI0Pytorch
from .processor_pi0 import Pi0NewLineProcessor

if TYPE_CHECKING or _transformers_available:
    from transformers import DynamicCache

if TYPE_CHECKING:
    from lerobot.export.configs import ExportConfig

    from ..pi05.modeling_pi05 import PI05Policy, PI05Pytorch


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


class PI0ExportRecipe(PaliGemmaChainRecipe):
    """pi0 as a chain of small programs: see this module's documentation.

    Args:
        cfg (`ExportConfig`):
            The `lerobot-export` configuration.
    """

    policy_class = PI0Policy
    policy_name = "pi0"
    step_modules = STEP_MODULES
    step_inputs = (PADDED_STATE,)
    text_steps = (Pi0NewLineProcessor, TokenizerProcessorStep)

    def __init__(self, cfg: ExportConfig) -> None:  # noqa: D107
        super().__init__(cfg)
        self._task_ids = self._task_token_ids()

    def _embed(
        self, policy: PI05Policy | PI0Policy, table: Tensor, frame_names: list[str]
    ) -> tuple[nn.Module, tuple[Tensor, ...], list[str]]:
        """The frame in; the program holds the rows of the exported task's tokens."""
        task_rows = table.detach()[self._task_ids][None]
        embed = PI0Embed(
            cast(PI0Policy, policy), self._tensor_preprocessor(), task_rows, frame_names, self._visual
        )
        return embed, (), []

    def _step(self, model: PI05Pytorch | PI0Pytorch) -> nn.Module:
        return PI0Step(cast(PI0Pytorch, model), self._dt)

    def _depth(self) -> int:
        return modeling_pi0.get_gemma_config(self._policy_config.paligemma_variant).depth

    def _task_token_ids(self) -> Tensor:
        """The ids of the task's tokens that the policy attends to, as its own processor makes them."""
        observation = prepare_observation_for_inference(dict(self._frame), self._device, self.task)
        with torch.no_grad():
            batch = self.preprocessor(observation)
        self.preprocessor.reset()
        return batch[OBS_LANGUAGE_TOKENS][0][batch[OBS_LANGUAGE_ATTENTION_MASK][0].bool()]
