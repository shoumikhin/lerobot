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

"""Export pi0.5 as a chain of small programs, so that it builds and runs on a board with 8 GB of memory.

One action chunk runs as these programs, which pass their values to each other by name:

::

    embed                the frame and the task's token rows -> the prefix's embeddings and padding mask
    language_model_0..k  3 layers of the language model each -> the next hidden states and their KV cache
    step x10             the KV cache, the noisy actions and a timestep -> the next noisy actions
    actions              the denoised actions -> the actions in robot units

Each part is built with only its own weights in memory. pi0.5 reads the task and the state, as bin numbers,
in its prompt. The bundle turns the task into the embedding rows of its tokens with the policy's own tokenizer
and embedding table (`task_tokens`), so it runs any task. The `embed` program adds the state's tokens after the
task's, from the few embedding rows the state and the end of the prompt can use, so it holds no copy of the
table.
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

from lerobot.bundle.manifest import NOISE, TASK_EMBEDDINGS, TASK_LENGTH, ProgramRun
from lerobot.bundle.task_tokens import clean, tokenize
from lerobot.configs import FeatureType
from lerobot.export.recipe import (
    FLOAT32_TOLERANCE,
    ExportRecipe,
    Part,
    RunPlan,
    TaskTokenData,
    TestCase,
    dataset_metadata,
    export_task,
    frame_shapes,
    load_processors,
    measure_action_scale,
    preprocessor_renames,
    random_frame,
    robot_features,
    to_policy_input,
    whole_policy_fits,
)
from lerobot.policies import prepare_observation_for_inference
from lerobot.policies.common.openpi_checkpoint import load_checkpoint_modules
from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
from lerobot.policies.pi_gemma import PiGemmaRMSNorm
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    DeviceProcessorStep,
    PolicyProcessorPipeline,
    TokenizerProcessorStep,
    TransitionKey,
    create_transition,
    to_absolute_actions,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import _transformers_available

from . import modeling_pi05
from .modeling_pi05 import PI05Policy, PI05Pytorch
from .processor_pi05 import Pi05PrepareStateTokenizerProcessorStep

if TYPE_CHECKING or _transformers_available:
    from transformers import DynamicCache

if TYPE_CHECKING:
    from lerobot.configs import PolicyFeature
    from lerobot.export.configs import ExportConfig

    from .configuration_pi05 import PI05Config

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
    "model.action_in_proj",
    "model.action_out_proj",
    "model.time_mlp_in",
    "model.time_mlp_out",
)
# The most language model layers whose build fits an 8 GB Jetson Orin Nano beside the rest of the export.
LANGUAGE_LAYERS_PER_PART = 3
# How `Pi05PrepareStateTokenizerProcessorStep` makes the prompt text before the state; `_check_prompt_ops`
# holds it to that.
TASK_PROMPT_OPS = [
    ["strip"],
    ["replace", "_", " "],
    ["replace", "\n", " "],
    ["template", "Task: {task}, State:"],
]
# The text after the state's bin numbers.
PROMPT_END = ";\nAction: "
# Tokens of room for tasks not known at export, when --max_task_tokens is not given.
TASK_TOKEN_MARGIN = 16
# Tasks whose prompt text is compared with the policy's processor at export, beside the known tasks.
EDGE_CASE_TASKS = ("", "!!!", "pick_up_the_cube", "pick up\nthe cube", "  Pick up the cube.  ")
# States far outside the training data, so the prompt check covers the first and last bins.
EXTREME_STATE = 1e6
# Random frames whose prompt is compared with the policy's processor at export, beside the test frame.
PROMPT_CHECK_FRAMES = 3
# Ten bfloat16 denoising steps amplify rounding: the LIBERO checkpoint's TensorRT bundle differed from its test
# case by 0.11 normalized, where eager PyTorch with another attention kernel moved the same frame by 0.03.
BFLOAT16_STEPS_TOLERANCE = 0.2


class StatePrompt(nn.Module):
    """The task and the normalized state as pi0.5's prompt, given as the embedding rows of its tokens.

    The task's rows come in, looked up by the bundle. Those of the state's bin numbers and of the end of the
    prompt are the only others a prompt holds, so the module keeps just them. The policy pads its prompt to the
    tokenizer's length and leaves the padding out of attention; this prompt is padded only to the longest one
    with `max_task_tokens` task tokens, which gives the same actions in less time.

    Args:
        pieces (`list[list[int]]`):
            The token ids of the bin numbers -1 to 255, each with the space before it.
        end (`list[int]`):
            The token ids of the end of the prompt, after the state.
        pad_id (`int`):
            The tokenizer's padding token.
        table (`Tensor`):
            The embedding table, which only the rows of these tokens are copied from.
        state_dim (`int`):
            The size of the state.
        max_task_tokens (`int`):
            The most tokens the task's prompt text takes.
    """

    def __init__(
        self,
        pieces: list[list[int]],
        end: list[int],
        pad_id: int,
        table: Tensor,
        state_dim: int,
        max_task_tokens: int,
    ):
        super().__init__()
        width = max(map(len, pieces))
        length = max_task_tokens + state_dim * width + len(end)
        ids = sorted({pad_id, *end, *(i for piece in pieces for i in piece)})
        local = {token: index for index, token in enumerate(ids)}
        # Row i of the rows the prompt draws from is task row i, then come these.
        offset = max_task_tokens
        self.local_ids = ids
        self.pad = offset + local[pad_id]
        device = table.device
        self.register_buffer("rows", table.detach()[torch.tensor(ids, device=device)].clone())
        self.register_buffer("edges", -1 + torch.arange(256, dtype=torch.float32, device=device) / 128)
        self.register_buffer("end", torch.tensor([offset + local[i] for i in end], device=device))
        padded = [[offset + local[i] for i in piece] + [self.pad] * (width - len(piece)) for piece in pieces]
        self.register_buffer("pieces", torch.tensor(padded, device=device))
        self.register_buffer("lengths", torch.tensor(list(map(len, pieces)), device=device))
        self.register_buffer("piece_positions", torch.arange(width, device=device))
        self.register_buffer("task_positions", torch.arange(max_task_tokens, device=device))
        self.register_buffer("positions", torch.arange(length, device=device))

    def forward(self, state: Tensor, task_embeddings: Tensor, task_length: Tensor) -> tuple[Tensor, Tensor]:
        """The prompt's token rows `[1, length, width]` and its padding mask `[1, length]`."""
        indices, mask = self.indices(state, task_length)
        rows = torch.cat((task_embeddings.to(self.rows.dtype), self.rows))
        return rows[indices][None], mask[None]

    def indices(self, state: Tensor, task_length: Tensor) -> tuple[Tensor, Tensor]:
        """Which row each position of the prompt takes, and which positions are not padding."""
        # np.digitize against 256 equal bins of [-1, 1], as the policy's processor does.
        bins = (state.flatten()[..., None] >= self.edges).sum(-1) - 1
        pieces = self.pieces[bins + 1].flatten()
        valid = (self.piece_positions < self.lengths[bins + 1, None]).flatten()
        tokens = torch.cat((self.task_positions, pieces, self.end))
        valid = torch.cat(
            (self.task_positions < task_length, valid, torch.ones_like(self.end, dtype=torch.bool))
        )
        # Each valid token owns one output position; unused task and piece slots own none.
        compact = (valid.cumsum(0)[:, None] - 1 == self.positions) & valid[:, None]
        indices = (tokens[:, None] * compact).sum(0)
        mask = self.positions < valid.sum()
        return torch.where(mask, indices, self.pad), mask

    def token_ids(self, state: Tensor, task_ids: list[int]) -> list[int]:
        """The token ids of the prompt's valid positions, as `forward` places their rows."""
        indices, mask = self.indices(state, torch.tensor([len(task_ids)], device=state.device))
        padded = task_ids + [0] * (len(self.task_positions) - len(task_ids))
        ids = torch.tensor(padded + self.local_ids, device=state.device)
        return ids[indices][mask].tolist()


class LookedUpEmbedding(nn.Module):
    """The language model's token embedding, given the rows its token ids pick instead of the ids.

    Its forward is the embedding's own after the table lookup: the scale a Gemma embedding multiplies its rows
    by, in the table's dtype.
    """

    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.dtype = embedding.weight.dtype
        scale = getattr(embedding, "embed_scale", None)
        self.register_buffer("scale", None if scale is None else scale.detach().to(self.dtype))

    def forward(self, rows: Tensor) -> Tensor:
        rows = rows.to(self.dtype)
        return rows if self.scale is None else rows * self.scale


class PI05Embed(nn.Module):
    """The raw frame and the task's rows in, the prefix's embeddings and padding mask out.

    The policy fills a camera the frame does not hold with an image of -1 that it leaves out of attention, and
    puts it after the others. Such a camera adds nothing and shifts no position, so the program leaves it out,
    which gives the same actions without running the vision tower on it.
    """

    def __init__(
        self,
        policy: PI05Policy,
        preprocessor: PolicyProcessorPipeline,
        prompt: StatePrompt,
        frame_names: list[str],
        visual: set[str],
    ):
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.prompt = prompt
        self.frame_names = frame_names
        self.visual = visual

    def forward(self, *inputs: Tensor) -> tuple[Tensor, Tensor]:
        *frame, task_embeddings, task_length = inputs
        batch = self.preprocessor(
            {
                name: to_policy_input(x, name in self.visual)
                for name, x in zip(self.frame_names, frame, strict=True)
            }
        )
        # The language tokens are the embedding rows of the prompt's tokens; see LookedUpEmbedding.
        tokens, masks = self.prompt(batch[OBS_STATE], task_embeddings, task_length)
        # The policy's own image resize and padding, which it exposes under no public name.
        images, img_masks = self.policy._preprocess_images(batch)
        cameras = len(self.visual)
        prefix_embs, prefix_pad_masks, _ = self.policy.model.embed_prefix(
            images[:cameras], img_masks[:cameras], tokens, masks
        )
        return prefix_embs, prefix_pad_masks


class PI05LanguageModelLayers(nn.Module):
    """Some of the language model's layers over the prefix: the hidden states in, the next ones and their KV
    cache out.

    pi0.5's prefix attends both ways inside the prompt, so the mask and positions come from the padding mask
    alone. Full attention over the whole prompt is what SDPA expresses, so the layers use it.
    """

    def __init__(self, model: PI05Pytorch, layers: range):
        super().__init__()
        language_model = model.paligemma_with_expert.paligemma.model.language_model
        language_model.set_attn_implementation("sdpa")
        self.layers = nn.ModuleList(language_model.layers[i] for i in layers)
        self.rotary_emb = language_model.rotary_emb
        self.last = layers.stop == len(language_model.layers)

    def forward(self, hidden_states: Tensor, prefix_pad_masks: Tensor) -> tuple[Tensor, ...]:
        if self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
        att_masks = torch.zeros_like(prefix_pad_masks)
        # PyTorch's attention kernels take an additive mask only in the attention's own dtype.
        attention_mask = prepare_attention_masks_4d(
            make_att_2d_masks(prefix_pad_masks, att_masks), dtype=hidden_states.dtype
        )
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        cache = DynamicCache()
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
        # Each layer writes its keys and values at its index in the whole model.
        own = [cache.layers[layer.self_attn.layer_idx] for layer in self.layers]
        keys_values = tuple(tensor for layer in own for tensor in (layer.keys, layer.values))
        # The steps read only the KV cache, so the last layer's attention output and MLP would be dead weight.
        return keys_values if self.last else (hidden_states, *keys_values)


class PI05Step(nn.Module):
    """One step of `denoise_step` with its Euler update: the prefix's mask and KV cache, the noisy actions and
    the timestep in, the next noisy actions out.

    The time conditions the action expert only through its adaptive norms, whose scale, shift and gate depend on
    the time alone, and the bundle fixes the schedule. So the step holds them for each of the schedule's
    timesteps, computed once in float32, instead of the time MLP and every norm's projection, which would be
    about a quarter of the weights each step reads. It runs only the schedule's timesteps.

    A few queries read a long cache here, which plain matrix products express, so the expert keeps the eager
    attention `denoise_step` sets.
    """

    def __init__(self, model: PI05Pytorch, timesteps: list[float], dt: float, bfloat16: bool):
        super().__init__()
        self.model = model
        self.dt = dt
        self.bfloat16 = bfloat16
        device = model.action_in_proj.weight.device
        self.register_buffer("timesteps", torch.tensor(timesteps, device=device))
        expert = model.paligemma_with_expert.gemma_expert.model
        # The time MLP runs in float32 here, whatever dtype the checkpoint holds it in.
        model.time_mlp_in.float()
        model.time_mlp_out.float()
        x_t = torch.zeros(
            len(timesteps),
            model.config.chunk_size,
            model.config.max_action_dim,
            dtype=model.action_in_proj.weight.dtype,
            device=device,
        )
        with torch.no_grad():
            *_, cond = model.embed_suffix(x_t, self.timesteps)

            def scheduled(norm: PiGemmaRMSNorm) -> PiGemmaRMSNorm:
                # Fed the step's one-hot row, the new projection returns that step's modulation exactly.
                table = PiGemmaRMSNorm(norm.dim, norm.eps, cond_dim=len(timesteps)).to(device)
                table.dense.weight.copy_(
                    nn.functional.linear(cond.float(), norm.dense.weight.float(), norm.dense.bias.float()).T
                )
                table.dense.bias.zero_()
                return table

            for layer in expert.layers:
                layer.input_layernorm = scheduled(layer.input_layernorm)
                layer.post_attention_layernorm = scheduled(layer.post_attention_layernorm)
            expert.norm = scheduled(expert.norm)
        del model.time_mlp_in, model.time_mlp_out
        if bfloat16:
            _hold_linear_weights_in_bfloat16(model)

    def forward(self, prefix_pad_masks: Tensor, *inputs: Tensor) -> Tensor:
        *cache, x_t, timestep = inputs
        past_key_values = DynamicCache(
            tuple((keys, values, None) for keys, values in zip(cache[::2], cache[1::2], strict=True))
        )
        # On a GPU, a bfloat16 policy runs every matrix product of the step in bfloat16, including the adaptive
        # norms the checkpoint keeps in float32, which halves what the step reads.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bfloat16):
            velocity = self.model.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t[None],
                timestep,
                (self.timesteps == timestep[:, None]).float(),
            )
        return x_t + self.dt * velocity[0].float()


class PI05Actions(nn.Module):
    """The denoised actions in, the actions to play out, in robot units.

    A policy trained on actions relative to the state turns them back around the robot's state. The policy's
    postprocessor adds the state its preprocessor cached in Python, which a program would hold as a constant,
    so this module adds the state it takes as an input instead.
    """

    def __init__(self, postprocessor: PolicyProcessorPipeline, execution_steps: int, action_dim: int):
        super().__init__()
        self.postprocessor, absolute_step = _split_absolute_step(postprocessor)
        self.execution_steps = execution_steps
        self.action_dim = action_dim
        self.mask: list[bool] | None = None
        if absolute_step is not None:
            if absolute_step.relative_step is None:
                raise ValueError(
                    "The postprocessor's relative-to-absolute step is linked to no relative step."
                )
            # The relative step has no public mask accessor; LeRobot's RTC and statistics code call this one too.
            self.mask = absolute_step.relative_step._build_mask(action_dim)

    def forward(self, actions: Tensor, *state: Tensor) -> Tensor:
        """The actions, and the robot's state for a policy trained on actions relative to it."""
        actions = self.postprocessor(actions[None, : self.execution_steps, : self.action_dim])
        if self.mask is not None:
            (anchor,) = state
            actions = to_absolute_actions(actions, anchor[None], self.mask)
        return actions[0].float()


class PI05ExportRecipe(ExportRecipe):
    """pi0.5 as a chain of small programs that runs any task: see this module's documentation.

    It builds the policy on the meta device and loads each part's weights when the part is made, so the trace
    holds one part's weights at a time. The test case's actions are the parts run one after another in PyTorch.
    When the whole policy fits in memory, they are also compared with the policy's own `predict_action_chunk`.

    Args:
        cfg (`ExportConfig`):
            The `lerobot-export` configuration.
    """

    @classmethod
    def load(cls, cfg: ExportConfig) -> PI05ExportRecipe:
        """The recipe, with no weight loaded yet."""
        return cls(cfg)

    def __init__(self, cfg: ExportConfig) -> None:  # noqa: D107
        from transformers.utils import cached_file

        policy_config = cast("PI05Config", cfg.policy)
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
            else BFLOAT16_STEPS_TOLERANCE
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
        self._state_dim = frame_shapes(self._features)[self._state_name][0]
        exported = self.task = export_task(cfg, self._metadata, "pi0.5")
        dataset_tasks = [] if self._metadata is None else [str(t) for t in self._metadata.tasks.index]
        known_tasks = list(dict.fromkeys([exported, *dataset_tasks]))
        for task in known_tasks:
            if not clean(TASK_PROMPT_OPS[:-1], task):
                raise ValueError(f"The task {task!r} is empty once cleaned, so pi0.5's prompt holds no task.")

        self._tokenizer_step = next(
            s for s in self.preprocessor.steps if isinstance(s, TokenizerProcessorStep)
        )
        if self._tokenizer_step.tokenizer_name is None:
            raise ValueError(
                "pi0.5's tokenizer step names no tokenizer, so its tokenizer file cannot be copied."
            )
        self._tokenizer_file = Path(cached_file(self._tokenizer_step.tokenizer_name, "tokenizer.json"))
        self._check_prompt_ops(known_tasks)
        ids = tokenize(self._tokenizer_file, [clean(TASK_PROMPT_OPS, task) for task in known_tasks])
        self._known_tasks = dict(zip(known_tasks, ids, strict=True))
        tokenizer = self._tokenizer_step.input_tokenizer
        self._pieces = [tokenizer.encode(f" {value}", add_special_tokens=False) for value in range(-1, 256)]
        self._end = tokenizer.encode(PROMPT_END, add_special_tokens=False)
        # The longest task prompt that leaves room for the widest state and the end of the prompt.
        window = (
            self._tokenizer_step.max_length - self._state_dim * max(map(len, self._pieces)) - len(self._end)
        )
        longest = max(map(len, ids))
        self._max_tokens = (
            cfg.max_task_tokens
            if cfg.max_task_tokens is not None
            else min(window, longest + TASK_TOKEN_MARGIN)
        )
        if self._tokenizer_step.padding_side != "right" or not 0 < self._max_tokens <= window:
            raise ValueError(
                f"The task prompt must fit in the policy's right-padded prompt: at most {window} tokens."
            )
        self._task_ids = self._known_tasks[exported]
        if len(self._task_ids) > self._max_tokens:
            raise ValueError(
                f"The task {exported!r} is {len(self._task_ids)} tokens long, more than the "
                f"{self._max_tokens} of --max_task_tokens."
            )
        too_long = [task for task, task_ids in self._known_tasks.items() if len(task_ids) > self._max_tokens]
        if too_long:
            logger.warning("These known tasks are longer than %d tokens: %s", self._max_tokens, too_long)

        self._frame = random_frame(self._features, self._generator)
        self._noise = torch.randn(
            policy_config.chunk_size, policy_config.max_action_dim, generator=self._generator
        )
        self._dt = -1.0 / policy_config.num_inference_steps
        # The schedule of `euler_integrate`.
        self._timesteps = [1.0 + step * self._dt for step in range(policy_config.num_inference_steps)]
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
            _hold_linear_weights_in_bfloat16(vlm.vision_tower)
            _hold_linear_weights_in_bfloat16(vlm.multi_modal_projector)
        language_model = policy.model.paligemma_with_expert.paligemma.model.language_model
        embedding = language_model.embed_tokens
        prompt = StatePrompt(
            self._pieces,
            self._end,
            self._tokenizer_step.input_tokenizer.pad_token_id,
            embedding.weight,
            self._state_dim,
            self._max_tokens,
        )
        task_inputs = self._task_inputs(self._task_ids, embedding.weight)
        # The bundle looks the task's rows up in the full table, so the program drops it.
        language_model.embed_tokens = LookedUpEmbedding(embedding)
        del embedding
        embed = PI05Embed(policy, self._tensor_preprocessor(), prompt, frame_names, self._visual)
        self._check_prompt(embed)
        inputs = (*frame, *task_inputs)
        with torch.no_grad():
            hidden, prefix_pad_masks = embed(*inputs)
        names = [*frame_names, TASK_EMBEDDINGS, TASK_LENGTH]
        yield self._part("embed", embed, inputs, names, ["prefix_embs", "prefix_pad_masks"])
        del policy, language_model, embed, prompt, inputs

        hidden_name = "prefix_embs"
        cache: dict[str, Tensor] = {}
        depth = modeling_pi05.get_gemma_config(config.paligemma_variant).depth
        for group, first in enumerate(range(0, depth, LANGUAGE_LAYERS_PER_PART)):
            layers = range(first, min(first + LANGUAGE_LAYERS_PER_PART, depth))
            module = PI05LanguageModelLayers(
                self._load(*(f"{LANGUAGE_MODEL}.layers.{i}" for i in layers)).model, layers
            )
            inputs = (hidden, prefix_pad_masks)
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

        bfloat16 = config.dtype == torch.bfloat16 and self._device.type == "cuda"
        step = PI05Step(self._load(*STEP_MODULES).model, self._timesteps, self._dt, bfloat16)
        cache_inputs = (prefix_pad_masks, *cache.values())
        cache_names = ["prefix_pad_masks", *cache]
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
        state = torch.from_numpy(self._frame[self._state_name]).to(self._device)
        inputs, names = ((x_t, state), [x_names[-1], self._state_name])
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
        postprocessor, _ = _split_absolute_step(self.postprocessor)
        width = self._actions.shape[-1]
        action_scale = measure_action_scale(
            lambda value: postprocessor(torch.full((1, 1, width), value, device=self._device))[0, 0]
        )
        case = TestCase(self._frame, self._noise.numpy(), self._actions, action_scale)
        self._check_against_policy(case)
        return case

    def plan(self) -> RunPlan:
        """The parts in order with the step run once per timestep, the timesteps, and the task's tokens."""
        # The bundle writes the table from the CPU, so it is read there, without a copy on the device.
        config = copy(self._policy_config)
        config.device = "cpu"
        policy = load_checkpoint_modules(PI05Policy, str(self._model_file), config, [EMBED_TOKENS])
        table = policy.model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight.detach()
        del policy
        return RunPlan(
            runs=self._runs,
            constants={f"timestep_{i}": np.array([t], np.float32) for i, t in enumerate(self._timesteps)},
            task_tokens=TaskTokenData(
                TASK_PROMPT_OPS, self._tokenizer_file, table, self._known_tasks, self._max_tokens
            ),
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

    def _load(self, *modules: str) -> PI05Policy:
        """The policy with only `modules`' weights read from the checkpoint; every other module is None."""
        return load_checkpoint_modules(PI05Policy, str(self._model_file), self._policy_config, list(modules))

    def _tensor_preprocessor(self) -> PolicyProcessorPipeline:
        """The preprocessor without the steps that make the prompt text and tokenize it, which StatePrompt does."""
        text_steps = (Pi05PrepareStateTokenizerProcessorStep, TokenizerProcessorStep)
        return replace(
            self.preprocessor, steps=[s for s in self.preprocessor.steps if not isinstance(s, text_steps)]
        )

    def _task_inputs(self, ids: list[int], table: Tensor) -> tuple[Tensor, Tensor]:
        """What the bundle gives the programs for a task: its prompt's embedding rows, padded, and their count."""
        rows = torch.zeros(self._max_tokens, table.shape[1], device=self._device)
        rows[: len(ids)] = table[torch.tensor(ids, device=table.device)].float()
        return rows, torch.tensor([len(ids)], device=self._device)

    def _check_prompt_ops(self, tasks: list[str]) -> None:
        """Refuse prompt ops that make another prompt text than the policy's processor, for any task tried.

        The processor's prompt for a task is the ops' text, then the state and the end of the prompt, which do
        not depend on the task.
        """
        step = next(
            s for s in self.preprocessor.steps if isinstance(s, Pi05PrepareStateTokenizerProcessorStep)
        )
        state = torch.zeros(1, self._state_dim)

        def processor_text(task: str) -> str:
            transition = create_transition(
                observation={OBS_STATE: state}, complementary_data={step.task_key: [task]}
            )
            return cast(dict, step(transition)[TransitionKey.COMPLEMENTARY_DATA])[step.task_key][0]

        tail = processor_text("").removeprefix(clean(TASK_PROMPT_OPS, ""))
        for task in [*tasks, *EDGE_CASE_TASKS]:
            if processor_text(task) != clean(TASK_PROMPT_OPS, task) + tail:
                raise ValueError(
                    f"pi0.5's processor makes the prompt {processor_text(task)!r} of the task {task!r}, but the "
                    f"bundle would make {clean(TASK_PROMPT_OPS, task) + tail!r}."
                )

    def _check_prompt(self, embed: PI05Embed) -> None:
        """Refuse a prompt that the program builds with other token ids than the policy's processor.

        StatePrompt assumes the tokenizer splits each bin number the same inside the prompt as alone. This
        compares the two on the test frame, a few random frames, and states beyond the first and last bins.
        """
        frames = [
            self._frame,
            *(random_frame(self._features, self._generator) for _ in range(PROMPT_CHECK_FRAMES)),
        ]
        for value in (-EXTREME_STATE, EXTREME_STATE):
            extreme = dict(self._frame)
            extreme[self._state_name] = np.full_like(self._frame[self._state_name], value)
            frames.append(extreme)
        for frame in frames:
            observation = prepare_observation_for_inference(dict(frame), self._device, self.task)
            with torch.no_grad():
                batch = self.preprocessor(observation)
                normalized = embed.preprocessor(
                    {
                        name: to_policy_input(torch.from_numpy(x).to(self._device), name in self._visual)
                        for name, x in frame.items()
                    }
                )[OBS_STATE]
            mask = batch[OBS_LANGUAGE_ATTENTION_MASK][0].bool()
            expected = batch[OBS_LANGUAGE_TOKENS][0][mask].tolist()
            actual = embed.prompt.token_ids(normalized, self._task_ids)
            if actual != expected:
                raise ValueError(
                    "The exported prompt holds other tokens than pi0.5's processor makes for the state "
                    f"{frame[self._state_name].tolist()}: {actual} instead of {expected}. This tokenizer splits "
                    "the state's numbers differently inside the prompt, so pi0.5 cannot be exported this way."
                )
        self.preprocessor.reset()

    def _check_against_policy(self, case: TestCase) -> None:
        """Compare the test case with the whole policy's `predict_action_chunk`, when the policy fits in memory."""
        if not whole_policy_fits(self._model_file, self._device):
            return
        policy = PI05Policy.from_pretrained(
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
                f"The chained parts compute other actions than pi0.5's predict_action_chunk ({error:.2e} > "
                f"{self.tolerance} in normalized units), so they do not export this checkpoint correctly."
            )


def _split_absolute_step(
    postprocessor: PolicyProcessorPipeline,
) -> tuple[PolicyProcessorPipeline, AbsoluteActionsProcessorStep | None]:
    """The postprocessor's steps before its enabled relative-to-absolute step, and that step."""
    steps = list(postprocessor.steps)
    index = next(
        (i for i, s in enumerate(steps) if isinstance(s, AbsoluteActionsProcessorStep) and s.enabled), None
    )
    if index is None:
        return postprocessor, None
    if not all(isinstance(s, DeviceProcessorStep) for s in steps[index + 1 :]):
        raise ValueError("The relative-to-absolute action step must be the postprocessor's last step.")
    return replace(postprocessor, steps=steps[:index]), cast(AbsoluteActionsProcessorStep, steps[index])


def _hold_linear_weights_in_bfloat16(module: nn.Module) -> None:
    """Store the linear layers of `module`, which runs them only under bfloat16 autocast, in bfloat16.

    Autocast casts a float32 linear layer's weight to bfloat16 on every call, so this is the same arithmetic
    without a float32 copy of the weights in the program.
    """
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            layer.to(torch.bfloat16)
