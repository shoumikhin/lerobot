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

"""Export LingBot-VA as small programs that run its streaming inference, chunk after chunk.

LingBot-VA predicts a chunk of video latents, then a chunk of actions, with one dual-stream transformer. It keeps
what it saw since the episode began: a KV cache of every past chunk's observed latents and executed actions, the
causal cache of its streaming video encoder (VAE), and the frame index of the next chunk. The bundle keeps them
too: the runs of a chunk write them, and the engine carries them to the next chunk (`RunPlan.carry`). The camera
frames the robot sent since the last chunk come from the bundle's history (`RunPlan.history`), and the two noises
the policy draws for a chunk are one flat noise input.

One chunk runs as these programs, which pass their values to each other by name:

::

    clear_predicted, encode_next     drop the cache's predicted entries; encode the frames seen since the last chunk
    alloc_video, video_in, blocks, write_video      write the observed latents into the KV cache
    alloc_action, action_in, blocks, write_action   write the executed actions into the KV cache
    advance, start                   the next frame index; the noise -> the starting video latents and actions
    alloc_video, (video_in, blocks, video_out) x steps, video_in, blocks, write_video     denoise the video
    alloc_action, (action_in, blocks, action_out) x steps, action_in, blocks, write_action  denoise the actions
    actions                          the actions in robot units, and the executed actions for the next chunk

The first chunk after a reset encodes the current frame instead, conditions its first frame on it and starts from
an empty cache; the second encodes the frames seen since and puts that first latent before them (`first_runs`).
Every pass of both streams runs the same `blocks` program, so the bundle holds the transformer's weights once: the
pass with fewer tokens is padded to as many as the other and its padding is masked out of the attention. The cache
is static: pools of every slot, which slots hold a key, how old each is and whether it was predicted. Slots are
taken and freed the way the policy's own cache takes and frees them.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor, nn

from lerobot.bundle.manifest import NOISE, ProgramRun
from lerobot.export.recipe import (
    BFLOAT16_STEPS_TOLERANCE,
    ActionChunkRecipe,
    Part,
    RunPlan,
    TestCase,
    export_task,
    preprocessor_renames,
    random_frame,
    to_policy_input,
)
from lerobot.processor import (
    PolicyProcessorPipeline,
    policy_output_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import ACTION

from .utils import FlowMatchScheduler, _vae_patchify, data_seq_to_patch, get_mesh_id

if TYPE_CHECKING:
    from lerobot.export.configs import ExportConfig

    from .modeling_lingbot_va import LingBotVAPolicy
    from .utils import WanAttention, WanTimeTextImageEmbedding, WanTransformer3DModel

logger = logging.getLogger(__name__)

# The first chunk, the second, and one like every later chunk.
CHUNKS = 3
# The Wan2.2 VAE encodes 4 camera frames into one latent frame.
VAE_FRAMES = 4

KEYS, VALUES, VALID, IDS, PREDICTED = "kv_keys", "kv_values", "kv_valid", "kv_ids", "kv_predicted"
CACHE = [KEYS, VALUES, VALID, IDS, PREDICTED]
FRAME_START = "frame_start"
VAE_CACHE = "vae_cache"
FIRST_LATENT = "first_latent"
EXECUTED = "executed"
CARRY = [*CACHE, FRAME_START, VAE_CACHE, FIRST_LATENT, EXECUTED]
FIRST, LATER = "first", "later"
TOKENS = ("video_tokens", "action_tokens")


class _Encode(nn.Module):
    """Some of the camera frames the robot sent in; their normalized video latents and the encoder's cache out.

    It does what the policy's `_encode_frames` does with its streaming VAE, with the causal cache of the encoder as
    an input and an output instead of a list the VAE keeps, packed in one tensor. The first call after a reset
    starts from no cache and keeps one frame where the later calls keep two; the frame it lacks is zeros, which is
    what the causal convolutions pad with, so every later call takes a cache of the same shape.

    Args:
        policy (`LingBotVAPolicy`):
            The policy, with its frozen VAE loaded.
        preprocessor (`PolicyProcessorPipeline`):
            Its preprocessor.
        cameras (`list[str]`):
            The keys of the robot's frame that become the policy's cameras, in the policy's order.
        frames (`list[int]`):
            Which of each camera's last frames to encode.
        cache_shapes (`list[torch.Size]`):
            The shape of each entry of the encoder's cache after a call that is not the first.
        first (`bool`):
            Whether this is the first call after a reset, which takes no cache.
        first_latent (`bool`):
            Whether to put the first chunk's latent before the latents, as the second chunk does.
    """

    def __init__(
        self,
        policy: LingBotVAPolicy,
        preprocessor: PolicyProcessorPipeline,
        cameras: list[str],
        frames: list[int],
        cache_shapes: list[torch.Size],
        first: bool,
        first_latent: bool,
    ):
        super().__init__()
        vae = policy._vae
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        self.patch_size = getattr(vae.config, "patch_size", None)
        self.entries = len(policy._streaming_vae.feat_cache)
        self.camera_frame = policy._camera_frame
        self.normalize = policy._normalize_vae_latent
        self.keys = list(policy.config.obs_cam_keys)
        self.preprocessor = preprocessor
        self.cameras = cameras
        self.frames = frames
        self.cache_shapes = cache_shapes
        self.first = first
        self.first_latent = first_latent

    def forward(self, *inputs: Tensor) -> tuple[Tensor, Tensor]:
        histories, rest = inputs[: len(self.cameras)], list(inputs[len(self.cameras) :])
        cache: list[Tensor | None] = [None] * self.entries
        if not self.first:
            sizes = [math.prod(shape) for shape in self.cache_shapes]
            cache[: len(sizes)] = [
                part.view(shape)
                for part, shape in zip(rest.pop(0).split(sizes), self.cache_shapes, strict=True)
            ]
        per_camera: list[list[Tensor]] = [[] for _ in self.keys]
        for index in self.frames:
            # Each frame goes through the preprocessor as its own control tick did.
            observation: dict[str, Any] = {
                name: to_policy_input(history[index], True)
                for name, history in zip(self.cameras, histories, strict=True)
            }
            batch = self.preprocessor({**observation, "task": "", "robot_type": ""})
            for frames, key in zip(per_camera, self.keys, strict=True):
                frames.append(self.camera_frame(batch, key))
        videos = torch.cat([torch.cat(frames, dim=2) for frames in per_camera], dim=0)
        if self.patch_size is not None:
            videos = _vae_patchify(videos, self.patch_size)
        encoded = self.quant_conv(self.encoder(videos, feat_cache=cache, feat_idx=[0]))
        latents = self.normalize(encoded)
        latents = torch.cat(latents.split(1, dim=0), dim=-1)
        if self.first_latent:
            latents = torch.cat([rest.pop(0), latents], dim=2)
        entries = [entry for entry in cache if entry is not None]
        packed = torch.cat(
            [
                F.pad(entry, (0, 0, 0, 0, shape[2] - entry.shape[2], 0)).flatten()
                for entry, shape in zip(entries, self.cache_shapes, strict=True)
            ]
        )
        return latents, packed


class _EmptyCache(nn.Module):
    """The KV cache after a reset, with every slot free, and the frame index of the first chunk, 0."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def forward(self) -> tuple[Tensor, ...]:
        slots = self.shape[2]
        # Zeros, not empty memory: the blocks attend over every slot and mask out the free ones, and a NaN in empty
        # memory would leak through the mask.
        keys = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        values = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        valid = torch.zeros(slots, dtype=torch.bool, device=self.device)
        ids = torch.full((slots,), -1, dtype=torch.int64, device=self.device)
        predicted = torch.zeros(slots, dtype=torch.bool, device=self.device)
        frame_start = torch.zeros(1, dtype=torch.int64, device=self.device)
        return keys, values, valid, ids, predicted, frame_start


class _ClearPredicted(nn.Module):
    """Frees the slots the last chunk predicted, as `clear_pred_cache` does before the observed ones are written."""

    def forward(self, valid: Tensor, predicted: Tensor) -> Tensor:
        return valid & ~predicted


class _Advance(nn.Module):
    """The frame index of the chunk after the one the KV cache just observed."""

    def __init__(self, frames: int):
        super().__init__()
        self.frames = frames

    def forward(self, frame_start: Tensor) -> Tensor:
        return frame_start + self.frames


class _Start(nn.Module):
    """The flat noise in; the starting video latents and actions, in the model's dtype, out."""

    def __init__(self, video_shape: tuple[int, ...], action_shape: tuple[int, ...], dtype: torch.dtype):
        super().__init__()
        self.video_shape = video_shape
        self.action_shape = action_shape
        self.dtype = dtype

    def forward(self, noise: Tensor) -> tuple[Tensor, Tensor]:
        video, action = noise.split([math.prod(self.video_shape), math.prod(self.action_shape)])
        return video.view(self.video_shape).to(self.dtype), action.view(self.action_shape).to(self.dtype)


class _Allocate(nn.Module):
    """Takes `count` slots of the KV cache, as `WanAttention.allocate_slots` takes them.

    It takes the first free slots. When too few are free, it first frees the oldest valid slots, by id and then by
    slot; the policy frees whole groups of one id, so the order inside a group never matters. It returns the valid
    slots after that, the slots taken, and the id the keys written there get. It ranks the slots by comparing each
    with every other rather than by sorting them, as ExecuTorch's CPU kernels have no sort.
    """

    def __init__(self, count: int):
        super().__init__()
        self.count = count

    def forward(self, valid: Tensor, ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        index = torch.arange(valid.shape[0], device=valid.device)
        shortfall = (self.count - (~valid).sum()).clamp(min=0)
        age = torch.where(valid, ids, torch.iinfo(ids.dtype).max)
        before = (age < age[:, None]) | ((age == age[:, None]) & (index < index[:, None]))
        rank = before.sum(1)
        valid = valid & ~(rank < shortfall)
        free = ~valid
        order = free.long().cumsum(0) - 1
        taken = free & (order == torch.arange(self.count, device=valid.device)[:, None])
        slots = torch.where(taken, index, 0).sum(1)
        newest = torch.where(valid, ids, -1).max()
        cache_id = torch.where(valid.any(), newest + 1, torch.zeros_like(newest))
        return valid, slots, cache_id.reshape(1)


class _Write(nn.Module):
    """Writes the first `count` tokens' keys and values of every layer into the slots, as `update_cache` does."""

    def __init__(self, count: int):
        super().__init__()
        self.count = count

    def forward(
        self,
        keys: Tensor,
        values: Tensor,
        valid: Tensor,
        ids: Tensor,
        predicted: Tensor,
        slots: Tensor,
        cache_id: Tensor,
        new_keys: Tensor,
        new_values: Tensor,
        flag: Tensor,
    ) -> tuple[Tensor, ...]:
        keys = keys.index_copy(2, slots, new_keys[:, :, : self.count])
        values = values.index_copy(2, slots, new_values[:, :, : self.count])
        valid = valid.index_fill(0, slots, True)
        ids = ids.index_copy(0, slots, cache_id.expand(self.count))
        predicted = predicted.index_copy(0, slots, flag.expand(self.count))
        return keys, values, valid, ids, predicted


class _TimeEmbedding(nn.Module):
    """`WanTimeTextImageEmbedding.forward` without its text projection, which `_Blocks` holds applied."""

    def __init__(self, embedding: WanTimeTextImageEmbedding):
        super().__init__()
        self.timesteps_proj = embedding.timesteps_proj
        self.time_embedder = embedding.time_embedder
        self.act_fn = embedding.act_fn
        self.time_proj = embedding.time_proj

    def forward(self, timestep: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        b, seq_len = timestep.shape
        timestep = self.timesteps_proj(timestep.reshape(-1))
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(b, seq_len, -1), timestep_proj.reshape(b, seq_len, -1)


def _pad_tokens(x: Tensor, tokens: int) -> Tensor:
    """`x` with zeros after its tokens, on its second axis, up to `tokens`."""
    padding = [0, 0] * (x.dim() - 2) + [0, tokens - x.shape[1]]
    return F.pad(x, padding)


def _euler_step(sample: Tensor, prediction: Tensor, step: Tensor) -> Tensor:
    """`FlowMatchScheduler.step`'s `sample + prediction * step`, rounded as PyTorch rounds it in the model's dtype.

    The policy's step is a float32 number on the CPU, which PyTorch multiplies in float32 before rounding the
    product to the sample's dtype; a step on the device would be rounded to that dtype first.
    """
    product = (prediction.float() * step).to(sample.dtype)
    return (sample.float() + product.float()).to(sample.dtype)


class _VideoIn(nn.Module):
    """The video latents, the condition flag, the frame index and the timesteps in; the tokens `blocks` takes out.

    It does what `WanTransformer3DModel.forward` does before its blocks, on the latents repeated for classifier-free
    guidance as `_repeat_input_for_cfg` repeats them. With the condition flag, the first frame is the first chunk's
    latent, as `_prepare_latent_input` sets it; the latents are returned as the step takes them.
    """

    def __init__(
        self, transformer: WanTransformer3DModel, latent_shape: tuple[int, int, int], batch: int, tokens: int
    ):
        super().__init__()
        self.patch_embedding_mlp = transformer.patch_embedding_mlp
        self.time = _TimeEmbedding(transformer.condition_embedder)
        self.patch_size = transformer.patch_size
        self.batch = batch
        self.tokens = tokens
        frames, height, width = latent_shape
        p = self.patch_size
        self.tokens_per_frame = (height // p[1]) * (width // p[2])
        self.register_buffer(
            "grid", get_mesh_id(frames // p[0], height // p[1], width // p[2], 0, 1, 0)[:3], persistent=False
        )
        self.register_buffer("frame_axis", torch.tensor([[1], [0], [0]]), persistent=False)

    def forward(
        self, sample: Tensor, first_latent: Tensor, condition: Tensor, frame_start: Tensor, timesteps: Tensor
    ) -> tuple[Tensor, ...]:
        conditioned = torch.cat([first_latent[:, :, 0:1].to(sample.dtype), sample[:, :, 1:]], dim=2)
        sample = torch.where(condition, conditioned, sample)
        p = self.patch_size
        hidden = rearrange(
            sample.repeat(self.batch, 1, 1, 1, 1),
            "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
            p1=p[0],
            p2=p[1],
            p3=p[2],
        )
        hidden = self.patch_embedding_mlp(hidden)
        steps = torch.repeat_interleave(timesteps[None].repeat(self.batch, 1), self.tokens_per_frame, dim=1)
        temb, time = self.time(steps, hidden.dtype)
        grid = (self.grid + self.frame_axis * frame_start).double()[None]
        time = time.unflatten(2, (6, -1))
        return (
            _pad_tokens(hidden, self.tokens),
            _pad_tokens(time, self.tokens),
            temb,
            _pad_tokens(grid.transpose(1, 2), self.tokens).transpose(1, 2),
            sample,
        )


class _ActionIn(nn.Module):
    """The actions, the condition flag, the frame index and the timesteps in; the tokens `blocks` takes out.

    Like [`_VideoIn`] for the action stream. The unused action channels are zeros and, with the condition flag, so
    is the first frame, as `_prepare_latent_input` sets them; the actions are returned as the step takes them.
    """

    def __init__(
        self,
        transformer: WanTransformer3DModel,
        keep: Tensor,
        action_shape: tuple[int, int],
        batch: int,
        tokens: int,
    ):
        super().__init__()
        self.action_embedder = transformer.action_embedder
        self.time = _TimeEmbedding(transformer.condition_embedder_action)
        self.batch = batch
        self.tokens = tokens
        frames, per_frame = action_shape
        self.per_frame = per_frame
        self.register_buffer("keep", keep, persistent=False)
        # One frame's grid holds the offsets of the actions inside a frame exactly. The frame index is added to them
        # as an integer, as get_mesh_id adds it, so the sum is rounded once.
        single = get_mesh_id(1, per_frame, 1, 1, 1, 0, action=True)
        self.register_buffer("frame", torch.arange(frames).repeat_interleave(per_frame), persistent=False)
        self.register_buffer("offset", single[0].repeat(frames), persistent=False)
        self.register_buffer("rest", single[1:3].repeat(1, frames).double(), persistent=False)

    def forward(
        self, sample: Tensor, condition: Tensor, frame_start: Tensor, timesteps: Tensor
    ) -> tuple[Tensor, ...]:
        zeroed = torch.cat([torch.zeros_like(sample[:, :, 0:1]), sample[:, :, 1:]], dim=2)
        sample = torch.where(condition, zeroed, sample) * self.keep
        hidden = rearrange(sample.repeat(self.batch, 1, 1, 1, 1), "b c f h w -> b (f h w) c")
        hidden = self.action_embedder(hidden)
        steps = torch.repeat_interleave(timesteps[None].repeat(self.batch, 1), self.per_frame, dim=1)
        temb, time = self.time(steps, hidden.dtype)
        frame = ((self.frame + frame_start).float() + self.offset).double()
        grid = torch.cat([frame[None], self.rest])[None]
        time = time.unflatten(2, (6, -1))
        return (
            _pad_tokens(hidden, self.tokens),
            _pad_tokens(time, self.tokens),
            temb,
            _pad_tokens(grid.transpose(1, 2), self.tokens).transpose(1, 2),
            sample,
        )


class _Blocks(nn.Module):
    """Every transformer block over the tokens of one pass, which attend to the KV cache and to each other.

    The self-attention is each block's own `attn1` over the valid slots of the cache and the pass's valid tokens:
    the keys the policy's cache attends to once it has written the pass's tokens. The keys and values of the
    pass's tokens in every layer are returned, for a pass that writes them. The text the cross-attention reads is
    fixed with the task, so the program holds it already projected.
    """

    def __init__(self, transformer: WanTransformer3DModel, text: Tensor):
        super().__init__()
        self.blocks = transformer.blocks
        self.rope = transformer.rope
        self.register_buffer("text", text)

    def forward(
        self,
        hidden: Tensor,
        time: Tensor,
        grid: Tensor,
        keys: Tensor,
        values: Tensor,
        valid: Tensor,
        tokens: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        rotary = self.rope(grid)[:, :, None]
        mask = torch.cat([valid, tokens])[None, None, None]
        new_keys: list[Tensor] = []
        new_values: list[Tensor] = []
        for block, past_keys, past_values in zip(self.blocks, keys, values, strict=True):

            def attention(
                x: Tensor,
                attn: WanAttention = block.attn1,
                past_k: Tensor = past_keys,
                past_v: Tensor = past_values,
            ) -> Tensor:
                query, key, value = attn.project(x, x, x, rotary)
                new_keys.append(key)
                new_values.append(value)
                out = F.scaled_dot_product_attention(
                    query.transpose(1, 2),
                    torch.cat([past_k, key], dim=1).transpose(1, 2),
                    torch.cat([past_v, value], dim=1).transpose(1, 2),
                    attn_mask=mask,
                )
                return attn.output(out.transpose(1, 2), query)

            hidden = block.forward_with(hidden, self.text, time, attention)
        return hidden, torch.stack(new_keys), torch.stack(new_values)


class _Head(nn.Module):
    """The transformer's output norm, modulated by the time embedding, on the first `count` tokens."""

    def __init__(self, transformer: WanTransformer3DModel, count: int):
        super().__init__()
        self.norm_out = transformer.norm_out
        self.scale_shift_table = transformer.scale_shift_table
        self.count = count

    def forward(self, hidden: Tensor, temb: Tensor) -> Tensor:
        hidden = hidden[:, : self.count]
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return (self.norm_out(hidden.float()) * (1.0 + scale) + shift).type_as(hidden)


def _guide(prediction: Tensor, guidance_scale: float) -> Tensor:
    """The prediction with classifier-free guidance, the conditional one first, as `_infer` combines them."""
    if guidance_scale > 1:
        return prediction[1:] + guidance_scale * (prediction[:1] - prediction[1:])
    return prediction[:1]


class _VideoOut(nn.Module):
    """The blocks' output, the latents and the step in; the next latents out, the first frame conditioned again."""

    def __init__(
        self,
        transformer: WanTransformer3DModel,
        latent_shape: tuple[int, int, int],
        batch: int,
        guidance: float,
    ):
        super().__init__()
        frames, height, width = latent_shape
        p = transformer.patch_size
        self.head = _Head(transformer, (frames // p[0]) * (height // p[1]) * (width // p[2]))
        self.proj_out = transformer.proj_out
        self.patch_size = p
        self.latent_shape = latent_shape
        self.batch = batch
        self.guidance = guidance

    def forward(
        self,
        hidden: Tensor,
        temb: Tensor,
        sample: Tensor,
        step: Tensor,
        first_latent: Tensor,
        condition: Tensor,
    ) -> Tensor:
        hidden = self.proj_out(self.head(hidden, temb))
        hidden = rearrange(hidden, "b l (n c) -> b (l n) c", n=math.prod(self.patch_size))
        prediction = data_seq_to_patch(self.patch_size, hidden, *self.latent_shape, batch_size=self.batch)
        sample = _euler_step(sample, _guide(prediction, self.guidance), step)
        conditioned = torch.cat([first_latent[:, :, 0:1].to(sample.dtype), sample[:, :, 1:]], dim=2)
        return torch.where(condition, conditioned, sample)


class _ActionOut(nn.Module):
    """The blocks' output, the actions and the step in; the next actions out, the first frame zeros again."""

    def __init__(self, transformer: WanTransformer3DModel, frames: int, count: int, guidance: float):
        super().__init__()
        self.head = _Head(transformer, count)
        self.action_proj_out = transformer.action_proj_out
        self.frames = frames
        self.guidance = guidance

    def forward(
        self, hidden: Tensor, temb: Tensor, sample: Tensor, step: Tensor, condition: Tensor
    ) -> Tensor:
        hidden = self.action_proj_out(self.head(hidden, temb))
        prediction = rearrange(hidden, "b (f n) c -> b c f n 1", f=self.frames)
        sample = _euler_step(sample, _guide(prediction, self.guidance), step)
        zeroed = torch.cat([torch.zeros_like(sample[:, :, 0:1]), sample[:, :, 1:]], dim=2)
        return torch.where(condition, zeroed, sample)


class _Actions(nn.Module):
    """The denoised actions in; the actions the robot plays, in robot units, and the executed actions out.

    The first chunk drops its first frame's actions, which its condition frame stands for, as `_predict_chunk` does.
    """

    def __init__(self, postprocessor: PolicyProcessorPipeline, keep: Tensor, used: list[int], first: bool):
        super().__init__()
        self.postprocessor = postprocessor
        self.register_buffer("keep", keep, persistent=False)
        self.register_buffer("used", torch.tensor(used), persistent=False)
        self.first = first

    def forward(self, sample: Tensor) -> tuple[Tensor, Tensor]:
        executed = sample * self.keep
        actions = executed.index_select(1, self.used)
        if self.first:
            actions = actions[:, :, 1:]
        actions = actions.squeeze(-1).flatten(2).transpose(1, 2).contiguous().to(torch.float32)
        return self.postprocessor(actions)[0].float(), executed


class LingBotVAExportRecipe(ActionChunkRecipe):
    """LingBot-VA as small programs that keep its caches between chunks: see this module's documentation.

    The test case plays the first chunks of an episode from a reset, tick by tick, with the policy's own
    `select_action` as `lerobot-rollout` calls it, and each chunk's noise. Before the parts are traced, they are run
    the way the bundle runs them, on the test case and on other random frames and noise, and must play the same
    actions within the tolerance.
    """

    # LingBot-VA runs 20 video and 50 action denoising steps in bfloat16.
    bfloat16_tolerance = BFLOAT16_STEPS_TOLERANCE
    policy: LingBotVAPolicy

    def __init__(
        self,
        policy: LingBotVAPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        cfg: ExportConfig,
    ) -> None:
        """Check that the policy can be exported this way, and make the test case."""
        config = policy.config
        if config.camera_layout != "width_concat":
            raise ValueError(
                f"LingBot-VA's {config.camera_layout!r} camera layout encodes its cameras with two streaming VAEs, "
                "which this export does not do; it exports the 'width_concat' layout."
            )
        if config.frame_chunk_size < 2:
            raise ValueError("LingBot-VA plays no action of its first chunk when frame_chunk_size is 1.")
        if config.action_per_frame % VAE_FRAMES:
            raise ValueError(
                f"LingBot-VA's export needs action_per_frame to be a multiple of {VAE_FRAMES}, the camera frames "
                f"its VAE encodes into one latent frame, not {config.action_per_frame}."
            )
        self._frames = config.frame_chunk_size
        self._per_frame = config.action_per_frame
        self._stride = config.action_per_frame // VAE_FRAMES
        self._history_size = config.frame_chunk_size * config.action_per_frame
        super().__init__(policy, preprocessor, postprocessor, cfg)

    def _execution_steps(self) -> int:
        return self.policy.config.chunk_size

    def _task(self, frame: dict[str, np.ndarray]) -> str | None:
        # The policy reads the task with its own text encoder, which the preprocessor does not show.
        return export_task(self.cfg, self._metadata, self.policy.config.type)

    def _make_case(self, frame: dict[str, np.ndarray]) -> TestCase:
        renames = preprocessor_renames(self.preprocessor)
        self._cameras = [
            next(name for name in self._features if renames.get(name, name) == key)
            for key in self.policy.config.obs_cam_keys
        ]
        policy = self.policy
        latent_h, latent_w = policy._latent_hw
        config = policy.config
        self._video_shape = (1, config.in_channels, self._frames, latent_h, latent_w)
        self._action_shape = (1, config.action_dim, self._frames, self._per_frame, 1)
        ticks = [frame, *self._random_ticks()[1:]]
        noise = self._random_noises()
        actions = self._rollout(ticks, noise)
        if not np.array_equal(actions, self._rollout(ticks, noise)):
            raise ValueError(
                "This LingBot-VA policy plays other actions from the same frames and noise each time, so a "
                "compiled copy cannot be checked against it."
            )
        stacked = {name: np.stack([tick[name] for tick in ticks]) for name in self._features}
        return TestCase(stacked, noise, actions, self._action_scale(actions.shape[-1]))

    def _random_ticks(self) -> list[dict[str, np.ndarray]]:
        """Random frames for every control tick of the test case's chunks."""
        count = (self._frames - 1) * self._per_frame + (CHUNKS - 1) * self._history_size
        return [random_frame(self._features, self._generator) for _ in range(count)]

    def _random_noises(self) -> np.ndarray:
        """The flat noise of each of the test case's chunks: the video latents', then the actions'."""
        size = math.prod(self._video_shape) + math.prod(self._action_shape)
        return torch.randn(CHUNKS, size, generator=self._generator).numpy()

    def _split_noise(self, noise: np.ndarray) -> tuple[Tensor, Tensor]:
        flat = torch.from_numpy(noise).to(self._device)
        video, action = flat.split([math.prod(self._video_shape), math.prod(self._action_shape)])
        return video.view(self._video_shape), action.view(self._action_shape)

    def _rollout(self, ticks: list[dict[str, np.ndarray]], noise: np.ndarray) -> np.ndarray:
        """The action `lerobot-rollout` plays at each of `ticks` from a reset, each chunk drawing its `noise`."""
        self._reset()
        played = []
        chunk = -1
        with torch.no_grad():
            for frame in ticks:
                if self.policy.count_queued_actions() == 0:
                    chunk += 1
                video, action = self._split_noise(noise[chunk])
                batch = self.preprocessor(self._observation(frame, self.task))
                output = self.policy.select_action(batch, noise=action, video_noise=video)
                actions = self.postprocessor(transition_to_policy_action(policy_output_to_transition(output)))
                played.append(actions[0].float().cpu().numpy())
        if chunk + 1 != len(noise):
            raise ValueError(
                f"The policy ran {chunk + 1} chunks over the test case's ticks, not {len(noise)}."
            )
        return np.stack(played)

    def parts(self) -> Iterator[Part]:
        """The parts, after checking that they play the policy's actions as the bundle runs them."""
        self.policy._ensure_frozen_modules()
        modules = self._modules()
        examples: dict[str, tuple[Tensor, ...]] = {}
        ticks = self._random_ticks()
        noise = self._random_noises()
        stacked = {name: np.stack([tick[name] for tick in ticks]) for name in self._features}
        assert self._case.noise is not None
        for frames, case_noise, expected in (
            (self._case.frame, self._case.noise, self._case.actions),
            (stacked, noise, self._rollout(ticks, noise)),
        ):
            actions = self._play(modules, frames, case_noise, examples)
            difference = float((np.abs(actions - expected) / self._case.action_scale).max())
            logger.info(
                "Parts against the policy over %d ticks: largest difference %.2e (tolerance %g)",
                len(actions),
                difference,
                self.tolerance,
            )
            if not difference <= self.tolerance:
                raise ValueError(
                    "The exported parts of the LingBot-VA policy play other actions than the policy itself "
                    f"(largest difference {difference:.3g} of the actions' spread, tolerance {self.tolerance})."
                )
        runs = {run.program: run for chunk in self._chunk_runs() for run in chunk}
        for name, module in modules.items():
            yield Part(name, module, examples[name], runs[name].inputs, runs[name].outputs)

    def plan(self) -> RunPlan:
        """The runs of the first two chunks and of every later one, their constants, carry and camera history."""
        first, second, later = self._chunk_runs()
        return RunPlan(
            runs=later,
            first_runs=[first, second],
            carry=dict.fromkeys(CARRY),
            constants=self._constants(),
            history=dict.fromkeys(self._cameras, self._history_size),
        )

    def _modules(self) -> dict[str, nn.Module]:
        policy = self.policy
        config = policy.config
        transformer = policy.transformer
        dtype = next(transformer.parameters()).dtype
        latent_h, latent_w = policy._latent_hw
        latent_shape = (self._frames, latent_h, latent_w)
        batch = 2 if policy._use_cfg else 1
        video_tokens, action_tokens, tokens = self._token_counts()
        keep = torch.zeros(config.action_dim, dtype=dtype, device=self._device)
        keep[config.used_action_channel_ids] = 1
        keep = keep.view(1, -1, 1, 1, 1)
        heads, head_dim = config.num_attention_heads, config.attention_head_dim
        slots = (config.attn_window // 2) * (video_tokens + action_tokens)
        cache_shape = (len(transformer.blocks), batch, slots, heads, head_dim)
        cache_shapes = self._vae_cache_shapes()
        start = range(self._per_frame + self._stride - 1, self._history_size, self._stride)
        later = range(self._stride - 1, self._history_size, self._stride)
        encode = {
            "encode_first": ([self._history_size - 1], True, False),
            "encode_start": (list(start), False, True),
            "encode_next": (list(later), False, False),
        }
        modules: dict[str, nn.Module] = {
            name: _Encode(policy, self.preprocessor, self._cameras, frames, cache_shapes, first, first_latent)
            for name, (frames, first, first_latent) in encode.items()
        }
        modules.update(
            empty_cache=_EmptyCache(cache_shape, dtype, self._device),
            clear_predicted=_ClearPredicted(),
            advance=_Advance(self._frames),
            start=_Start(self._video_shape, self._action_shape, dtype),
            alloc_video=_Allocate(video_tokens),
            alloc_action=_Allocate(action_tokens),
            write_video=_Write(video_tokens),
            write_action=_Write(action_tokens),
            video_in=_VideoIn(transformer, latent_shape, batch, tokens),
            action_in=_ActionIn(transformer, keep, (self._frames, self._per_frame), batch, tokens),
            blocks=_Blocks(transformer, self._text()),
            video_out=_VideoOut(transformer, latent_shape, batch, config.guidance_scale),
            action_out=_ActionOut(transformer, self._frames, action_tokens, config.action_guidance_scale),
            actions_first=_Actions(self.postprocessor, keep, config.used_action_channel_ids, first=True),
            actions=_Actions(self.postprocessor, keep, config.used_action_channel_ids, first=False),
        )
        # The policy's modules are there already; this moves the tables the parts made.
        return {name: module.to(self._device) for name, module in modules.items()}

    def _token_counts(self) -> tuple[int, int, int]:
        """The tokens of a chunk's video latents and of its actions, and the tokens of every pass of `blocks`."""
        config = self.policy.config
        latent_h, latent_w = self.policy._latent_hw
        p = config.patch_size
        video = (self._frames // p[0]) * (latent_h // p[1]) * (latent_w // p[2])
        action = self._frames * self._per_frame
        return video, action, max(video, action)

    def _vae_cache_shapes(self) -> list[torch.Size]:
        """The shapes of the streaming VAE's cache after a call that is not the first, which later calls keep."""
        frame = random_frame(self._features, self._generator)
        batch = self.preprocessor(self._observation(frame, self.task))
        observed = self.policy._extract_raw_obs(batch)
        vae = self.policy._streaming_vae
        vae.clear_cache()
        with torch.no_grad():
            self.policy._encode_frames([observed])
            self.policy._encode_frames([observed] * VAE_FRAMES)
        shapes = [entry.shape for entry in vae.feat_cache if entry is not None]
        vae.clear_cache()
        self.preprocessor.reset()
        return shapes

    def _text(self) -> Tensor:
        """The task's text embeddings as the blocks' cross-attention reads them.

        With classifier-free guidance, the negative prompt's follow them, as `_repeat_input_for_cfg` stacks them.
        """
        policy = self.policy
        dtype = next(policy.transformer.parameters()).dtype
        prompt, negative = policy._encode_prompt(self.task)
        text = prompt.to(dtype) if negative is None else torch.cat([prompt.to(dtype), negative.to(dtype)])
        with torch.no_grad():
            return policy.transformer.condition_embedder.text_embedder(text).detach()

    def _schedules(self) -> dict[str, tuple[list[float], list[float]]]:
        """The timesteps of each stream's passes and the step of each pass but the last, as `_infer` runs them."""
        config = self.policy.config
        streams = {
            "video": (config.snr_shift, config.num_inference_steps, config.video_exec_step),
            "action": (config.action_snr_shift, config.action_num_inference_steps, -1),
        }
        schedules = {}
        for name, (shift, steps, exec_step) in streams.items():
            scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
            scheduler.set_timesteps(steps)
            timesteps = F.pad(scheduler.timesteps, (0, 1), mode="constant", value=0)
            if exec_step != -1:
                timesteps = timesteps[:exec_step]
            one, zero = torch.ones(()), torch.zeros(())
            schedules[name] = (
                [float(t) for t in timesteps],
                [float(scheduler.step(one, t, zero)) for t in timesteps[:-1]],
            )
        return schedules

    def _constants(self) -> dict[str, np.ndarray]:
        video_tokens, action_tokens, tokens = self._token_counts()
        constants = {
            FIRST: np.array([True]),
            LATER: np.array([False]),
            "predicted": np.array([True]),
            "observed": np.array([False]),
            "feedback_time": np.zeros(self._frames, np.float32),
            TOKENS[0]: np.arange(tokens) < video_tokens,
            TOKENS[1]: np.arange(tokens) < action_tokens,
        }
        for stream, (timesteps, steps) in self._schedules().items():
            for i, t in enumerate(timesteps):
                times: np.ndarray = np.full(self._frames, t, np.float32)
                constants[f"{stream}_time_{i}"] = times
                constants[f"{stream}_time_first_{i}"] = np.concatenate([[0.0], times[1:]]).astype(np.float32)
            for i, step in enumerate(steps):
                constants[f"{stream}_step_{i}"] = np.array([step], np.float32)
        return constants

    def _chunk_runs(self) -> tuple[list[ProgramRun], list[ProgramRun], list[ProgramRun]]:
        """The runs of the first chunk after a reset, of the second, and of every later one."""
        schedules = self._schedules()
        first = [
            ProgramRun("encode_first", "encode_first", list(self._cameras), [FIRST_LATENT, VAE_CACHE]),
            ProgramRun("empty_cache", "empty_cache", [], [*CACHE, FRAME_START]),
            *self._denoise(schedules, first=True),
            ProgramRun("actions", "actions_first", ["action_cond"], [ACTION, EXECUTED]),
        ]
        chunks = []
        for encode, encode_inputs in (
            ("encode_start", [*self._cameras, VAE_CACHE, FIRST_LATENT]),
            ("encode_next", [*self._cameras, VAE_CACHE]),
        ):
            chunks.append(
                [
                    ProgramRun("clear_predicted", "clear_predicted", [VALID, PREDICTED], [VALID]),
                    ProgramRun(encode, encode, encode_inputs, ["observed_latents", VAE_CACHE]),
                    *self._pass(
                        "feedback_video",
                        "video",
                        ["observed_latents", FIRST_LATENT, LATER, FRAME_START],
                        "feedback_time",
                    ),
                    ProgramRun(
                        "feedback_video_write", "write_video", self._write_inputs("observed"), list(CACHE)
                    ),
                    *self._pass("feedback_action", "action", [EXECUTED, LATER, FRAME_START], "feedback_time"),
                    ProgramRun(
                        "feedback_action_write", "write_action", self._write_inputs("observed"), list(CACHE)
                    ),
                    ProgramRun("advance", "advance", [FRAME_START], [FRAME_START]),
                    *self._denoise(schedules, first=False),
                    ProgramRun("actions", "actions", ["action_cond"], [ACTION, EXECUTED]),
                ]
            )
        return first, chunks[0], chunks[1]

    def _pass(self, name: str, stream: str, inputs: list[str], time: str) -> list[ProgramRun]:
        """The runs of one transformer pass of `stream`: free slots for its tokens, embed them, run the blocks."""
        tokens = TOKENS[0] if stream == "video" else TOKENS[1]
        return [
            ProgramRun(f"{name}_alloc", f"alloc_{stream}", [VALID, IDS], [VALID, "slots", "cache_id"]),
            ProgramRun(
                f"{name}_in",
                f"{stream}_in",
                [*inputs, time],
                ["hidden", "time", "temb", "grid", f"{stream}_cond"],
            ),
            ProgramRun(
                f"{name}_blocks",
                "blocks",
                ["hidden", "time", "grid", KEYS, VALUES, VALID, tokens],
                ["hidden_out", "pass_keys", "pass_values"],
            ),
        ]

    @staticmethod
    def _write_inputs(flag: str) -> list[str]:
        return [*CACHE, "slots", "cache_id", "pass_keys", "pass_values", flag]

    def _denoise(
        self, schedules: dict[str, tuple[list[float], list[float]]], first: bool
    ) -> list[ProgramRun]:
        """The runs that start from the noise and denoise the video latents, then the actions."""
        condition = FIRST if first else LATER
        suffix = "_first" if first else ""
        runs = [ProgramRun("start", "start", [NOISE], ["video_sample", "action_sample"])]
        for stream, extra in (
            ("video", [FIRST_LATENT, condition, FRAME_START]),
            ("action", [condition, FRAME_START]),
        ):
            timesteps, steps = schedules[stream]
            sample = f"{stream}_sample"
            for i in range(len(timesteps)):
                pass_runs = self._pass(
                    f"{stream}_{i}", stream, [sample, *extra], f"{stream}_time{suffix}_{i}"
                )
                # The slots are taken once per stream; every pass of it attends as if it wrote there.
                runs += pass_runs if i == 0 else pass_runs[1:]
                if i < len(steps):
                    out_inputs = ["hidden_out", "temb", f"{stream}_cond", f"{stream}_step_{i}"]
                    out_inputs += [FIRST_LATENT, condition] if stream == "video" else [condition]
                    runs.append(ProgramRun(f"{stream}_{i}_out", f"{stream}_out", out_inputs, [sample]))
            runs.append(
                ProgramRun(f"{stream}_write", f"write_{stream}", self._write_inputs("predicted"), list(CACHE))
            )
        return runs

    def _play(
        self,
        modules: dict[str, nn.Module],
        frames: dict[str, np.ndarray],
        noise: np.ndarray,
        examples: dict[str, tuple[Tensor, ...]],
    ) -> np.ndarray:
        """The actions the parts play at each tick of `frames`, run as the bundle runs them, from a reset.

        The first inputs each part gets are kept in `examples`, to trace it with.
        """
        chunk_runs = self._chunk_runs()
        constants = {
            name: torch.from_numpy(value).to(self._device) for name, value in self._constants().items()
        }
        history: dict[str, deque[Tensor]] = {name: deque(maxlen=self._history_size) for name in self._cameras}
        carried: dict[str, Tensor] = {}
        queued: deque[np.ndarray] = deque()
        played = []
        chunk = 0
        ticks = len(next(iter(frames.values())))
        with torch.no_grad():
            for tick in range(ticks):
                for name, past in history.items():
                    value = torch.from_numpy(frames[name][tick]).to(self._device)
                    past.extend([value] * (past.maxlen or 0) if not past else [value])
                if not queued:
                    values: dict[str, Tensor] = {**constants, **carried}
                    values[NOISE] = torch.from_numpy(noise[chunk]).to(self._device)
                    values.update({name: torch.stack(list(past)) for name, past in history.items()})
                    for run in chunk_runs[min(chunk, len(chunk_runs) - 1)]:
                        inputs = tuple(values[name] for name in run.inputs)
                        examples.setdefault(run.program, inputs)
                        outputs = modules[run.program](*inputs)
                        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
                        values.update(zip(run.outputs, outputs, strict=True))
                    queued.extend(values[ACTION].cpu().numpy())
                    carried = {name: values[name] for name in CARRY}
                    chunk += 1
                played.append(queued.popleft())
        return np.stack(played)
