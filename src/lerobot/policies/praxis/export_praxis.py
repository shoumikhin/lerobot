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

"""Export Praxis-1 as small programs that run its sampler for one chunk, over its window of two chunks.

Praxis-1 denoises the video and the actions of the next chunk together with the chunk the robot just executed:
with `max_context_chunks=1`, as the released checkpoints run, its window holds that executed chunk and the next
one. It reads the executed chunk as the video latents of its last 21 camera frames, after an anchor latent of
the first of them, and each arm's pose at the start of each chunk. The bundle keeps those frames and states, and
the current ones (`RunPlan.history`). The noise of both chunks is one flat noise input.

One chunk runs as these programs, which pass their values to each other by name:

::

    encode                        the kept frames and states -> the anchor latent, the clean latents, the proprio
    start                         the noise -> the starting video latents and actions of both chunks
    velocity, decoder, step       each step but the last: the velocities, one Euler step
    velocity, actions             the last step, and the actions in robot units

`velocity` runs the encoder tower and the action expert for the task's text and for the blank caption of the
unguided pass, so the bundle holds the tower's weights once: the two texts have different numbers of tokens.
`decoder` runs the decoder tower for one of them. The last step's video moves no action, so it runs no decoder.

The first chunk after a reset has no executed chunk: the policy denoises a window of one chunk. That chunk is
the first chunk of the two-chunk window, which reads neither the clean latents nor the second chunk, so every
chunk runs the same programs. After a reset the bundle fills the kept frames with the first one, so `encode`
reads the current frame as the anchor frame, as the policy does. Only `actions_first` differs: it plays the
first chunk's 21 actions, where `actions` plays the second chunk's 24 (`first_runs`).
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
from collections import deque
from collections.abc import Iterator
from types import ModuleType
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from torch import Tensor, nn

from lerobot.bundle.manifest import NOISE, ProgramRun
from lerobot.export.backends.executorch_tensorrt import ExecuTorchTensorRTConfig, torch_tensorrt_has_setting
from lerobot.export.recipe import (
    ActionChunkRecipe,
    Part,
    ProgramCheck,
    RunPlan,
    TestCase,
    export_task,
    preprocessor_renames,
    random_frame,
)
from lerobot.processor import (
    PolicyProcessorPipeline,
    policy_output_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from .modeling_praxis import _praxis_numerics
from .praxis_core import bimanual, layers as praxis_layers, single_arm
from .praxis_core.action_expert import run_expert_on_taps_k
from .praxis_core.robot_io import FRAMES_PER_TOKEN, contiguous_chunk_geometry, text_encoder_prompt
from .praxis_core.sampler import ACTION_SALT, VIDEO_SALT, guidance_combine, seeded_normal, window_time_grid
from .processor_praxis import tile_frames, tile_frames_bimanual

if TYPE_CHECKING:
    from lerobot.export.configs import ExportConfig

    from .configuration_praxis import PraxisConfig
    from .modeling_praxis import PraxisPolicy
    from .praxis_core.model import PraxisModel

logger = logging.getLogger(__name__)

# The chunks of the window with one executed chunk as context.
WINDOW_CHUNKS = 2
# The first chunk after a reset, the second, and the third, the first whose window drops frames.
CHUNKS = 3

ANCHOR, CLEAN, PROPRIO = "anchor", "clean", "proprio"
VIDEO, ACTIONS = "video_sample", "action_sample"

# The parts that hold the towers' mixture-of-experts layers.
MOE_PARTS = ("velocity", "decoder")
# The Torch-TensorRT setting that runs the grouped matrix multiplies of those layers inside TensorRT.
GROUPED_MM_KERNEL = "grouped_mm_kernel"


# ---------------------------------------------------------------------------
# Poses: the NumPy conversions of the processors and the session, as tensor operations
# ---------------------------------------------------------------------------


def _quat_mul_xyzw(p: Tensor, q: Tensor) -> Tensor:
    """`Rotation.quat_mul_xyzw`."""
    p0, p1, p2, p3 = p.unbind(-1)
    q0, q1, q2, q3 = q.unbind(-1)
    c0 = p1 * q2 - p2 * q1
    c1 = p2 * q0 - p0 * q2
    c2 = p0 * q1 - p1 * q0
    w = p3 * q3 - p0 * q0 - p1 * q1
    return torch.stack(
        [p3 * q0 + q3 * p0 + c0, p3 * q1 + q3 * p1 + c1, p3 * q2 + q3 * p2 + c2, w - p2 * q2], dim=-1
    )


def _euler_xyz_to_quat_xyzw(rpy: Tensor) -> Tensor:
    """`Rotation.euler_xyz_to_quat_xyzw`."""
    zeros = torch.zeros_like(rpy[..., 0])
    quat = None
    for axis in range(3):
        half = rpy[..., axis] / 2
        parts = [zeros, zeros, zeros, half.cos()]
        parts[axis] = half.sin()
        element = torch.stack(parts, dim=-1)
        quat = element if quat is None else _quat_mul_xyzw(element, quat)
    return cast(Tensor, quat)


def _unit_quat_xyzw(q: Tensor) -> Tensor:
    """`Rotation._unit_quat_xyzw`."""
    x, y, z, w = q.unbind(-1)
    return q / (x * x + y * y + z * z + w * w).sqrt()[..., None]


def _quat_xyzw_matrix(q: Tensor) -> Tensor:
    """`Rotation._quat_xyzw_matrix`: the rows of the matrix of a unit quaternion."""
    x, y, z, w = q.unbind(-1)
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    rows = [
        [x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw)],
        [2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw)],
        [2 * (xz - yw), 2 * (yz + xw), -x2 - y2 + z2 + w2],
    ]
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def _matrix_to_quat_xyzw(m: Tensor) -> Tensor:
    """`Rotation.matrix_to_quat_xyzw`: the quaternion of each branch, picked by the largest component."""
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    branches = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        q = [torch.zeros_like(trace)] * 4
        q[i] = 1 - trace + 2 * m[..., i, i]
        q[j] = m[..., j, i] + m[..., i, j]
        q[k] = m[..., k, i] + m[..., i, k]
        q[3] = m[..., k, j] - m[..., j, k]
        branches.append(torch.stack(q, dim=-1))
    branches.append(
        torch.stack(
            [
                m[..., 2, 1] - m[..., 1, 2],
                m[..., 0, 2] - m[..., 2, 0],
                m[..., 1, 0] - m[..., 0, 1],
                1 + trace,
            ],
            dim=-1,
        )
    )
    choice = torch.stack([m[..., 0, 0], m[..., 1, 1], m[..., 2, 2], trace], dim=-1).argmax(dim=-1)
    picked = torch.stack(branches, dim=-2).gather(-2, choice[..., None, None].expand(*choice.shape, 1, 4))
    return _unit_quat_xyzw(picked.squeeze(-2))


def _session_poses(states: Tensor, arms: int) -> Tensor:
    """The pose8 of each arm `(..., arms, 8)` the session reads from DROID rows of `7 * arms` values, float32.

    `droid7_to_pose8` of the pose processor step, in float64, then on the single-arm rig `pose8_to_model` of
    its session.
    """
    droid = states.double().unflatten(-1, (arms, 7))
    rotation = _quat_xyzw_matrix(_euler_xyz_to_quat_xyzw(droid[..., 3:6]))
    quat_xyzw = _matrix_to_quat_xyzw(rotation)
    pose8 = torch.cat(
        [droid[..., :3], quat_xyzw[..., [3, 0, 1, 2]], droid[..., 6:7].clamp(0, 1)], dim=-1
    ).float()
    if arms > 1:
        return pose8
    rotation = _quat_xyzw_matrix(_unit_quat_xyzw(pose8.double()[..., [4, 5, 6, 3]]))
    quat_wxyz = _matrix_to_quat_xyzw(rotation)[..., [3, 0, 1, 2]]
    return torch.cat(
        [pose8[..., :3].double(), quat_wxyz, pose8[..., 7:8].double().clamp(0, 1)], dim=-1
    ).float()


def _proprio(pose8: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """`proprio_tensors` of the session: `pose8_to_10d` of each arm, then the mean and std normalization, in
    float32 as NumPy runs them."""
    quat = pose8[..., 3:7]
    quat = quat / (quat.square().sum(-1, keepdim=True).sqrt() + 1e-8)
    w, x, y, z = quat.unbind(-1)
    columns = [
        [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
        [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
    ]
    rot6d = torch.stack([value for column in columns for value in column], dim=-1)
    rows = torch.cat([pose8[..., :3], rot6d, pose8[..., 7:8]], dim=-1).flatten(-2)
    return (rows - mean) / std.clamp(min=1e-6)


def _rot6d_to_matrix(r6: Tensor) -> Tensor:
    """`rot6d_to_matrix`: Gram-Schmidt on the two columns, in float64."""
    a0, a1 = r6[..., :3].double(), r6[..., 3:6].double()
    c0 = a0 / (a0.square().sum(-1, keepdim=True).sqrt() + 1e-8)
    a1 = a1 - (c0 * a1).sum(-1, keepdim=True) * c0
    c1 = a1 / (a1.square().sum(-1, keepdim=True).sqrt() + 1e-8)
    return torch.stack([c0, c1, torch.linalg.cross(c0, c1)], dim=-1)


def _matrix_to_euler_xyz(r: Tensor) -> Tensor:
    """`Rotation.matrix_to_euler_xyz`."""
    pitch = (-r[..., 2, 0]).clamp(-1, 1).asin()
    regular = pitch.cos().abs() > 1e-6
    roll = torch.where(regular, torch.atan2(r[..., 2, 1], r[..., 2, 2]), torch.zeros_like(pitch))
    yaw = torch.where(
        regular, torch.atan2(r[..., 1, 0], r[..., 0, 0]), torch.atan2(-r[..., 0, 1], r[..., 1, 1])
    )
    return torch.stack([roll, pitch, yaw], dim=-1)


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------


class _Encode(nn.Module):
    """The kept camera frames and states in; the anchor latent, the clean latents and the proprio out.

    The executed chunk is the first `executed` kept frames, which the session encodes as the clean latents,
    after the first of them alone as the anchor. The autoencoder is causal and encodes the first frame alone,
    so the first clean latent is that anchor latent. Each frame becomes its tiles as the preprocessor makes
    them, side by side in the order of the cameras. The proprio is the normalized pose at the first kept frame
    and at the current one, the last.

    Args:
        policy (`PraxisPolicy`):
            The policy, with its frozen autoencoder loaded.
        executed (`int`):
            The camera frames of the executed chunk.
    """

    def __init__(self, policy: PraxisPolicy, executed: int):
        super().__init__()
        vae = policy.vae
        # The autoencoder is no submodule of the policy. Only the modules `encode_eager` runs, its encoder half,
        # are registered, so the program holds none of the decoder's weights.
        object.__setattr__(self, "vae", vae)
        self.vae_encoder = nn.ModuleList([vae.encoder, vae.conv1])
        self.tile_height, self.tile_width = (int(size) for size in policy.variant.tile_size)
        self.aspect_tol = float(policy.variant.aspect_tol)
        self.antialias = bool(policy.variant.antialias)
        self.arms = policy.config.n_arms
        self.executed = executed
        self.register_buffer("mean", torch.from_numpy(policy.stats.proprio_mean), persistent=False)
        self.register_buffer("std", torch.from_numpy(policy.stats.proprio_std), persistent=False)

    def forward(self, *inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        *cameras, states = inputs
        frames = [camera[: self.executed].permute(0, 3, 1, 2) for camera in cameras]
        if self.arms > 1:
            tiles = tile_frames_bimanual(frames, self.tile_height, self.tile_width)
        else:
            tiles = [
                tile_frames(view, self.tile_height, self.tile_width, self.aspect_tol, self.antialias)
                for view in frames
            ]
        clip = torch.cat(tiles, dim=-1).float().permute(1, 0, 2, 3).unsqueeze(0)
        clean = self.vae.encode_eager(clip)
        anchor = clean[:, :, :1].contiguous()
        poses = _session_poses(torch.stack([states[0], states[-1]]), self.arms)
        return anchor, clean, _proprio(poses, self.mean, self.std)[None]


class _Start(nn.Module):
    """The flat noise in; the starting video latents and actions of the window, float32, out."""

    def __init__(self, video_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__()
        self.video_shape = video_shape
        self.action_shape = action_shape

    def forward(self, noise: Tensor) -> tuple[Tensor, Tensor]:
        video, actions = noise.split([math.prod(self.video_shape), math.prod(self.action_shape)])
        return video.view(self.video_shape), actions.view(self.action_shape)


class _Velocity(nn.Module):
    """The window's latents and actions in; the encoder's output of each pass, and the action velocity of each
    pass whose actions the step combines, out.

    It runs `velocity` of `PraxisModel.sample_window_joint` without the decoder: the encoder tower and the action
    expert on its taps, for the task's text and, with guidance, for the blank caption. Without guidance of the
    actions, the sampler takes the guided pass's action velocity as it is, so the blank caption's pass runs no
    expert. The texts are fixed with the task, so the program holds them.
    """

    def __init__(self, model: PraxisModel, texts: list[Tensor], action_passes: int):
        super().__init__()
        self.encoder = model.backbone.encoder
        self.expert = model.action_expert
        self.action_passes = action_passes
        self.register_buffer("text", texts[0])
        self.register_buffer("text_uncond", texts[1] if len(texts) > 1 else None)

    def forward(
        self, anchor: Tensor, clean: Tensor, video: Tensor, actions: Tensor, proprio: Tensor, time: Tensor
    ) -> tuple[Tensor, ...]:
        dtype = self.encoder.registers.dtype
        chunks = proprio.shape[1]
        anchor, clean, video = anchor.to(dtype), clean.to(dtype), video.to(dtype)
        outputs = []
        texts = [self.text] if self.text_uncond is None else [self.text, self.text_uncond]
        for i, text in enumerate(texts):
            cond, taps = self.encoder.forward_chunk_causal(
                text, anchor, clean, video, time.to(dtype).expand(1, chunks)
            )
            outputs.append(cond)
            if i >= self.action_passes:
                continue
            velocity = run_expert_on_taps_k(
                self.expert,
                taps.taps,
                taps.embeddings,
                taps.rope_ids,
                global_spans=taps.global_spans,
                noised_span=taps.noised_span,
                clean_span=taps.clean_span,
                n_chunks=chunks,
                frames_per_chunk=taps.frames_per_chunk,
                tokens_per_frame=taps.tokens_per_frame,
                noised_actions=actions[None],
                action_t_per_chunk=time.expand(1, chunks),
                proprio_per_chunk=proprio,
            )
            outputs.append(velocity.float()[0])
        return tuple(outputs)


class _Decoder(nn.Module):
    """The window's latents, one pass's encoder output and the time in; that pass's video velocity out."""

    def __init__(self, model: PraxisModel):
        super().__init__()
        self.decoder = model.backbone.decoder

    def forward(self, video: Tensor, cond: Tensor, time: Tensor) -> Tensor:
        dtype = self.decoder.registers.dtype
        timesteps = time.to(dtype).expand(1, WINDOW_CHUNKS)
        return self.decoder.forward_chunk_causal(video.to(dtype), cond, timesteps).float()


def _guided(velocities: list[Tensor], scale: float) -> Tensor:
    """The guided velocity of the guided pass and, if any, the unguided one, as the sampler combines them."""
    return velocities[0] if len(velocities) == 1 else guidance_combine(velocities[0], velocities[1], scale)


class _Step(nn.Module):
    """The latents, the actions, their velocities of each pass and the step in; the latents and actions after
    one Euler step out."""

    def __init__(self, video_passes: int, video_scale: float, action_scale: float):
        super().__init__()
        self.video_passes = video_passes
        self.video_scale = video_scale
        self.action_scale = action_scale

    def forward(self, video: Tensor, actions: Tensor, *inputs: Tensor) -> tuple[Tensor, Tensor]:
        *velocities, step = inputs
        video_velocity = _guided(velocities[: self.video_passes], self.video_scale)
        action_velocity = _guided(velocities[self.video_passes :], self.action_scale)
        return video + step * video_velocity, actions + step * action_velocity


class _Actions(nn.Module):
    """The actions, their velocities of each pass, the last step and the kept states in; the actions the robot
    plays, in robot units, out.

    It takes the last Euler step, then does what the session and the command processor step do with the actions
    of the chunk: unpack the action tokens to rows, undo the normalization, compose each arm's rows with its
    current pose, and turn each pose into the DROID `[x, y, z, roll, pitch, yaw, gripper]` of that arm. The first
    chunk after a reset plays the first chunk of the window, whose first token stands for its anchor frame and
    gives one row.

    Args:
        policy (`PraxisPolicy`):
            The policy.
        first (`bool`):
            Whether this is the first chunk after a reset.
    """

    def __init__(self, policy: PraxisPolicy, first: bool):
        super().__init__()
        config = policy.config
        self.action_scale = policy.sampler_settings.cfg_scale
        self.chunk = 0 if first else WINDOW_CHUNKS - 1
        self.tokens = int(config.model_config["chunking"]["latents_per_chunk"])
        self.first = first
        self.arms = config.n_arms
        rig = _rig(config)
        self.width, self.grips = rig.ACTION_DIM, rig.GRIP_DIMS.tolist()
        normalized = torch.zeros(self.width, dtype=torch.bool)
        normalized[rig.q99_dims(config.model_config)] = True
        stats = policy.stats
        self.register_buffer("normalized", normalized, persistent=False)
        self.register_buffer("low", torch.from_numpy(stats.action_q01[: self.width]), persistent=False)
        self.register_buffer("high", torch.from_numpy(stats.action_q99[: self.width]), persistent=False)

    def forward(self, actions: Tensor, *inputs: Tensor) -> Tensor:
        *velocities, step, states = inputs
        actions = actions + step * _guided(list(velocities), self.action_scale)
        tokens = actions[self.chunk * self.tokens : (self.chunk + 1) * self.tokens]
        rows = tokens.view(self.tokens, FRAMES_PER_TOKEN, -1)[..., : self.width].flatten(0, 1)
        if self.first:
            rows = torch.cat([rows[:1], rows[FRAMES_PER_TOKEN:]])
        clipped = torch.where(self.normalized, rows.clamp(-1, 1), rows)
        relative = torch.where(
            self.normalized, (clipped + 1.0) / 2.0 * (self.high - self.low) + self.low, clipped
        )
        poses = _session_poses(states[-1], self.arms)
        arms = []
        for arm, (pose8, grip) in enumerate(zip(poses, self.grips, strict=True)):
            rel = relative[:, arm * single_arm.ACTION_DIM : (arm + 1) * single_arm.ACTION_DIM]
            anchor_rotation = _quat_xyzw_matrix(_unit_quat_xyzw(pose8.double()[[4, 5, 6, 3]]))
            xyz = (rel[:, :3].double() + pose8[:3].double()).float()
            rotation = (_rot6d_to_matrix(rel[:, 3:9]) @ anchor_rotation).float()
            rpy = _matrix_to_euler_xyz(rotation.double())
            arms += [xyz.double(), rpy, relative[:, grip, None].clamp(0, 1).double()]
        return torch.cat(arms, dim=-1).float()


def _rig(config: PraxisConfig) -> ModuleType:
    """The rig module of the checkpoint: `single_arm` (DROID) or `bimanual` (YAM)."""
    return bimanual if config.n_arms > 1 else single_arm


def _check_config(config: PraxisConfig) -> None:
    """Refuse a setting whose sessions the programs do not run."""
    if config.max_context_chunks != 1:
        raise ValueError(
            f"Praxis-1's export runs a window of the executed chunk and the next one, as the released "
            f"checkpoints run, not max_context_chunks={config.max_context_chunks}. Export it with "
            "--policy.max_context_chunks=1."
        )
    if config.n_action_steps != config.chunk_size:
        raise ValueError(
            f"With n_action_steps={config.n_action_steps} below chunk_size={config.chunk_size}, every chunk of "
            "Praxis-1 starts without context, which its export does not run. Export it with "
            f"--policy.n_action_steps={config.chunk_size}."
        )


def _check_torch_tensorrt(backend: ExecuTorchTensorRTConfig) -> None:
    """Refuse a Torch-TensorRT that cannot run the mixture-of-experts layers in TensorRT, or none at all.

    Such a Torch-TensorRT ignores `grouped_mm_kernel` without a word and leaves each such layer to PyTorch.
    """
    try:
        import torch_tensorrt
    except ImportError as error:
        raise backend.install_error(error) from error
    if not torch_tensorrt_has_setting(GROUPED_MM_KERNEL):
        raise ValueError(
            f"Torch-TensorRT {torch_tensorrt.__version__} has no {GROUPED_MM_KERNEL} setting, which Praxis-1 needs "
            "to run its mixture-of-experts layers inside TensorRT (pytorch/TensorRT#5021, in Torch-TensorRT "
            "2.15.1)."
        )


class PraxisExportRecipe(ActionChunkRecipe):
    """Praxis-1 as small programs that run its sampler for one chunk: see this module's documentation.

    The test case plays the first chunks of an episode from a reset, tick by tick, with the policy's own
    `select_action` as `lerobot-rollout` calls it, and the noise the policy draws for each chunk. Before the parts
    are traced, they are run the way the bundle runs them, on the test case and on other random frames and noise,
    and must play the same actions within the tolerance. In bfloat16 and float16, the `velocity` and `decoder`
    programs are also checked on their own (`ProgramCheck`).
    """

    # In bfloat16, the 27B mixture of experts amplifies rounding over the test case's 3 chunks: on one test case,
    # eager PyTorch fed inputs moved by one rounding step played actions up to 0.55 away, and a correct TensorRT
    # bundle 0.62 (0.82 on another). A bundle whose encoder tower skipped a block or swapped its keys and values
    # moved them by about 1.0, so `velocity` and `decoder` are also checked on their own, and the test case only
    # catches a bundle that runs its parts far wrong: zeroed velocities moved the actions by 3.9 and swapped cameras by 1.6.
    bfloat16_tolerance = 1.0
    policy: PraxisPolicy

    @classmethod
    def load(cls, cfg: ExportConfig) -> PraxisExportRecipe:
        """Refuse a setting this recipe cannot export before any weight loads, then load the policy."""
        _check_config(cast("PraxisConfig", cfg.policy))
        if isinstance(cfg.backend, ExecuTorchTensorRTConfig):
            _check_torch_tensorrt(cfg.backend)
        return cast(PraxisExportRecipe, super().load(cfg))

    def __init__(
        self,
        policy: PraxisPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        cfg: ExportConfig,
    ) -> None:
        """Check that the policy can be exported this way, and make the test case."""
        _check_config(policy.config)
        # The camera frames of the executed chunk: the first chunk of a window spans 21.
        self._executed = contiguous_chunk_geometry(policy.config.chunk_size, WINDOW_CHUNKS)[1][0]
        super().__init__(policy, preprocessor, postprocessor, cfg)
        # The programs hold the frozen modules, and a local path can name the exporter's user.
        if os.path.exists(self.config.vae_id):
            self.config.vae_id = ""
        if os.path.exists(self.config.text_encoder_id):
            self.config.text_encoder_id = ""

    def _task(self, frame: dict[str, np.ndarray]) -> str | None:
        # The policy encodes the task with its own text encoder, which the preprocessor's output does not show.
        return export_task(self.cfg, self._metadata, self.policy.config.type)

    def _make_case(self, frame: dict[str, np.ndarray]) -> TestCase:
        config = self.policy.config
        renames = preprocessor_renames(self.preprocessor)
        names = {renames.get(name, name): name for name in self._features}
        self._cameras = [names[key] for key in config.image_keys]
        self._state = names[OBS_STATE]
        height, width = _rig(config).canvas_size(config.model_config)
        with torch.no_grad():
            latent = self.policy.vae.encode(torch.zeros(1, 3, 1, height, width, device=self._device))
        channels, _, latent_height, latent_width = latent.shape[1:]
        self._latents = int(config.model_config["chunking"]["latents_per_chunk"])
        self._video_shape = (1, channels, WINDOW_CHUNKS * self._latents, latent_height, latent_width)
        self._action_shape = (
            WINDOW_CHUNKS * self._latents,
            int(config.model_config["action_expert"]["action_patch_dim"]),
        )
        ticks = [frame, *self._random_ticks()[1:]]
        noise = self._noise(config.seed)
        actions = self._rollout(ticks, config.seed)
        if not np.array_equal(actions, self._rollout(ticks, config.seed)):
            raise ValueError(
                "This Praxis-1 policy plays other actions from the same frames each time, so a compiled copy "
                "cannot be checked against it."
            )
        self._texts = self._encoded_texts()
        stacked = {name: np.stack([tick[name] for tick in ticks]) for name in self._features}
        return TestCase(stacked, noise, actions, self._scale())

    def _random_ticks(self) -> list[dict[str, np.ndarray]]:
        """Random frames for every control tick up to the third chunk: 21 actions, then 24, then one."""
        count = self._executed + self.policy.config.chunk_size * (CHUNKS - 2) + 1
        return [random_frame(self._features, self._generator) for _ in range(count)]

    def _noise(self, seed: int) -> np.ndarray:
        """The flat noise of each of the test case's chunks, as the policy draws it with `seed`.

        Chunk `k` draws its video latents and actions with `seeded_normal` and the seed `seed + k`, for the chunks of
        its window. The first chunk's window holds one chunk, so the second chunk of the bundle's window, which it
        does not read, takes other standard normal draws.
        """
        video_shape, action_shape = list(self._video_shape), list(self._action_shape)
        chunks = []
        for k in range(CHUNKS):
            window = 1 if k == 0 else WINDOW_CHUNKS
            video_shape[2], action_shape[0] = window * self._latents, window * self._latents
            video = seeded_normal(video_shape, seed + k, VIDEO_SALT)
            actions = seeded_normal(action_shape, seed + k, ACTION_SALT)
            if window < WINDOW_CHUNKS:
                video_shape[2], action_shape[0] = self._latents, self._latents
                video = torch.cat([video, torch.randn(video_shape, generator=self._generator)], dim=2)
                actions = torch.cat([actions, torch.randn(action_shape, generator=self._generator)])
            chunks.append(torch.cat([video.flatten(), actions.flatten()]))
        return torch.stack(chunks).numpy()

    def _rollout(self, ticks: list[dict[str, np.ndarray]], seed: int) -> np.ndarray:
        """The action `lerobot-rollout` plays at each of `ticks` from a reset, the policy seeding its noise with
        `seed`."""
        config = self.policy.config
        saved, config.seed = config.seed, seed
        try:
            self._reset()
            played = []
            with torch.no_grad():
                for frame in ticks:
                    batch = self.preprocessor(self._observation(frame, self.task))
                    output = self.policy.select_action(batch)
                    actions = self.postprocessor(
                        transition_to_policy_action(policy_output_to_transition(output))
                    )
                    played.append(actions[0].float().cpu().numpy())
        finally:
            config.seed = saved
        state = self.policy.sessions()[0].state
        if state is None or state.chunk_index != CHUNKS:
            raise ValueError(
                f"The policy ran another number of chunks than {CHUNKS} over the test case's ticks."
            )
        return np.stack(played)

    def _encoded_texts(self) -> list[Tensor]:
        """The task's text, as the policy's session encoded it, and with guidance the blank caption's."""
        policy = self.policy
        env = policy.sessions()[0]
        with _praxis_numerics():
            texts = [
                env.text if env.text is not None else policy._text.encode_text(text_encoder_prompt(env.task))
            ]
            if policy.sampler_settings.guided:
                texts.append(policy._text.encode_uncond())
        return [text.to(self._device, policy.model.dtype) for text in texts]

    def _scale(self) -> np.ndarray:
        """Robot units per normalized unit of each arm's DROID actions: half the q99 spread of the position and
        the gripper.

        The rotation is not normalized. One unit of its 6D form turns it by about one radian.
        """
        config, stats = self.policy.config, self.policy.stats
        rig = _rig(config)
        spread = np.ones(rig.ACTION_DIM, dtype=np.float32)
        dims = rig.q99_dims(config.model_config)
        spread[dims] = (stats.action_q99[dims] - stats.action_q01[dims]) / 2
        arms = spread.reshape(config.n_arms, single_arm.ACTION_DIM)
        return np.concatenate([np.concatenate([arm[:3], np.ones(3, np.float32), arm[9:]]) for arm in arms])

    def parts(self) -> Iterator[Part]:
        """The parts, after checking that they play the policy's actions as the bundle runs them."""
        ticks = self._random_ticks()
        # The test case's chunks draw with the seeds `seed` to `seed + CHUNKS - 1`.
        seed = self.policy.config.seed + CHUNKS
        noise = self._noise(seed)
        expected = self._rollout(ticks, seed)
        stacked = {name: np.stack([tick[name] for tick in ticks]) for name in self._features}
        assert self._case.noise is not None
        with self._plain_pytorch():
            modules = self._modules()
            examples: dict[str, tuple[Tensor, ...]] = {}
            for frames, case_noise, case_actions in (
                (self._case.frame, self._case.noise, self._case.actions),
                (stacked, noise, expected),
            ):
                actions = self._play(modules, frames, case_noise, examples)
                difference = float((np.abs(actions - case_actions) / self._case.action_scale).max())
                logger.info(
                    "Parts against the policy over %d ticks: largest difference %.2e (tolerance %g)",
                    len(actions),
                    difference,
                    self.tolerance,
                )
                if not difference <= self.tolerance:
                    raise ValueError(
                        "The exported parts of the Praxis-1 policy play other actions than the policy itself "
                        f"(largest difference {difference:.3g} of the actions' spread, tolerance {self.tolerance})."
                    )
            runs = {run.program: run for chunk in self._chunk_runs() for run in chunk}
            checked = self.policy.model.dtype in (torch.bfloat16, torch.float16)
            tensorrt = isinstance(self.cfg.backend, ExecuTorchTensorRTConfig)
            for name, module in modules.items():
                check = None
                # The other parts compute in float32 and float64, or under an autocast of their own like the
                # autoencoder of `encode`, so no rerun measures how a compiler may round them otherwise.
                if checked and name in ("velocity", "decoder"):
                    # No float32 rerun: beside the policy, `velocity` in float32 needs 38 GB more, more than
                    # an 80 GB GPU has, and `decoder` 16 GB more, more than Thor has free while it exports. Other
                    # kernels and matmul settings already move their outputs by 1 to 48 percent of their largest
                    # value, and correct TensorRT programs stayed within 1.8 times that.
                    check = ProgramCheck.measure(module, examples[name], with_float32=False)
                # Torch-TensorRT runs the grouped matrix multiplies of mixture-of-experts layers only with this
                # setting. Without it each such layer stays in PyTorch and splits the program around it.
                options = {GROUPED_MM_KERNEL: True} if tensorrt and name in MOE_PARTS else {}
                yield Part(
                    name, module, examples[name], runs[name].inputs, runs[name].outputs, check, options
                )

    @contextlib.contextmanager
    def _plain_pytorch(self) -> Iterator[None]:
        """The policy running only operations `torch.export` records: dense attention, no custom kernels.

        The flex attention's masks and the custom Triton kernels are not PyTorch operations. The dense attention
        evaluates the same masks on every pair of tokens. The masks of the backbone come from the eager runs of the
        parts before they are traced, so the programs read them as constants.
        """
        config = self.policy.config
        backend, config.attention_backend = config.attention_backend, "eager"
        self.policy._apply_attention_backend(self._device)
        kernels = praxis_layers.custom_kernels_enabled()
        praxis_layers.set_custom_kernels(False)
        try:
            yield
        finally:
            praxis_layers.set_custom_kernels(kernels)
            config.attention_backend = backend
            self.policy._apply_attention_backend(self._device)

    def plan(self) -> RunPlan:
        """The runs of the first chunk and of every later one, their constants and the kept frames and states."""
        first, later = self._chunk_runs()
        return RunPlan(
            runs=later,
            first_runs=[first],
            constants=self._constants(),
            history=dict.fromkeys([*self._cameras, self._state], self._executed + 1),
        )

    def _passes(self) -> tuple[int, int]:
        """The passes whose video velocities and whose action velocities the sampler combines: 2 with guidance."""
        settings = self.policy.sampler_settings
        return 1 + (settings.cfg_video_scale != 1), 1 + (settings.cfg_scale != 1)

    def _modules(self) -> dict[str, nn.Module]:
        policy, settings = self.policy, self.policy.sampler_settings
        video_passes, action_passes = self._passes()
        modules: dict[str, nn.Module] = {
            "encode": _Encode(policy, self._executed),
            "start": _Start(self._video_shape, self._action_shape),
            "velocity": _Velocity(policy.model, self._texts, action_passes),
        }
        if settings.num_steps > 1:
            modules["decoder"] = _Decoder(policy.model)
            modules["step"] = _Step(video_passes, settings.cfg_video_scale, settings.cfg_scale)
        modules["actions_first"] = _Actions(policy, first=True)
        modules["actions"] = _Actions(policy, first=False)
        # The policy's modules are there already; this moves the tables the parts made.
        return {name: module.to(self._device) for name, module in modules.items()}

    def _constants(self) -> dict[str, np.ndarray]:
        """The time of each step and its Euler step, on the sampler's float64 grid, as float32."""
        settings = self.policy.sampler_settings
        times = window_time_grid(settings.num_steps, settings.t_shift)
        constants = {}
        for i in range(settings.num_steps):
            constants[f"time_{i}"] = np.array([times[i]], dtype=np.float32)
            constants[f"step_{i}"] = np.array([times[i + 1] - times[i]], dtype=np.float32)
        return constants

    def _chunk_runs(self) -> tuple[list[ProgramRun], list[ProgramRun]]:
        """The runs of the first chunk after a reset and of every later one."""
        settings = self.policy.sampler_settings
        video_passes, action_passes = self._passes()
        conds = ["cond", "cond_uncond"]
        video_velocities = ["video_velocity", "video_velocity_uncond"][:video_passes]
        action_velocities = ["action_velocity", "action_velocity_uncond"][:action_passes]
        velocity_outputs = ["cond", "action_velocity", "cond_uncond", "action_velocity_uncond"]
        velocity_outputs = velocity_outputs[: len(self._texts) + action_passes]
        runs = [
            ProgramRun("encode", "encode", [*self._cameras, self._state], [ANCHOR, CLEAN, PROPRIO]),
            ProgramRun("start", "start", [NOISE], [VIDEO, ACTIONS]),
        ]
        last = settings.num_steps - 1
        for i in range(settings.num_steps):
            inputs = [ANCHOR, CLEAN, VIDEO, ACTIONS, PROPRIO, f"time_{i}"]
            runs.append(ProgramRun(f"velocity_{i}", "velocity", inputs, velocity_outputs))
            if i == last:
                break
            for cond, velocity in zip(conds, video_velocities, strict=False):
                name = f"decoder_{i}" if cond == "cond" else f"decoder_uncond_{i}"
                runs.append(ProgramRun(name, "decoder", [VIDEO, cond, f"time_{i}"], [velocity]))
            step_inputs = [VIDEO, ACTIONS, *video_velocities, *action_velocities, f"step_{i}"]
            runs.append(ProgramRun(f"step_{i}", "step", step_inputs, [VIDEO, ACTIONS]))
        finish = [ACTIONS, *action_velocities, f"step_{last}", self._state]
        return (
            [*runs, ProgramRun("actions", "actions_first", finish, [ACTION])],
            [*runs, ProgramRun("actions", "actions", finish, [ACTION])],
        )

    def _play(
        self,
        modules: dict[str, nn.Module],
        frames: dict[str, np.ndarray],
        noise: np.ndarray,
        examples: dict[str, tuple[Tensor, ...]],
    ) -> np.ndarray:
        """The actions the parts play at each tick of `frames`, run as the bundle runs them, from a reset.

        The last inputs each part gets are kept in `examples`, to trace it with.
        """
        first, later = self._chunk_runs()
        constants = {
            name: torch.from_numpy(value).to(self._device) for name, value in self._constants().items()
        }
        history: dict[str, deque[Tensor]] = {
            name: deque(maxlen=self._executed + 1) for name in [*self._cameras, self._state]
        }
        queued: deque[np.ndarray] = deque()
        played = []
        chunk = 0
        with torch.no_grad():
            for tick in range(len(frames[self._state])):
                for name, past in history.items():
                    value = torch.from_numpy(frames[name][tick]).to(self._device)
                    past.extend([value] * (past.maxlen or 0) if not past else [value])
                if not queued:
                    values: dict[str, Tensor] = {
                        **constants,
                        NOISE: torch.from_numpy(noise[chunk]).to(self._device),
                    }
                    values.update({name: torch.stack(list(past)) for name, past in history.items()})
                    for run in first if chunk == 0 else later:
                        inputs = tuple(values[name] for name in run.inputs)
                        examples[run.program] = inputs
                        outputs = modules[run.program](*inputs)
                        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
                        values.update(zip(run.outputs, outputs, strict=True))
                    queued.extend(values[ACTION].cpu().numpy())
                    chunk += 1
                played.append(queued.popleft())
        return np.stack(played)
