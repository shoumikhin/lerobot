# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot in 2026.

"""Export G0.5's System 1 flow head as a chain of programs: a prefix, a flow step run once per step, and the actions.

One action chunk runs as these programs, which pass their values to each other by name:

::

    prefix        the frame -> the keys and values of the backbone's full-attention layers over the prompt
    step x10      those keys and values, the noisy actions and a timestep -> the next noisy actions
    actions       the denoised chunk -> the actions to play, in robot units

The prompt holds the task, the cameras' image tokens and a state token. Its token ids do not depend on the frame,
so the programs hold them, the attention masks and the rope positions as constants, and the bundle runs the task
it was exported for. The action expert reads only the cache of the backbone's full-attention layers, so the state
of its linear-attention (Gated DeltaNet) layers stays inside the prefix. The Qwen3.5 vision tower reads its image
grid on the host, which `torch.export` cannot trace, so the prefix runs it on the grid of the export's frames.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import copy
from typing import TYPE_CHECKING, Any, cast

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
from lerobot.processor import PolicyProcessorPipeline, RenderRuntimeMessagesStep, RenderTrainingMessagesStep
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import _transformers_available

from ..common.qwen3_vision import Qwen3VisionGrid
from ..common.qwen_vision import replay_vision_block
from .modeling_g05 import G05NativeBackend, G05Policy
from .processor_g05 import G05TokenizerStep
from .tokenizer_g05 import (
    G05_INPUT_IDS,
    G05_LABELS,
    G05_SPLIT_INDEX,
    G05_TOKEN_TYPES,
    G05SequenceBatch,
    G05TokenType,
)

if TYPE_CHECKING or _transformers_available:
    from transformers import DynamicCache
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        apply_rotary_pos_emb_vision,
        eager_attention_forward,
    )

if TYPE_CHECKING:
    from lerobot.export.configs import ExportConfig


def _autocast(device: torch.device, dtype: torch.dtype | None) -> torch.autocast:
    """The autocast `G05Policy` runs inference under: bfloat16 on a GPU for a bfloat16 policy."""
    return torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=dtype == torch.bfloat16 and device.type == "cuda"
    )


class G05Prefix(nn.Module):
    """The frame in, the keys and values of the backbone's full-attention layers over the prompt out.

    It runs what `G05NativeBackend._prefill` runs, with the prompt's token ids, masks and positions held, and the
    vision tower on a fixed image grid.
    """

    def __init__(
        self,
        policy: G05Policy,
        preprocessor: PolicyProcessorPipeline,
        frame_names: list[str],
        visual: set[str],
        sequence: dict[str, Tensor],
        grids: dict[str, Qwen3VisionGrid],
    ):
        super().__init__()
        self.backend = policy.backend
        self.preprocessor = preprocessor
        self.frame_names = frame_names
        self.visual = visual
        self.cameras = list(policy.config.camera_order)
        self.grids = grids
        self.device = policy.config.device
        self.dtype = policy.config.dtype
        self.full_layers = [
            i for i, kind in enumerate(self.backend.model.vlm.config.layer_types) if kind == "full_attention"
        ]
        for name, value in sequence.items():
            self.register_buffer(name, value)

    def forward(self, *frame: Tensor) -> tuple[Tensor, ...]:
        batch = self.preprocessor(
            {
                name: to_policy_input(x, name in self.visual)
                for name, x in zip(self.frame_names, frame, strict=True)
            }
        )
        model = self.backend.model
        with _autocast(torch.device(self.device), self.dtype):
            images = torch.cat(
                [self.encode_camera(batch[camera], self.grids[camera]) for camera in self.cameras], 1
            )
            text = model.vlm.embed(self.input_ids).to(images.dtype)
            embeddings = torch.where(self.image_mask, images[:, self.image_rows], text)
            proprio = model.proprio_embedder(batch[OBS_STATE][:, None]).to(embeddings.dtype)
            embeddings = torch.where(self.state_mask, proprio[:, self.state_rows], embeddings)
            _, cache = model.vlm(
                embeddings,
                full_attention_mask=self.full_mask,
                linear_attention_mask=self.linear_mask,
                position_ids=self.positions,
            )
        return tuple(
            tensor for i in self.full_layers for tensor in (cache.layers[i].keys, cache.layers[i].values)
        )

    def encode_camera(self, image: Tensor, grid: Qwen3VisionGrid) -> Tensor:
        """`G05NativeBackend._encode_camera_transformers` for one frame, on the tower's fixed grid."""
        tower = self.backend.model.vision_tower
        config = tower.config
        patches = G05NativeBackend._patchify(
            image, int(config.patch_size), int(config.temporal_patch_size), int(config.spatial_merge_size)
        )
        hidden_states = tower.patch_embed(patches)
        position = (tower.pos_embed(grid.pos_embed_indices) * grid.pos_embed_weights[:, :, None]).sum(1)
        hidden_states = hidden_states + position.to(hidden_states.dtype)
        for block in tower.blocks:
            hidden_states = replay_vision_block(
                block,
                hidden_states,
                grid.position_embeddings,
                grid.chunk_lengths,
                apply_rotary_pos_emb_vision,
                eager_attention_forward,
            )
        merged = tower.merger(hidden_states)
        return merged.reshape(image.shape[0], -1, merged.shape[-1])


class G05Step(nn.Module):
    """One Euler step of `G05NativeBackend._infer_flow`: the prefix's keys and values, the noisy actions and the
    timestep in, the next noisy actions out."""

    def __init__(self, policy: G05Policy, sequence: dict[str, Tensor], dim_mask: Tensor | None, delta: float):
        super().__init__()
        self.backend = policy.backend
        self.device = policy.config.device
        self.dtype = policy.config.dtype
        self.delta = delta
        vlm_config = self.backend.model.vlm.config
        self.vlm_config = vlm_config
        self.full_layers = [i for i, kind in enumerate(vlm_config.layer_types) if kind == "full_attention"]
        self.register_buffer("token_types", sequence["token_types"])
        self.register_buffer("positions", sequence["positions"])
        self.register_buffer("dim_mask", dim_mask)

    def forward(self, *inputs: Tensor) -> Tensor:
        *cache, x_t, timestep = inputs
        vlm_cache = DynamicCache(config=self.vlm_config)
        for i, keys, values in zip(self.full_layers, cache[::2], cache[1::2], strict=True):
            vlm_cache.layers[i].update(keys, values)
        actions = x_t[None]
        if self.dim_mask is not None:
            # The policy zeroes the padded dimensions of its noise before the first step, and after each one.
            actions = actions.masked_fill(self.dim_mask, 0)
        with _autocast(torch.device(self.device), self.dtype):
            velocity = self.backend._velocity(
                actions, timestep, vlm_cache=vlm_cache, token_types=self.token_types, positions=self.positions
            )
        actions = actions - self.delta * velocity
        if self.dim_mask is not None:
            actions = actions.masked_fill(self.dim_mask, 0)
        return actions[0]


class G05Actions(nn.Module):
    """The denoised chunk in, the actions to play out, in robot units.

    G0.5 unnormalizes each step of its chunk with that step's own statistics, so this runs the postprocessor on the
    whole chunk, as `lerobot-rollout` does, and then keeps the steps it plays.
    """

    def __init__(self, postprocessor: PolicyProcessorPipeline, execution_steps: int):
        super().__init__()
        self.postprocessor = postprocessor
        self.execution_steps = execution_steps

    def forward(self, actions: Tensor) -> Tensor:
        return self.postprocessor(actions[None])[0, : self.execution_steps].float()


class G05ExportRecipe(ChainRecipe):
    """G0.5's System 1 flow head as a chain of programs: see this module's documentation.

    Args:
        policy (`G05Policy`):
            The trained policy, in eval mode on the backend's device.
        preprocessor (`PolicyProcessorPipeline`):
            Its saved preprocessor.
        postprocessor (`PolicyProcessorPipeline`):
            Its saved postprocessor.
        cfg (`ExportConfig`):
            The `lerobot-export` configuration.
    """

    # The flow head runs ten denoising steps by default.
    bfloat16_tolerance = BFLOAT16_STEPS_TOLERANCE
    policy_name = "G0.5"
    text_steps = (RenderRuntimeMessagesStep, RenderTrainingMessagesStep, G05TokenizerStep)
    policy: G05Policy
    task: str

    def __init__(
        self,
        policy: G05Policy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        cfg: ExportConfig,
    ) -> None:  # noqa: D107
        self._device = torch.device(cfg.backend.device)
        # On a GPU this applies the policy's bfloat16 weights and float32 islands, as for inference.
        policy.to(self._device)
        super().__init__(policy, preprocessor, postprocessor, cfg)
        config = policy.config
        model_config = policy.backend.model_config
        if config.runtime_system != "system1" or config.action_head != "flow":
            raise ValueError(
                "G0.5 exports its System 1 flow head only: use a checkpoint with runtime_system=system1 and "
                f"action_head=flow, not {config.runtime_system} and {config.action_head}."
            )
        if config.n_obs_steps != 1:
            raise ValueError("G0.5 exports a checkpoint that reads the current frame only (n_obs_steps=1).")
        self._steps = policy.chunk_inference_spec().execution_steps
        self._generator = torch.Generator().manual_seed(cfg.seed)
        self._metadata = dataset_metadata(cfg)
        self._features = robot_features(config, preprocessor, self._metadata)
        self._visual = {name for name, f in self._features.items() if f.type is FeatureType.VISUAL}
        self.config = copy(config)
        self.config.input_features = self._features
        self.task = export_task(cfg, self._metadata, "g05")
        fm = model_config["fm"]
        self._frame = random_frame(self._features, self._generator)
        self._noise = torch.randn(int(fm["horizon_steps"]), int(fm["action_dim"]), generator=self._generator)
        steps = int(fm["num_inference_steps"])
        self._delta = 1.0 / steps
        # `_infer_flow` subtracts the step from the time in its dtype, float32 here, one step at a time.
        timestep, self._timesteps = torch.ones(1), []
        for _ in range(steps):
            self._timesteps.append(timestep)
            timestep = timestep - self._delta
        self._sequence, self._dim_mask, self._grids = self._prompt()
        self._runs: list[ProgramRun] = []
        self._constants: dict[str, np.ndarray] = {}
        self._actions: np.ndarray | None = None

    def _prompt(self) -> tuple[dict[str, Tensor], Tensor | None, dict[str, Qwen3VisionGrid]]:
        """The prompt's token ids, masks and positions, the padded action dimensions and the cameras' grids.

        They are computed from two random frames, and refused unless both give the same.
        """
        backend = self.policy.backend
        second = random_frame(self._features, self._generator)
        batches = []
        with torch.no_grad():
            for frame in (self._frame, second):
                self.preprocessor.reset()
                batches.append(
                    self.preprocessor(prepare_observation_for_inference(dict(frame), self._device, self.task))
                )
        first, other = batches
        for name in (G05_INPUT_IDS, G05_TOKEN_TYPES, "action_dim_is_pad"):
            a, b = first.get(name), other.get(name)
            if isinstance(a, Tensor) != isinstance(b, Tensor) or (
                isinstance(a, Tensor) and not torch.equal(a, b)
            ):
                raise ValueError(f"G0.5's {name} differs between two frames, so the programs cannot hold it.")
        input_ids, token_types = first[G05_INPUT_IDS], first[G05_TOKEN_TYPES]
        dim_mask = first.get("action_dim_is_pad")
        dim_mask = dim_mask.bool().unsqueeze(1) if isinstance(dim_mask, Tensor) else None

        grids = {}
        patch = int(backend.model.vision_tower.config.patch_size)
        for camera, image in first["pixel_values"].items():
            height, width = image.shape[-2:]
            grids[camera] = Qwen3VisionGrid.build(
                backend.model.vision_tower, torch.tensor([[1, height // patch, width // patch]])
            )
        with torch.no_grad(), _autocast(self._device, self.policy.config.dtype):
            # The policy's own embedding records the grids its rope positions read.
            embeddings = backend._embed(
                _sequence_batch(first),
                first["pixel_values"],
                backend._proprio(list(first["samples"]), self._device),
            )
            positions = backend._mrope_positions(token_types)
            full_mask, linear_mask = backend._causal_mask(token_types, embeddings.dtype)
        image_mask = token_types == G05TokenType.IMAGE
        state_mask = token_types == G05TokenType.PROPRIO
        sequence = {
            "input_ids": input_ids,
            "token_types": token_types,
            "positions": positions,
            "full_mask": full_mask,
            "linear_mask": linear_mask,
            "image_mask": image_mask[..., None],
            "state_mask": state_mask[..., None],
            # Which image token, and which state row, each position of the prompt takes, as `_embed` gathers them.
            "image_rows": (image_mask.long().cumsum(1) - 1).clamp(min=0)[0],
            "state_rows": (state_mask.long().cumsum(1) - 1).clamp(min=0)[0],
        }
        return sequence, dim_mask, grids

    def parts(self) -> Iterator[Part]:
        """The chain's parts, each run in PyTorch to make the next part's inputs."""
        frame_names = list(self._features)
        frame = tuple(torch.from_numpy(x).to(self._device) for x in self._frame.values())
        prefix = G05Prefix(
            self.policy, self._tensor_preprocessor(), frame_names, self._visual, self._sequence, self._grids
        )
        with torch.no_grad():
            cache = prefix(*frame)
        cache_names = [f"past_{kind}_{i}" for i in prefix.full_layers for kind in ("key", "value")]
        yield self._part("prefix", prefix, frame, frame_names, cache_names)

        step = G05Step(self.policy, self._sequence, self._dim_mask, self._delta)
        x_t = self._noise.to(self._device)
        with torch.no_grad():
            for timestep in self._timesteps:
                x_t = step(*cache, x_t, timestep.to(self._device))
        step_inputs = (*cache, self._noise.to(self._device), self._timesteps[0].to(self._device))
        step_names = [*cache_names, "x", "timestep"]
        yield Part("step", step, step_inputs, step_names, ["x"])
        step_constants = [{"timestep": t.numpy().astype(np.float32)} for t in self._timesteps]
        step_runs, self._constants = repeated_runs(
            "step", step_names, ["x"], step_constants, start={"x": NOISE}
        )
        self._runs += step_runs

        actions = G05Actions(self.postprocessor, self._steps)
        with torch.no_grad():
            self._actions = actions(x_t).cpu().numpy()
        yield self._part("actions", actions, (x_t,), ["x"], ["action"])

    def test_case(self) -> TestCase:
        """The random frame and noise, and the actions the parts compute for them, checked against the policy."""
        if self._actions is None:
            raise RuntimeError("The test case is made by `parts`, which has not run yet.")
        if not np.isfinite(self._actions).all():
            raise ValueError(
                "The chained parts compute actions that are not finite, so they cannot be checked."
            )
        width = int(self.policy.backend.model_config["fm"]["action_dim"])
        action_scale = measure_action_scale(
            lambda value: self.postprocessor(torch.full((1, 1, width), value, device=self._device))[0, 0]
        )
        case = TestCase(self._frame, self._noise.numpy(), self._actions, action_scale)
        self._check_against_policy(case)
        return case

    def plan(self) -> RunPlan:
        """The parts in order with the step run once per timestep, and the timesteps."""
        return RunPlan(runs=self._runs, constants=self._constants)

    def _policy_actions(self, case: TestCase) -> np.ndarray:
        """The policy's own actions for the test case's frame and noise."""
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.policy.reset()
        observation = prepare_observation_for_inference(dict(case.frame), self._device, self.task)
        noise = torch.from_numpy(case.noise).to(self._device)
        with torch.no_grad(), ReplayDraws([noise[None]]):
            predicted = self.policy.predict_action_chunk(self.preprocessor(observation))
        with torch.no_grad():
            return self.postprocessor(cast(Tensor, predicted))[0, : self._steps].float().cpu().numpy()


def _sequence_batch(batch: dict[str, Any]) -> G05SequenceBatch:
    """The token sequence of a preprocessed batch, as `G05NativeBackend.predict_action` reads it."""
    return G05SequenceBatch(
        input_ids=batch[G05_INPUT_IDS],
        labels=batch[G05_LABELS],
        token_types=batch[G05_TOKEN_TYPES],
        split_index=batch.get(G05_SPLIT_INDEX),
    )
