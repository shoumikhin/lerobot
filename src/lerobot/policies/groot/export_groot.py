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

"""The GR00T N1.7 export recipe: one action chunk as a chain of small programs.

GR00T holds about 3 billion weights, so one program of the whole policy does not build on an 8 GB board.
The chain is the vision encoder, the language model in groups of up to six layers, the keys and values
every cross-attention block reads, the diffusion model in groups of up to eight blocks run once per denoising
step, and the actions. Each part is traced with only its own weights in memory.

The programs take the robot's frame as it comes: each camera is resized and cropped inside the vision
program, by the saved processor's own resize and crop written as constant matrices. The task comes in as
the embedding rows of its tokens (`task_embeddings`, `task_length`), so a bundle runs any task up to its
token budget. The bundle pads the task's rows to that budget, and the programs keep the padding out of every
attention, so the actions are those of the unpadded prompt.

Qwen3-VL computes its rope positions and image grid with Python loops no exporter can follow. For one
camera size they never change up to the task's text, which follows the images, so they are computed here
and held as constants, as are the timestep tables of the denoising steps.
"""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Iterator
from copy import copy
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from torch import Tensor, nn

from lerobot.bundle.manifest import NOISE, TASK_EMBEDDINGS, TASK_LENGTH, ProgramRun
from lerobot.bundle.task_tokens import clean, tokenize
from lerobot.configs import FeatureType
from lerobot.export.recipe import (
    BFLOAT16_TOLERANCE,
    FLOAT32_TOLERANCE,
    ChainRecipe,
    Part,
    RunPlan,
    TaskTokenData,
    TestCase,
    dataset_metadata,
    export_task,
    load_policy,
    load_processors,
    measure_action_scale,
    preprocessor_renames,
    random_frame,
    refuse_depth_maps,
    refuse_peft_adapter,
    split_absolute_step,
    to_policy_input,
    whole_policy_fits,
)
from lerobot.policies import prepare_observation_for_inference
from lerobot.processor import (
    DeviceProcessorStep,
    PolicyProcessorPipeline,
    TransitionKey,
    policy_output_to_transition,
    to_absolute_actions,
    transition_to_policy_action,
)
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR
from lerobot.utils.feature_utils import dataset_to_policy_features

from .configuration_groot import GROOT_ACTION_DECODE_TRANSFORM_LIBERO, GrootConfig
from .groot_n1_7 import CategorySpecificLinear
from .modeling_groot import GrootPolicy
from .processor_groot import (
    GrootN17ActionDecodeStep,
    GrootN17PackInputsStep,
    GrootN17VLMEncodeStep,
    _n1_7_decode_stats_for_action,
    _n1_7_decode_valid_horizon,
    _transform_n1_7_image_for_vlm_albumentations,
    _transform_n1_7_image_for_vlm_torch,
)
from .utils import config_value, stat_dim_from_entry

if TYPE_CHECKING:
    from lerobot.configs import PolicyFeature
    from lerobot.export.configs import ExportConfig

logger = logging.getLogger(__name__)

VISION = "_groot_model.backbone.model.model.visual"
LANGUAGE_MODEL = "_groot_model.backbone.model.model.language_model"
ACTION_HEAD = "_groot_model.action_head"
# The largest groups that build on an 8 GB board.
LANGUAGE_LAYERS_PER_PART = 6
DIT_BLOCKS_PER_PART = 8
# The text after the images in GR00T's chat prompt, and what its processor reads for an empty task.
TASK_TEMPLATE = "{task}<|im_end|>\n"
DEFAULT_TASK = "Perform the task."
# Room above the longest known task, so a new task a little longer still fits. Each token adds to every chunk.
TASK_TOKEN_MARGIN = 8
# Tasks that probe each text op against the real processor.
EDGE_CASE_TASKS = ("", "!!!", "Pick UP the Red cube, then STOP.", "put_the  bowl\tdown", "Ünïcode 42")
# In 8-bit levels. The program rounds once per stage from exact weights. OpenCV's fixed-point weights, and the
# float32 rounding of torchvision's resize as its GPU kernel computes it, move a pixel by up to one level.
PIXEL_TOLERANCE = 2.0
# The normalized state is computed as the packing step does, so it differs by float32 rounding at most.
STATE_TOLERANCE = 1e-6
# The most pixels of line images handed to the processor's transform at once. All the column lines of a
# 1080x1920 camera at once would take about 15 GiB.
LINE_BATCH_PIXELS = 2**24


class GrootExportRecipe(ChainRecipe):
    """Exports one action chunk of a GR00T N1.7 checkpoint as a chain of programs, a part at a time.

    The parts are `vision`; `language_model_0..`, up to six Qwen3 layers each, the last also running the
    action head's encoders; `keys_values`, every cross-attention block's keys and values, once per chunk;
    `step_0..`, up to eight diffusion blocks each, run once per denoising step, the last with the Euler
    update; and `actions`, which turns the denoised chunk into robot actions. Only one part's model weights
    are in memory at a time, so an 8 GB board can export it.

    When the whole policy fits in memory, the test case is also checked against the policy's own
    `predict_action_chunk` under the same noise.
    """

    policy_name = "GR00T"
    # `export_task` always gives one.
    task: str

    @classmethod
    def load(cls, cfg: ExportConfig) -> GrootExportRecipe:
        """The recipe, which holds no weights: each part loads its own."""
        return cls(cfg)

    def __init__(self, cfg: ExportConfig) -> None:  # noqa: D107
        self.cfg = cfg
        config = cast(GrootConfig, copy(cfg.policy))
        refuse_peft_adapter(config, "GR00T")
        if config.use_flash_attention:
            raise ValueError(
                "GR00T export gives the language model a 4-dimensional attention mask, which flash attention "
                "does not take. Export with --policy.use_flash_attention=false."
            )
        config.device = cfg.backend.device
        self._policy_config = config
        self._device = torch.device(cfg.backend.device)
        self._checkpoint = str(config.pretrained_path)
        if not os.path.isdir(self._checkpoint):
            # Each part reads the checkpoint again, so all read its current commit: a branch that moves during
            # the export cannot mix two versions of the policy in one bundle.
            config.pretrained_revision = self._model_file().parent.name
            cast(GrootConfig, cfg.policy).pretrained_revision = config.pretrained_revision
        self.preprocessor, self.postprocessor = load_processors(config)
        self._encode = _step(self.preprocessor, GrootN17VLMEncodeStep)
        pack = _step(self.preprocessor, GrootN17PackInputsStep)
        self._metadata = dataset_metadata(cfg)
        self._generator = torch.Generator().manual_seed(cfg.seed)
        bfloat16 = config.use_bf16 or config.dtype == torch.bfloat16
        default_tolerance = BFLOAT16_TOLERANCE if bfloat16 else FLOAT32_TOLERANCE
        self.tolerance = cfg.tolerance if cfg.tolerance is not None else default_tolerance
        self.task = export_task(cfg, self._metadata, "GR00T")
        features = self._robot_features(config)
        visual = {name for name, f in features.items() if f.type is FeatureType.VISUAL}
        if self._metadata is not None:
            # Before the preprocessor, which fails on a depth map's single channel. Its packing step picks the
            # cameras by name.
            renames = preprocessor_renames(self.preprocessor)
            picked = set(pack._ordered_image_keys(dict.fromkeys(renames.get(name, name) for name in visual)))
            refuse_depth_maps((name for name in visual if renames.get(name, name) in picked), self._metadata)
        frame = random_frame(features, self._generator)
        self._batch = self._preprocess(frame, self.task)
        self._images = _ImagesAndState(self.preprocessor, frame, visual, self._batch).to(self._device)
        # The robot sends only the cameras the policy reads, and its state.
        self._frame = {name: frame[name] for name in self._images.frame_names}
        self.config = copy(config)
        self.config.input_features = {name: features[name] for name in self._frame}
        self._check_pixels()
        # A small submodule every checkpoint has, for the model's config.
        meta = self._load_modules(f"{VISION}.patch_embed")
        model = meta._groot_model
        self._model_config = model.config
        # The backbone keeps at most select_layer of its layers.
        text_layers = model.backbone.model.config.text_config.num_hidden_layers
        self._language_depth = min(self._model_config.select_layer, text_layers)
        self._steps = meta.chunk_inference_spec().execution_steps
        self._action_dim = cast("dict[str, PolicyFeature]", config.output_features)[ACTION].shape[0]
        # Built before the large parts load, so a checkpoint whose actions do not export is refused early.
        self._chunk_actions = _chunk_actions(
            self._steps, self._action_dim, self.preprocessor, self.postprocessor
        ).to(self._device)
        self._prompt_ops = _prompt_ops(pack.formalize_language)
        self._prompt = self._prompt_constants(model.backbone.model)
        self._noise = torch.randn(
            self._model_config.action_horizon, self._model_config.max_action_dim, generator=self._generator
        )
        steps = self._model_config.num_inference_timesteps
        if steps < 1:
            raise ValueError(f"GR00T denoises in at least one step, not num_inference_timesteps={steps}.")
        buckets = self._model_config.num_timestep_buckets
        self._timesteps = [int(i / steps * buckets) for i in range(steps)]
        self._step_tables: dict[str, np.ndarray] = {}
        self._runs: list[ProgramRun] = []
        self._actions: np.ndarray | None = None

    def parts(self) -> Iterator[Part]:
        """The chain's parts, each run in PyTorch to make the next part's inputs, one in memory at a time.

        The test case's actions are these parts chained, the step parts once per denoising step.
        """
        device, use_bf16 = self._device, self._policy_config.use_bf16
        frame = tuple(torch.from_numpy(x).to(device) for x in self._frame.values())
        policy = self._load_modules(VISION)
        module = GrootVision(
            policy._groot_model.backbone.visual, self._images, self._batch["image_grid_thw"], use_bf16
        ).eval()
        del policy
        with torch.no_grad():
            hidden, *deepstack, state = module(*frame)
        deepstack_names = [f"deepstack_{i}" for i in range(len(deepstack))]
        yield self._part(
            "vision", module, frame, list(self._frame), ["image_embeds", *deepstack_names, "state"]
        )
        del module
        _free_memory()

        table = self._embedding_table()
        head_embeds = table[self._prompt["head_ids"]].to(device, self._policy_config.dtype)[None]
        task_embeddings, task_length = self._task_rows(table, self.task)
        del table
        depth, hidden_name = self._language_depth, "image_embeds"
        for group, first in enumerate(range(0, depth, LANGUAGE_LAYERS_PER_PART)):
            stop = min(first + LANGUAGE_LAYERS_PER_PART, depth)
            modules = [f"{LANGUAGE_MODEL}.layers.{i}" for i in range(first, stop)]
            if stop == depth:
                modules += [f"{ACTION_HEAD}.{n}" for n in ("vlln", "vl_self_attention", "state_encoder")]
            _free_memory()
            policy = self._load_modules(*modules)
            module = GrootLanguageModel(
                policy._groot_model.backbone.language_model,
                policy._groot_model.action_head,
                self._prompt,
                head_embeds,
                first,
                stop,
                use_bf16,
                len(deepstack),
            ).eval()
            del policy
            task_input = ((task_embeddings,), [TASK_EMBEDDINGS]) if first == 0 else ((), [])
            state_input = ((state, task_length), ["state", TASK_LENGTH]) if module.last else ((), [])
            inputs = (
                hidden,
                *(deepstack[i] for i in module.deepstack_layers),
                *task_input[0],
                *state_input[0],
            )
            names = [
                hidden_name,
                *(deepstack_names[i] for i in module.deepstack_layers),
                *task_input[1],
                *state_input[1],
            ]
            with torch.no_grad():
                result = module(*inputs)
            outputs = ["vl_embeds", "state_features"] if module.last else [f"language_hidden_{group}"]
            yield self._part(f"language_model_{group}", module, inputs, names, outputs)
            if module.last:
                vl_embeds, state_features = result
            else:
                hidden = result
            hidden_name = outputs[0]
            del module
        _free_memory()

        crossing = self._cross_blocks()
        modules = [
            f"{ACTION_HEAD}.model.transformer_blocks.{i}.attn1.{p}"
            for i in crossing
            for p in ("to_k", "to_v")
        ]
        module = GrootKeysValues(
            self._load_modules(*modules)._groot_model.action_head, crossing, use_bf16
        ).eval()
        with torch.no_grad():
            keys_values = module(vl_embeds)
        kv_names = [f"{kind}_{i}" for i in crossing for kind in ("key", "value")]
        yield self._part("keys_values", module, (vl_embeds,), ["vl_embeds"], kv_names)
        del module
        given = dict(zip(kv_names, keys_values, strict=True))

        depth = self._model_config.diffusion_model_cfg["num_layers"]
        x_names = [NOISE, *(f"x_{i + 1}" for i in range(len(self._timesteps)))]
        actions = self._noise.to(device)
        for iteration in range(len(self._timesteps)):
            carried: list[Tensor] = []
            for group, first in enumerate(range(0, depth, DIT_BLOCKS_PER_PART)):
                stop = min(first + DIT_BLOCKS_PER_PART, depth)
                _free_memory()
                module = GrootStep(
                    self._load_modules(*self._step_modules(first, stop, depth))._groot_model.action_head,
                    self._prompt,
                    first,
                    stop,
                    crossing,
                    use_bf16,
                    1.0 / len(self._timesteps),
                    self._policy_config.dtype,
                    vl_embeds.dtype,
                ).eval()
                if first == 0 and not self._step_tables:
                    self._step_tables = module.time_tables(self._timesteps)
                time_encoding, temb = (
                    torch.from_numpy(self._step_tables[f"{name}_{iteration}"]).to(device)
                    for name in ("time_encoding", "temb")
                )
                block_names = [f"{kind}_{i}" for i in module.cross for kind in ("key", "value")]
                step_inputs: tuple[Tensor, ...] = tuple(given[n] for n in block_names)
                if first == 0:
                    step_inputs += (state_features, actions, time_encoding, temb)
                    names = ["state_features", "x_t", "time_encoding", "temb"]
                else:
                    step_inputs += (carried[0], temb)
                    names = [f"step_hidden_{group - 1}", "temb"]
                    if module.last:
                        step_inputs, names = (*step_inputs, actions), [*names, "x_t"]
                step_inputs, names = (*step_inputs, task_length), [*block_names, *names, TASK_LENGTH]
                with torch.no_grad():
                    result = module(*step_inputs)
                outputs = ["x_next"] if module.last else [f"step_hidden_{group}"]
                # Every denoising step runs the same programs: yield them once, then only finish the test case.
                if iteration == 0:
                    yield Part(f"step_{group}", module, step_inputs, names, outputs)
                step_values = {
                    "x_t": x_names[iteration],
                    "x_next": x_names[iteration + 1],
                    "time_encoding": f"time_encoding_{iteration}",
                    "temb": f"temb_{iteration}",
                }
                self._runs.append(
                    ProgramRun(
                        f"step_{group}_{iteration}",
                        f"step_{group}",
                        [step_values.get(name, name) for name in names],
                        [step_values.get(name, name) for name in outputs],
                    )
                )
                if module.last:
                    actions = result
                else:
                    carried = [result]
                del module
        _free_memory()

        module = self._chunk_actions
        inputs, names = (
            (actions, torch.from_numpy(self._frame[self._images.state_name]).to(device)),
            [x_names[-1], self._images.state_name],
        )
        if not module.reads_state:
            # A program takes only the inputs it reads.
            inputs, names = inputs[:1], names[:1]
        with torch.no_grad():
            self._actions = module(*inputs).float().cpu().numpy()
        if isinstance(module, _N17ChunkActions):
            self._check_n17_decode(actions)
        yield self._part("actions", module, inputs, names, [ACTION])

    def plan(self) -> RunPlan:
        """The runs, with the step parts once per denoising step, their time tables, and the task's tokens."""
        task_tokens = TaskTokenData(
            prompt_ops=self._prompt_ops,
            tokenizer=self._prompt["tokenizer"],
            embeddings=self._embedding_table(),
            known_tasks=self._prompt["known_tasks"],
            max_tokens=self._prompt["max_tokens"],
        )
        return RunPlan(runs=self._runs, constants=dict(self._step_tables), task_tokens=task_tokens)

    def test_case(self) -> TestCase:
        """The frame, the noise and the chained parts' actions, checked against the policy when it fits."""
        if self._actions is None:
            raise RuntimeError("The test case is made by `parts`, which has not run yet.")
        case = TestCase(self._frame, self._noise.numpy(), self._actions, self._action_scale())
        self._check_against_policy(case)
        return case

    def _load_modules(self, *modules: str) -> GrootPolicy:
        """The policy with only `modules`' weights read, and its linear layers in bfloat16 under `use_bf16`.

        Autocast would cast a float32 linear layer's weight to bfloat16 on every call; holding it in bfloat16
        is the same arithmetic, once.
        """
        config = copy(self._policy_config)
        policy = GrootPolicy.load_checkpoint_modules(
            self._checkpoint, config, list(modules), revision=config.pretrained_revision
        )
        if self._policy_config.use_bf16:
            for layer in policy.modules():
                if isinstance(layer, nn.Linear):
                    layer.to(torch.bfloat16)
        return policy

    def _embedding_table(self) -> Tensor:
        """The token embedding table, in bfloat16 when that holds it exactly, on the CPU."""
        policy = self._load_modules(f"{LANGUAGE_MODEL}.embed_tokens")
        table = policy._groot_model.backbone.language_model.get_input_embeddings().weight.detach().cpu()
        del policy
        if table.dtype == torch.float32 and torch.equal(table, table.to(torch.bfloat16).float()):
            table = table.to(torch.bfloat16)
        return table

    def _task_rows(self, table: Tensor, task: str) -> tuple[Tensor, Tensor]:
        """What the bundle gives the programs for `task`: its rows padded to the budget, and their count."""
        ids = self._prompt["known_tasks"][task]
        rows = torch.zeros(self._prompt["max_tokens"], table.shape[1])
        rows[: len(ids)] = table[ids].float()
        return rows.to(self._device), torch.tensor([len(ids)], device=self._device)

    def _cross_blocks(self) -> list[int]:
        """The diffusion blocks that attend to the vision-language features; with interleaving, every other one."""
        dit = self._model_config.diffusion_model_cfg
        interleave = dit.get("interleave_self_attention", False)
        return [i for i in range(dit["num_layers"]) if not (interleave and i % 2 == 1)]

    def _step_modules(self, first: int, stop: int, depth: int) -> list[str]:
        modules = [f"{ACTION_HEAD}.model.transformer_blocks.{i}" for i in range(first, stop)]
        if first == 0:
            modules += [f"{ACTION_HEAD}.action_encoder", f"{ACTION_HEAD}.model.timestep_encoder"]
            if self._model_config.add_pos_embed:
                modules.append(f"{ACTION_HEAD}.position_embedding")
        if stop == depth:
            modules += [f"{ACTION_HEAD}.model.{n}" for n in ("norm_out", "proj_out_1", "proj_out_2")]
            modules.append(f"{ACTION_HEAD}.action_decoder")
        return modules

    def _robot_features(self, config: GrootConfig) -> dict[str, PolicyFeature]:
        """The arrays the robot may send: the policy's inputs, or the dataset's observations it may read.

        With a dataset, these are the observations the policy reads after its preprocessor's renames, and
        every camera: GR00T's processor picks its cameras by the checkpoint's own camera names, not by the
        policy's input features, which may name placeholders. The frame then keeps the cameras the processor
        picks.
        """
        features = dict(config.input_features or {})
        if self._metadata is None:
            return features
        renames = preprocessor_renames(self.preprocessor)
        dataset = dataset_to_policy_features(self._metadata.features)
        kept = {
            name: f
            for name, f in dataset.items()
            if name.startswith(f"{OBS_STR}.")
            and (renames.get(name, name) in features or f.type is FeatureType.VISUAL)
        }
        given = {renames.get(name, name) for name in kept}
        missing = sorted(
            n for n, f in features.items() if f.type is not FeatureType.VISUAL and n not in given
        )
        if missing:
            raise ValueError(
                f"The dataset gives nothing the policy reads as {missing}, even after its preprocessor's "
                f"renames ({renames}). The dataset has {sorted(dataset)}."
            )
        return kept

    def _preprocess(self, frame: dict[str, np.ndarray], task: str) -> dict[str, Any]:
        """The saved preprocessor's batch for `frame` and `task`, as `lerobot-rollout` makes it."""
        observation = prepare_observation_for_inference(dict(frame), self._device, task)
        with torch.no_grad():
            return self.preprocessor(observation)

    def _check_pixels(self) -> None:
        """Refuse an export whose program images or state differ from the saved processor's on the test frame."""
        inputs = tuple(
            torch.from_numpy(self._frame[name]).to(self._device) for name in self._images.frame_names
        )
        with torch.no_grad():
            pixels, state = self._images(*inputs)
        expected = self._processor_pixels().to(pixels)
        if pixels.shape != expected.shape:
            raise ValueError(
                f"The program's image patches are shaped {tuple(pixels.shape)}, the processor's {tuple(expected.shape)}."
            )
        # Rounded, as float32 normalization leaves a whole level a hair above itself.
        levels = round(float(self._images.levels(pixels - expected).abs().max()), 3)
        logger.info(
            "Image check: the program's pixels are within %.3g levels of the saved processor's", levels
        )
        if not levels <= PIXEL_TOLERANCE:
            raise ValueError(
                f"The program's resized images differ from the saved processor's by {levels:.3g} levels, more "
                f"than {PIXEL_TOLERANCE}. The processor's resize is not one this recipe reproduces."
            )
        state_error = float((state - self._batch["state"].to(state)).abs().max())
        if not state_error <= STATE_TOLERANCE:
            raise ValueError(f"The program's state differs from the saved processor's by {state_error:.3g}.")

    def _processor_pixels(self) -> Tensor:
        """The saved processor's image patches for the test frame, with its resizes computed as on a GPU.

        torchvision's GPU kernel computes each resize in float32 and rounds it once to 8 bits, as the program
        does each stage. Its 8-bit CPU kernel also rounds and clamps between the two passes of a resize, by up
        to a dozen levels next to sharp edges, so on the CPU the processor's torchvision transform runs here
        stage by stage in float32 instead.
        """
        if self._encode.use_albumentations or self._device.type != "cpu":
            return self._batch["pixel_values"]
        steps = self.preprocessor.steps
        pack = next(i for i, s in enumerate(steps) if isinstance(s, GrootN17PackInputsStep))
        observation = prepare_observation_for_inference(dict(self._frame), self._device, self.task)
        transition = self.preprocessor.to_transition(observation)
        with torch.no_grad():
            for step in steps[: pack + 1]:
                transition = step(transition)
        # The 8-bit cameras the encoding step reads: batch, time, camera, height, width, channel.
        video = torch.as_tensor(cast(dict, transition[TransitionKey.OBSERVATION])["video"])
        frames = []
        for camera in video[0].flatten(0, 1):
            image = camera.permute(2, 0, 1).float()
            for geometry in _stage_geometries(self._encode):
                image = _transform_n1_7_image_for_vlm_torch(image, **geometry).clamp(0, 255).round()
            frames.append(image.to(torch.uint8))
        return self._encode.proc.image_processor(images=frames, return_tensors="pt")["pixel_values"]

    def _prompt_constants(self, qwen: nn.Module) -> dict[str, Any]:
        """The prompt's fixed head, its rope positions, and the token ids of every task known at export.

        The prompt is a head that ends with the last image, then the task's text. The prompt ops must give
        the processor's own ids for every known task and a few edge cases; this refuses otherwise.
        """
        from transformers.utils import cached_file

        ids = self._batch["input_ids"]
        vision_end = self._encode.proc.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        head = int(torch.nonzero(ids[0] == vision_end).max()) + 1
        tokenizer_file = cached_file(self._encode.model_name, "tokenizer.json")
        # A dataset without episodes has no tasks.
        tasks = None if self._metadata is None else self._metadata.tasks
        dataset_tasks = [] if tasks is None else [str(task) for task in tasks.index]
        known_tasks = list(dict.fromkeys([self.task, *dataset_tasks]))
        checked = list(dict.fromkeys([*known_tasks, *EDGE_CASE_TASKS]))
        texts = [clean(self._prompt_ops, task) for task in checked]
        token_ids = dict(zip(checked, tokenize(tokenizer_file, texts), strict=True))
        for task in checked:
            batch = self._batch if task == self.task else self._preprocess(self._frame, task)
            prompt = batch["input_ids"][0].tolist()
            if prompt[:head] != ids[0, :head].tolist() or prompt[head:] != token_ids[task]:
                raise ValueError(
                    f"GR00T's processor turns the task {task!r} into other tokens than the bundle's prompt ops "
                    f"{self._prompt_ops} and tokenizer do."
                )
        max_tokens = (
            self.cfg.max_task_tokens or max(len(token_ids[t]) for t in known_tasks) + TASK_TOKEN_MARGIN
        )
        if len(token_ids[self.task]) > max_tokens:
            raise ValueError(
                f"The task is {len(token_ids[self.task])} tokens long; raise --max_task_tokens to at least that."
            )
        known = {}
        for task in known_tasks:
            if len(token_ids[task]) <= max_tokens:
                known[task] = token_ids[task]
            else:
                logger.warning(
                    "Leaving out the dataset task %r: it is longer than %d tokens.", task, max_tokens
                )
        image_mask = ids == qwen.config.image_token_id
        positions, _ = qwen.model.get_rope_index(
            input_ids=ids,
            mm_token_type_ids=self._batch.get("mm_token_type_ids", image_mask.int()),
            image_grid_thw=self._batch["image_grid_thw"],
            attention_mask=self._batch["attention_mask"],
        )
        # Qwen3-VL numbers the text after the images on from the largest image position, in all three axes.
        task_positions = torch.arange(max_tokens, device=ids.device)
        text_positions = positions[:, :, :head].max() + 1 + task_positions
        if not torch.equal(positions[:, :, head:], text_positions[: ids.shape[1] - head].expand(3, 1, -1)):
            raise ValueError("Qwen3-VL's rope positions of the task's text are not consecutive.")
        return {
            "head": head,
            "head_ids": ids[0, :head].tolist(),
            "position_ids": torch.cat((positions[:, :, :head], text_positions.expand(3, 1, -1)), dim=2),
            "image_mask": F.pad(image_mask[:, :head], (0, max_tokens)),
            "task_positions": task_positions,
            "embodiment_id": self._batch["embodiment_id"],
            "tokenizer": Path(tokenizer_file),
            "known_tasks": known,
            "max_tokens": max_tokens,
        }

    def _action_scale(self) -> np.ndarray:
        """Robot units per normalized unit for each action, at the chunk step where it is largest.

        N1.7's decoding gives it from its statistics, since a relative end-effector pose turns with the test
        state. Otherwise it is measured on the actions part.
        """
        if isinstance(self._chunk_actions, _N17ChunkActions):
            return self._chunk_actions.decode.action_scale()
        width = self._model_config.max_action_dim
        state = torch.from_numpy(self._frame[self._images.state_name]).to(self._device)
        scale = measure_action_scale(
            lambda value: self._chunk_actions(
                torch.full((self._steps, width), value, device=self._device), state
            )
        )
        return scale.max(0)

    def _check_n17_decode(self, actions: Tensor) -> None:
        """`N17ActionDecode` must decode the test chunk as `GrootN17ActionDecodeStep` does."""
        # The step reads the reference state its packing step cached for the test frame.
        self._preprocess(self._frame, self.task)
        chunk = actions[None, : self._steps, : self._action_dim].float()
        with torch.no_grad():
            expected = self.postprocessor(chunk)[0].float().cpu().numpy()
        error = float((np.abs(self._actions - expected) / self._action_scale()).max())
        logger.info("N1.7 decode check: %.2e normalized units from GrootN17ActionDecodeStep", error)
        if not error <= FLOAT32_TOLERANCE:
            raise ValueError(
                f"N17ActionDecode moved the test chunk's actions by {error:.2e} from GrootN17ActionDecodeStep."
            )

    def _policy_actions(self, case: TestCase) -> np.ndarray | None:
        """The policy's own actions for the test case, when the whole policy fits in memory."""
        if not whole_policy_fits(self._model_file(), self._device, self.policy_name, 3):
            return None
        policy, preprocessor, postprocessor = load_policy(self.cfg)
        observation = prepare_observation_for_inference(dict(case.frame), self._device, self.task)
        noise = torch.from_numpy(case.noise).to(self._device)[None]
        with torch.no_grad():
            output = policy.predict_action_chunk(preprocessor(observation), noise=noise)
            chunk = transition_to_policy_action(policy_output_to_transition(output))[:, : self._steps]
            expected = postprocessor(chunk)[0].float().cpu().numpy()
        del policy
        _free_memory()
        return expected

    def _model_file(self) -> Path:
        from transformers.utils import cached_file

        revision = self._policy_config.pretrained_revision
        return Path(cast(str, cached_file(self._checkpoint, SAFETENSORS_SINGLE_FILE, revision=revision)))


class _ImagesAndState(nn.Module):
    """The saved preprocessor's image and state steps: the robot's frame in, image patches and state out.

    Each camera goes through the saved processor's resize and crop as constant matrices, one pair per
    stage, rounded to whole levels where the processor rounds to 8 bits. The matrices are read off the
    processor's own transform by running it on images of lines, so both its OpenCV mode and its torchvision
    mode, and a letterbox pad, are reproduced. torchvision's mode is reproduced as its GPU kernel computes it,
    in float32; its 8-bit CPU kernel differs by up to a dozen levels next to sharp edges. Then the images are
    normalized and cut into patches.
    """

    def __init__(
        self,
        preprocessor: PolicyProcessorPipeline,
        frame: dict[str, np.ndarray],
        visual: set[str],
        batch: dict[str, Any],
    ):
        super().__init__()
        pack_index = next(
            i for i, s in enumerate(preprocessor.steps) if isinstance(s, GrootN17PackInputsStep)
        )
        pack = cast(GrootN17PackInputsStep, preprocessor.steps[pack_index])
        encode = _step(preprocessor, GrootN17VLMEncodeStep)
        if pack.video_horizon not in (None, 1):
            raise ValueError("GR00T export takes one observation frame per camera, not a video history.")
        ip = encode.proc.image_processor
        self.preprocessor = PolicyProcessorPipeline(steps=preprocessor.steps[:pack_index])
        self.visual = visual
        observation = {n: to_policy_input(torch.from_numpy(x), n in visual) for n, x in frame.items()}
        sample = self.preprocessor(observation)
        # The cameras the policy feeds, in its order, as its own packing step picks them.
        self.cameras = pack._ordered_image_keys(sample)
        if not self.cameras:
            raise ValueError("The frame has no camera the policy reads.")
        renames = preprocessor_renames(self.preprocessor)
        robot_names = {renames.get(name, name): name for name in frame}
        self.state_name = robot_names[OBS_STATE]
        self.frame_names = [n for n in frame if renames.get(n, n) in self.cameras or n == self.state_name]
        grids = batch["image_grid_thw"].tolist()
        if len(grids) != len(self.cameras) or any(g != grids[0] for g in grids) or grids[0][0] != 1:
            raise ValueError("GR00T export needs one image grid, the same for every camera.")
        _, self.grid_h, self.grid_w = grids[0]
        self.patch_size, self.temporal_patch_size, self.merge_size = (
            ip.patch_size,
            ip.temporal_patch_size,
            ip.merge_size,
        )
        size = (self.grid_h * ip.patch_size, self.grid_w * ip.patch_size)
        self.num_stages: list[int] = []
        for index, camera in enumerate(self.cameras):
            height, width = frame[robot_names[camera]].shape[:2]
            stages = image_stages(encode, height, width)
            out = (stages[-1][0].shape[0], stages[-1][1].shape[0]) if stages else (height, width)
            if out != size:
                raise ValueError(
                    f"The processor resizes {camera} to {out}, then Qwen3-VL's image processor to {size}; "
                    "export reproduces only the first resize. Set the checkpoint's image_target_size to a "
                    f"size Qwen3-VL keeps: each side a multiple of {ip.patch_size * ip.merge_size}, the patch "
                    "size times the merge size, and the area within its image processor's pixel limits."
                )
            for stage, (rows, columns) in enumerate(stages):
                self.register_buffer(f"rows_{index}_{stage}", rows)
                self.register_buffer(f"columns_{index}_{stage}", columns)
            self.num_stages.append(len(stages))
        self.rescale, self.normalize = ip.do_rescale, ip.do_normalize
        self.rescale_factor = ip.rescale_factor
        scale = 1 / ip.rescale_factor if self.rescale else 1
        self.register_buffer("image_mean", (torch.tensor(ip.image_mean) * scale)[None, :, None, None])
        self.register_buffer("image_std", (torch.tensor(ip.image_std) * scale)[None, :, None, None])
        dim = frame[self.state_name].size
        self.max_state_dim = pack.max_state_dim
        self.normalize_state = pack.normalize_min_max and pack.stats is not None and OBS_STATE in pack.stats
        self.clip_state = pack.clip_outliers
        if self.normalize_state:
            stats = cast(dict, pack.stats)[OBS_STATE]
            low = torch.as_tensor(stats.get("min", torch.zeros(dim)), dtype=torch.float32).flatten()[:dim]
            high = torch.as_tensor(stats.get("max", torch.ones(dim)), dtype=torch.float32).flatten()[:dim]
            low, high = F.pad(low, (0, dim - low.numel())), F.pad(high, (0, dim - high.numel()), value=1)
            span = high - low
            self.register_buffer("state_min", low)
            self.register_buffer("state_nonzero", span != 0)
            self.register_buffer("state_span", torch.where(span != 0, span, torch.ones_like(span)))

    def levels(self, patches: Tensor) -> Tensor:
        """Differences of patches in 8-bit levels per channel."""
        channels = patches.reshape(patches.shape[0], 3, -1)
        if self.normalize:
            return channels * self.image_std.reshape(1, 3, 1)
        return channels / self.rescale_factor if self.rescale else channels

    def forward(self, *frame: Tensor) -> tuple[Tensor, Tensor]:
        obs = self.preprocessor(
            {n: to_policy_input(x, n in self.visual) for n, x in zip(self.frame_names, frame, strict=True)}
        )
        images = []
        for index, camera in enumerate(self.cameras):
            # As the packing step converts each camera to 8 bits.
            image = (obs[camera].clamp(0, 1) * 255).trunc()
            for stage in range(self.num_stages[index]):
                rows, columns = (
                    getattr(self, f"rows_{index}_{stage}"),
                    getattr(self, f"columns_{index}_{stage}"),
                )
                image = (rows @ image @ columns.T).clamp(0, 255).round()
            images.append(image)
        images = torch.cat(images)
        if self.normalize:
            images = (images - self.image_mean) / self.image_std
        elif self.rescale:
            images = images * self.rescale_factor
        p, t, m = self.patch_size, self.temporal_patch_size, self.merge_size
        # Channel and repeated time are joined to stay within TensorRT's eight dimensions.
        patches = images[:, :, None].repeat(1, 1, t, 1, 1)
        patches = patches.reshape(len(self.cameras), 3 * t, self.grid_h // m, m, p, self.grid_w // m, m, p)
        pixels = patches.permute(0, 2, 5, 3, 6, 1, 4, 7).reshape(-1, 3 * t * p * p)
        state = obs[OBS_STATE]
        if self.normalize_state:
            state = torch.where(self.state_nonzero, 2 * (state - self.state_min) / self.state_span - 1, 0)
            if self.clip_state:
                state = state.clamp(-1, 1)
        return pixels, F.pad(state[:, None], (0, self.max_state_dim - state.shape[-1]))


def image_stages(encode: GrootN17VLMEncodeStep, height: int, width: int) -> list[tuple[Tensor, Tensor]]:
    """The saved processor's resize and crop of a `height` x `width` camera, as (rows, columns) matrix pairs.

    A stage maps an image `x` to `rows @ x @ columns.T`, then the processor rounds it to 8 bits. The first
    stage is the first resize, after the letterbox pad when there is one; the second the crop and the last
    resize. Each matrix is read off the processor's own transform, run in float32 on images of single lines,
    and the stages together are checked against one call of the whole transform. A stage that changes nothing
    is left out.
    """
    geometry = _geometry(encode)
    if encode.image_target_size is None:
        return []
    stages = []
    shape = (height, width)
    for stage_geometry in _stage_geometries(encode):
        rows, columns = _separable_map(lambda lines, g=stage_geometry: _transform(encode, g, lines), shape)
        shape = (rows.shape[0], columns.shape[0])
        identity = torch.equal(rows, torch.eye(rows.shape[0])) and torch.equal(
            columns, torch.eye(columns.shape[0])
        )
        if not identity:
            stages.append((rows, columns))
    probe = (
        torch.rand(1, height, width, generator=torch.Generator().manual_seed(0), dtype=torch.float64) * 255
    )
    expected = torch.from_numpy(_transform(encode, geometry, probe.float().numpy())).double()
    actual = probe
    for rows, columns in stages:
        actual = rows.double() @ actual @ columns.double().T
    if actual.shape != expected.shape or not torch.allclose(actual, expected, rtol=0, atol=1e-2):
        raise ValueError(
            "GR00T's image transform is not a resize, crop and resize this recipe can write as matrices."
        )
    return stages


def _geometry(encode: GrootN17VLMEncodeStep) -> dict[str, Any]:
    """The arguments the processor's image transforms take from its encoding step."""
    return {
        "image_crop_size": encode.image_crop_size,
        "image_target_size": encode.image_target_size,
        "shortest_image_edge": encode.shortest_image_edge,
        "crop_fraction": encode.crop_fraction,
        "letter_box_transform": encode.letter_box_transform,
    }


def _stage_geometries(encode: GrootN17VLMEncodeStep) -> list[dict[str, Any]]:
    """The transform's arguments for each stage `image_stages` writes: the pad and first resize, then the
    rest."""
    geometry = _geometry(encode)
    if encode.image_target_size is None:
        return []
    edge = encode.shortest_image_edge or encode.image_target_size[0]
    first = {**geometry, "image_crop_size": None, "crop_fraction": 1.0, "shortest_image_edge": edge}
    if not encode.use_albumentations:
        # The torchvision mode resizes to the target after the crop; without a crop that is a second resize.
        first["image_target_size"] = [edge, edge]
    return [first, geometry]


def _transform(encode: GrootN17VLMEncodeStep, geometry: dict[str, Any], lines: np.ndarray) -> np.ndarray:
    """The processor's eval transform of each float32 image in `lines` (count, height, width)."""
    if encode.use_albumentations:
        # Three channels, as the camera images it takes.
        return np.stack(
            [
                _transform_n1_7_image_for_vlm_albumentations(
                    np.repeat(line[:, :, None], 3, axis=2), **geometry
                )[:, :, 0]
                for line in lines
            ]
        )
    return _transform_n1_7_image_for_vlm_torch(
        torch.from_numpy(np.ascontiguousarray(lines)), **geometry
    ).numpy()


def _separable_map(transform, shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """Matrices `rows` and `columns` with `transform(x) == rows @ x @ columns.T` for a separable linear map.

    The response to a full row of ones is a column of `rows` times the summed columns map, and likewise for
    a column of ones, so both are read off up to one scale, fixed by the response to an image of ones.
    """
    height, width = shape
    ones = transform(np.ones((1, height, width), np.float32))[0].astype(np.float64)
    i0, j0 = np.unravel_index(np.argmax(np.abs(ones)), ones.shape)
    batch = max(1, LINE_BATCH_PIXELS // (height * width))
    row_responses: list[np.ndarray] = []
    column_responses: list[np.ndarray] = []
    for first in range(0, height, batch):
        index = np.arange(first, min(first + batch, height))
        lines: np.ndarray = np.zeros((len(index), height, width), np.float32)
        lines[np.arange(len(index)), index, :] = 1
        row_responses.append(transform(lines)[:, :, j0])
    for first in range(0, width, batch):
        index = np.arange(first, min(first + batch, width))
        lines = np.zeros((len(index), height, width), np.float32)
        lines[np.arange(len(index)), :, index] = 1
        column_responses.append(transform(lines)[:, i0, :])
    rows = np.concatenate(row_responses).astype(np.float64).T
    columns = np.concatenate(column_responses).astype(np.float64).T / ones[i0, j0]
    # Contiguous, so an exporter does not warn about the strides of a transposed array.
    return torch.from_numpy(rows).float().contiguous(), torch.from_numpy(columns).float().contiguous()


class _VisionTower(nn.Module):
    """Qwen3-VL's vision tower for one image grid, with its position and rotary embeddings computed once.

    The rotary table is computed from the exact float32 frequencies: a policy built in bfloat16 rounds its
    own copy of them.
    """

    def __init__(self, vision_model: nn.Module, grid_thw: Tensor):
        from transformers.vision_utils import (
            get_vision_interpolation_indices_and_weights,
            get_vision_position_ids,
        )

        super().__init__()
        self.chunk_sizes = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).tolist()
        self.patch_embed = vision_model.patch_embed
        self.blocks = vision_model.blocks
        self.merger = vision_model.merger
        self.deepstack_visual_indexes = vision_model.deepstack_visual_indexes
        self.deepstack_merger_list = vision_model.deepstack_merger_list
        rotary = vision_model.rotary_pos_emb
        exact, _ = rotary.compute_axial_rope_parameters(rotary.config)
        rotary.inv_freq = _exact_inv_freq(rotary.inv_freq, exact).to(rotary.inv_freq.device)
        with torch.no_grad():
            # As the tower's own forward derives them from the grid.
            indices, weights = get_vision_interpolation_indices_and_weights(
                grid_thw,
                num_grid_per_side=vision_model.num_grid_per_side,
                mode=vision_model.interpolation_mode,
                align_corners=vision_model.interpolation_align_corners,
                spatial_merge_size=vision_model.config.spatial_merge_size,
            )
            pos_embeds = (vision_model.pos_embed(indices) * weights[:, :, None]).sum(1)
            position_ids = get_vision_position_ids(grid_thw, vision_model.spatial_merge_size)
            # The rotary embedding only takes the device from its first argument.
            cos, sin = rotary(position_ids, position_ids)
        self.register_buffer("pos_embeds", pos_embeds.to(vision_model.pos_embed.weight.dtype).contiguous())
        self.register_buffer("rotary_cos", cos.contiguous())
        self.register_buffer("rotary_sin", sin.contiguous())

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, ...]:
        """The merged image features, then one tensor per deepstack layer."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

        # Added in float32 under bfloat16 autocast, so the residual stream stays float32 as in NVIDIA's GR00T
        # (transformers 4.57). Transformers 5.9 rounds the sum to bfloat16, nearly doubling the features' error.
        hidden = self.patch_embed(pixel_values) + self.pos_embeds
        deepstack = []
        for layer, block in enumerate(self.blocks):
            attn = block.attn
            x = block.norm1(hidden)
            q, k, v = attn.qkv(x).reshape(x.shape[0], 3, attn.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            q, k = apply_rotary_pos_emb_vision(q, k, self.rotary_cos, self.rotary_sin)
            # Each image attends only to itself; its patch count is fixed here instead of read in the graph.
            outputs = []
            for q_c, k_c, v_c in zip(
                q.split(self.chunk_sizes), k.split(self.chunk_sizes), v.split(self.chunk_sizes), strict=True
            ):
                q_c, k_c, v_c = (t.transpose(0, 1)[None] for t in (q_c, k_c, v_c))
                outputs.append(
                    F.scaled_dot_product_attention(q_c, k_c, v_c, scale=attn.scaling)[0].transpose(0, 1)
                )
            hidden = hidden + attn.proj(torch.cat(outputs).reshape(x.shape[0], -1).contiguous())
            hidden = hidden + block.mlp(block.norm2(hidden))
            if layer in self.deepstack_visual_indexes:
                deepstack.append(
                    self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer)](hidden)
                )
        return self.merger(hidden), *deepstack


class GrootVision(nn.Module):
    """The vision part: the robot's frame in; the image features, deepstack features and normalized state out."""

    def __init__(self, vision: nn.Module, images: _ImagesAndState, grid_thw: Tensor, use_bf16: bool):
        super().__init__()
        self.images = images
        self.vision = _VisionTower(vision, grid_thw)
        self.use_bf16 = use_bf16

    def forward(self, *frame: Tensor) -> tuple[Tensor, ...]:
        pixels, state = self.images(*frame)
        with torch.autocast(pixels.device.type, torch.bfloat16, enabled=self.use_bf16):
            features = self.vision(pixels)
        return *features, state


def _select_embodiment(module: nn.Module, embodiment_id: Tensor) -> None:
    """Keep only this robot's weights: the action head holds one set per embodiment it can be trained on."""
    for layer in module.modules():
        if isinstance(layer, CategorySpecificLinear):
            layer.W = nn.Parameter(layer.W[embodiment_id].detach(), requires_grad=False)
            layer.b = nn.Parameter(layer.b[embodiment_id].detach(), requires_grad=False)


def _prompt_mask(head: int, task_positions: Tensor, task_length: Tensor) -> Tensor:
    """Which tokens of the padded prompt are real: the `head` tokens before the task, then the task's own."""
    head_mask = torch.ones(head, dtype=torch.bool, device=task_positions.device)
    return torch.cat((head_mask, task_positions < task_length))[None]


def _place_image_features(hidden: Tensor, features: Tensor, image_mask: Tensor, rows: Tensor) -> Tensor:
    """`hidden` with the image features at the image tokens, in order, as Qwen3-VL's `masked_scatter` puts them.

    The image tokens are fixed, so a gather by the constant `rows` replaces `masked_scatter`, whose
    data-dependent size would make TensorRT allocate memory on every run.
    """
    return torch.where(image_mask[..., None], features[rows], hidden)


class GrootLanguageModel(nn.Module):
    """Some language model layers; the first group inserts the image features, the last runs the action head's encoders.

    The program takes only what its layers read: the hidden states, the deepstack features added in its layer
    range, in the first group the task's embeddings, and in the last the state and the task's length.

    The task's padding follows every real token, so the causal mask already keeps it out of the real tokens'
    attention; the action head's self-attention, which attends both ways, leaves it out with the prompt mask.
    """

    def __init__(
        self,
        text_model: nn.Module,
        action_head: nn.Module,
        prompt: dict[str, Any],
        head_embeds: Tensor,
        start: int,
        stop: int,
        use_bf16: bool,
        num_deepstack: int,
    ):
        super().__init__()
        if not 0 <= start < stop <= len(text_model.layers):
            raise ValueError("Invalid language model layer range.")
        self.start, self.last = start, stop == len(text_model.layers)
        # GR00T reads the last layer's output before Qwen3-VL adds its deepstack features.
        self.deepstack_layers = list(range(start, min(stop, num_deepstack, len(text_model.layers) - 1)))
        self.layers = nn.ModuleList(text_model.layers[start:stop])
        self.use_bf16 = use_bf16
        self.head = prompt["head"]
        image_mask = prompt["image_mask"]
        device = head_embeds.device
        self.register_buffer("image_mask", image_mask.to(device))
        self.register_buffer("image_rows", (image_mask.cumsum(-1) - 1).clamp(min=0).to(device))
        self.register_buffer("task_positions", prompt["task_positions"].to(device))
        length = image_mask.shape[1]
        # Half the float32 minimum stays finite in bfloat16, where the minimum itself becomes -inf.
        causal = torch.full((length, length), torch.finfo(torch.float32).min * 0.5).triu(1)
        self.register_buffer("causal_mask", causal[None, None].to(device))
        cos, sin = _text_rotary(text_model.rotary_emb, prompt["position_ids"])
        self.register_buffer("rotary_cos", cos.to(device))
        self.register_buffer("rotary_sin", sin.to(device))
        if start == 0:
            self.register_buffer("head_embeds", head_embeds)
        if self.last:
            self.vlln = action_head.vlln
            self.vl_self_attention = action_head.vl_self_attention
            self.state_encoder = action_head.state_encoder
            # The weights keep only this robot's embodiment, which is then the first.
            self.register_buffer("embodiment_id", torch.zeros_like(prompt["embodiment_id"]).to(device))
            _select_embodiment(self.state_encoder, prompt["embodiment_id"])

    def forward(self, hidden: Tensor, *inputs: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """`hidden` is the image features in the first group, else the previous group's hidden states.

        `inputs` is the deepstack features this layer range adds, in layer order; then the task's embeddings
        in the first group; then the state and the task's length in the last.
        """
        deepstack = dict(zip(self.deepstack_layers, inputs[: len(self.deepstack_layers)], strict=True))
        inputs = inputs[len(self.deepstack_layers) :]
        with torch.autocast(hidden.device.type, torch.bfloat16, enabled=self.use_bf16):
            if self.start == 0:
                task_embeddings, *rest = inputs
                inputs = tuple(rest)
                # The prompt's rows in the dtype of the policy's token embeddings, as its own lookup gives them.
                text = torch.cat((self.head_embeds, task_embeddings[None].to(self.head_embeds.dtype)), dim=1)
                hidden = _place_image_features(text, hidden.to(text.dtype), self.image_mask, self.image_rows)
            mask = self.causal_mask.to(hidden.dtype)
            rotary = (self.rotary_cos.to(hidden.dtype), self.rotary_sin.to(hidden.dtype))
            for index, layer in enumerate(self.layers, self.start):
                hidden = layer(
                    hidden,
                    position_embeddings=rotary,
                    attention_mask=mask,
                    past_key_values=None,
                )
                if index in deepstack:
                    features = deepstack[index].to(hidden.dtype)
                    hidden = hidden + _place_image_features(
                        torch.zeros_like(hidden), features, self.image_mask, self.image_rows
                    )
            if not self.last:
                return hidden
            state, task_length = inputs
            # GR00T reads the decoder output before Qwen's final norm.
            features = self.vlln(hidden).contiguous()
            prompt = _prompt_mask(self.head, self.task_positions, task_length)
            if not isinstance(self.vl_self_attention, nn.Identity):
                features = self.vl_self_attention(features, attention_mask=prompt)
            # As GR00T's own input step does, since autocast is off when use_bf16 is false.
            state = state.reshape(state.shape[0], 1, -1).to(self.state_encoder.layer1.W.dtype)
            return features, self.state_encoder(state, self.embodiment_id)


def _text_rotary(rotary: nn.Module, position_ids: Tensor) -> tuple[Tensor, Tensor]:
    """The text rope's cos and sin for the prompt's positions, from the exact float32 frequencies."""
    if getattr(rotary, "rope_type", "default") != "default":
        raise ValueError(f"GR00T export computes the default rope, not {rotary.rope_type!r}.")
    exact, _ = type(rotary).compute_default_rope_parameters(rotary.config, device=None)
    rotary.inv_freq = _exact_inv_freq(rotary.inv_freq, exact).to(rotary.inv_freq.device)
    probe = torch.zeros(1, position_ids.shape[-1], 1, device=rotary.inv_freq.device)
    with torch.no_grad():
        return rotary(probe, position_ids.to(probe.device))


def _exact_inv_freq(loaded: Tensor, exact: Tensor) -> Tensor:
    """The exact rope frequencies, after checking the policy's own copy rounds to them in bfloat16.

    A policy built in bfloat16 rounds the frequencies; a copy that does not round to the same bfloat16 values
    means another rope than the one written here.
    """
    loaded = loaded.detach().float().cpu()
    if loaded.shape != exact.shape or not torch.equal(loaded.to(torch.bfloat16), exact.to(torch.bfloat16)):
        raise ValueError("The policy's rope frequencies are not the ones its config gives.")
    if not torch.equal(loaded, exact):
        logger.info(
            "The policy's rope frequencies are rounded (by up to %.1e); the bundle uses the exact ones.",
            float((loaded - exact).abs().max()),
        )
    return exact


class _GivenKeysValues:
    """diffusers' default attention processor, `AttnProcessor2_0`, given the keys and values it would compute.

    `encoder_hidden_states` is the pair of what `to_k` and `to_v` make of the encoder's states, which
    `GrootKeysValues` computes once per chunk. The boolean mask becomes an additive float mask with the same
    meaning, which compilers fuse into their attention kernels more often.
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        key, value = encoder_hidden_states
        batch_size, _, inner_dim = key.shape
        query = attn.to_q(hidden_states)
        head_dim = inner_dim // attn.heads
        query, key, value = (
            x.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2) for x in (query, key, value)
        )
        bias = None
        if attention_mask is not None:
            bias = torch.zeros(attention_mask.shape, dtype=query.dtype, device=query.device)
            bias = bias.masked_fill(~attention_mask, torch.finfo(query.dtype).min)[:, None, None, :]
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=bias, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states.to(query.dtype)))
        return hidden_states / attn.rescale_output_factor


def _check_cross_attention(attn: nn.Module) -> None:
    """Refuse an attention `_GivenKeysValues` would compute differently from its own processor."""
    from diffusers.models.attention_processor import AttnProcessor2_0

    plain = all(
        getattr(attn, name) is None for name in ("spatial_norm", "group_norm", "norm_q", "norm_k")
    ) and not (attn.norm_cross or attn.residual_connection)
    if not plain or type(attn.processor) is not AttnProcessor2_0:
        raise ValueError(
            "Only diffusers' default attention processor, with no extra norms and no residual connection, "
            "can take its keys and values."
        )


class GrootKeysValues(nn.Module):
    """The keys and values the diffusion model's cross-attention blocks `cross` make of the vision-language features.

    They are the same in every denoising step, so the chain computes them once per chunk, here.
    """

    def __init__(self, action_head: nn.Module, cross: list[int], use_bf16: bool):
        super().__init__()
        blocks = action_head.model.transformer_blocks
        self.keys = nn.ModuleList(blocks[i].attn1.to_k for i in cross)
        self.values = nn.ModuleList(blocks[i].attn1.to_v for i in cross)
        self.use_bf16 = use_bf16

    def forward(self, vl_embeds: Tensor) -> tuple[Tensor, ...]:
        with torch.autocast(vl_embeds.device.type, torch.bfloat16, enabled=self.use_bf16):
            vl_embeds = vl_embeds.contiguous()
            return tuple(
                x for k, v in zip(self.keys, self.values, strict=True) for x in (k(vl_embeds), v(vl_embeds))
            )


class GrootStep(nn.Module):
    """Some diffusion blocks of one denoising step.

    The first group encodes the noisy actions, with the step's time encodings taken as inputs from the
    bundle's constants. The last returns the next actions, with the Euler update inside, so the rollout does
    no arithmetic between programs. The cross-attention blocks take the keys and values `GrootKeysValues`
    computed for them.
    """

    def __init__(
        self,
        action_head: nn.Module,
        prompt: dict[str, Any],
        start: int,
        stop: int,
        cross: list[int],
        use_bf16: bool,
        dt: float,
        dtype: torch.dtype,
        actions_dtype: torch.dtype,
    ):
        super().__init__()
        model = action_head.model
        if not 0 <= start < stop <= len(model.transformer_blocks):
            raise ValueError("Invalid diffusion block range.")
        self.start, self.stop, self.last = start, stop, stop == len(model.transformer_blocks)
        self.dt = dt
        # Only this group's blocks are loaded; the others are None.
        self.model = model
        self.cross = [i for i in cross if start <= i < stop]
        for i in self.cross:
            attn = model.transformer_blocks[i].attn1
            _check_cross_attention(attn)
            attn.set_processor(_GivenKeysValues())
            # GrootKeysValues holds these projections; the step only reads what they compute.
            del attn.to_k, attn.to_v
        self.use_bf16 = use_bf16
        self.dtype, self.actions_dtype = dtype, actions_dtype
        self.horizon = action_head.action_horizon
        self.head = prompt["head"]
        device = prompt["embodiment_id"].device
        self.register_buffer("image_mask", prompt["image_mask"].to(device))
        self.register_buffer("task_positions", prompt["task_positions"].to(device))
        # The weights keep only this robot's embodiment, which is then the first.
        self.register_buffer("embodiment_id", torch.zeros_like(prompt["embodiment_id"]))
        if start == 0:
            self.action_encoder = action_head.action_encoder
            self.position_embedding = (
                action_head.position_embedding if action_head.config.add_pos_embed else None
            )
            _select_embodiment(self.action_encoder, prompt["embodiment_id"])
        if self.last:
            self.action_decoder = action_head.action_decoder
            _select_embodiment(self.action_decoder, prompt["embodiment_id"])

    def time_tables(self, timesteps: list[int]) -> dict[str, np.ndarray]:
        """For each denoising step, the action encoder's sinusoidal time encoding and the blocks' time embedding."""
        tables = {}
        device = self.embodiment_id.device
        for index, bucket in enumerate(timesteps):
            time = torch.tensor([bucket], device=device)
            with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=self.use_bf16):
                encoding = self.action_encoder.pos_encoding(time[:, None])
                temb = self.model.timestep_encoder(time)
            tables[f"time_encoding_{index}"] = encoding.float().cpu().numpy()
            tables[f"temb_{index}"] = temb.float().cpu().numpy()
        return tables

    def forward(self, *inputs: Tensor) -> Tensor:
        """`inputs` is the key and value of each cross-attention block, then by group, then the task's length.

        - The first group, also when it is the only one: the state features, the noisy actions, the step's
          time encoding and its time embedding.
        - A later group: the previous group's hidden states and the time embedding.
        - The last of several groups: those, then the noisy actions.
        """
        keys_values, inputs = inputs[: 2 * len(self.cross)], inputs[2 * len(self.cross) :]
        given = {block: keys_values[2 * n : 2 * n + 2] for n, block in enumerate(self.cross)}
        *rest, task_length = inputs
        # The policy attends to every token of its unpadded prompt; the padding stays out of every cross-attention.
        prompt = _prompt_mask(self.head, self.task_positions, task_length)
        with torch.autocast(task_length.device.type, torch.bfloat16, enabled=self.use_bf16):
            if self.start == 0:
                state_features, x_t, time_encoding, temb = rest
                hidden = self.action_encoder.encode(
                    x_t[None].to(self.actions_dtype), time_encoding, self.embodiment_id
                )
                if self.position_embedding is not None:
                    positions = torch.arange(hidden.shape[1], device=hidden.device)
                    hidden = hidden + self.position_embedding(positions)[None]
                hidden = torch.cat((state_features, hidden), dim=1)
            elif self.last:
                hidden, temb, x_t = rest
            else:
                hidden, temb = rest
            hidden, temb = hidden.contiguous(), temb.to(self.dtype)
            masks = self.model.encoder_masks(self.image_mask, prompt)
            for index in range(self.start, self.stop):
                hidden = self.model.run_block(index, hidden, given.get(index), temb, masks[index])
            if not self.last:
                return hidden
            prediction = self.action_decoder(self.model.output(hidden, temb), self.embodiment_id)
            # As the policy, hold the noisy actions in the dtype of its vision-language features.
            update = self.dt * prediction[0, -self.horizon :]
            return (x_t.to(self.actions_dtype) + update.to(self.actions_dtype)).float()


def _rot6d_to_matrix(rot6d: Tensor) -> Tensor:
    """GR00T's `rot6d_to_matrix` on a batch, in float32 where GR00T computes in float64, which TensorRT lacks."""
    row1, row2 = rot6d.unflatten(-1, (2, 3)).unbind(-2)
    row1 = row1 / row1.norm(dim=-1, keepdim=True)
    row2 = row2 - (row1 * row2).sum(-1, keepdim=True) * row1
    row2 = row2 / row2.norm(dim=-1, keepdim=True)
    return torch.stack((row1, row2, torch.linalg.cross(row1, row2)), dim=-2)


def _relative_eef_to_absolute(action: Tensor, reference: Tensor) -> Tensor:
    """GR00T's `relative_eef_to_absolute`: xyz+rot6d poses relative to `reference`, made absolute."""
    turn = _rot6d_to_matrix(reference[:, None, 3:])
    rotation = turn @ _rot6d_to_matrix(action[..., 3:])
    xyz = (turn @ action[..., :3, None])[..., 0] + reference[:, None, :3]
    return torch.cat((xyz, rotation[..., :2, :].flatten(-2)), dim=-1)


def _unnormalize(normalized: Tensor | float, low: Tensor, high: Tensor) -> Tensor:
    """GR00T's min-max unnormalization of actions in [-1, 1]."""
    return (normalized + 1.0) * 0.5 * (high - low) + low


class N17ActionDecode(nn.Module):
    """`GrootN17ActionDecodeStep`, GR00T N1.7's decoding with its embodiment statistics, as tensor operations.

    The step decodes in NumPy after a copy to the host, which no program can hold, and reads the reference
    state of relative actions from its packing step's cache; here the robot state is an input, split into
    the packing step's state groups. Each action group is unnormalized with its statistics, relative groups
    are then added to, or composed with, their state group, and the groups are joined and cut to the
    robot's actions, as the step does.
    """

    def __init__(self, decode: GrootN17ActionDecodeStep, pack: GrootN17PackInputsStep, horizon: int):
        super().__init__()
        if not decode.raw_stats or not decode.modality_config:
            raise ValueError("GrootN17ActionDecodeStep has no statistics or modality config to decode with.")
        raw_stats, action = decode.raw_stats, decode.modality_config["action"]
        keys, configs = action.get("modality_keys", []), action.get("action_configs", [])
        self.horizon = _n1_7_decode_valid_horizon(action, np.zeros((1, horizon, 1))) or horizon
        self.env_action_dim = decode.env_action_dim
        self.groups: list[tuple[str, int, int]] = []
        self.relative: list[tuple[str, str, str]] = []
        start = 0
        for index, key in enumerate(keys):
            entry = raw_stats.get("action", {}).get(key) if isinstance(key, str) else None
            dim = stat_dim_from_entry(entry) if isinstance(entry, dict) else 0
            if dim <= 0:
                continue
            config = configs[index] if index < len(configs) and isinstance(configs[index], dict) else {}
            low, high = _n1_7_decode_stats_for_action(
                raw_stats,
                key,
                config,
                use_relative_action=decode.use_relative_action,
                use_percentiles=decode.use_percentiles,
            )
            # Per-step statistics hold a row per chunk step, and a chunk starts at step 0.
            if low.ndim == 2:
                low, high = low[: self.horizon], high[: self.horizon]
            # Named by index: a modality key may hold a dot, which a buffer name cannot.
            self.register_buffer(f"low_{len(self.groups)}", torch.from_numpy(np.asarray(low, np.float32)))
            self.register_buffer(f"high_{len(self.groups)}", torch.from_numpy(np.asarray(high, np.float32)))
            self.groups.append((key, start, dim))
            start += dim
            if decode.use_relative_action and config_value(config.get("rep")) == "relative":
                kind = config_value(config.get("type")), config_value(config.get("format"))
                if kind[0] != "non_eef" and kind != ("eef", "xyz+rot6d"):
                    raise ValueError(f"Unsupported relative N1.7 action config for {key!r}: {config}")
                self.relative.append((key, config.get("state_key") or key, kind[0]))
        if not self.groups:
            raise ValueError("GrootN17ActionDecodeStep has no action group with statistics to decode.")
        self.state_groups: dict[str, tuple[int, int]] = {}
        state_start = 0
        for key in (pack.modality_config or {}).get("state", {}).get("modality_keys", []):
            entry = (pack.raw_stats or {}).get("state", {}).get(key) if isinstance(key, str) else None
            # Sized as the packing step sizes them.
            dim = stat_dim_from_entry(entry) if isinstance(entry, dict) else 0
            if dim > 0:
                self.state_groups[key] = (state_start, dim)
                state_start += dim
        missing = sorted({state_key for _, state_key, _ in self.relative} - set(self.state_groups))
        if missing:
            raise ValueError(
                f"The relative N1.7 actions are about the state groups {missing}, which are not packed."
            )
        self.gripper = None
        if decode.action_decode_transform == GROOT_ACTION_DECODE_TRANSFORM_LIBERO:
            gripper = [start for key, start, dim in self.groups if key == "gripper" and dim == 1]
            if not gripper or (self.env_action_dim and gripper[0] >= self.env_action_dim):
                raise ValueError("The LIBERO action decode transform needs a scalar gripper action group.")
            self.gripper = gripper[0]
        elif decode.action_decode_transform is not None:
            raise ValueError(f"Unsupported N1.7 action decode transform {decode.action_decode_transform!r}.")

    def action_scale(self) -> np.ndarray:
        """Robot units per normalized unit of each decoded action, at the chunk step where it is largest.

        A relative end-effector pose turns its xyz by the state's rotation, which mixes the axes, so they share
        the largest of their scales; its rotation is made of unit vectors, so its scale is 1.
        """
        poses = {key for key, _, kind in self.relative if kind == "eef"}
        scales = []
        for index, (key, _, _) in enumerate(self.groups):
            low, high = getattr(self, f"low_{index}"), getattr(self, f"high_{index}")
            scale = measure_action_scale(partial(_unnormalize, low=low, high=high))
            scale = scale.reshape(-1, scale.shape[-1]).max(0)
            if key in poses:
                scale[:3], scale[3:] = scale[:3].max(), 1
            scales.append(scale)
        scale = np.concatenate(scales)
        if self.env_action_dim:
            scale = scale[: self.env_action_dim]
        if self.gripper is not None:
            # The LIBERO gripper plays -1 or 1, and 0 exactly at its threshold.
            scale[self.gripper] = 1
        return scale

    def forward(self, actions: Tensor, state: Tensor | None) -> Tensor:
        """`actions` (batch, steps, width) to robot actions, about `state` (batch, state width)."""
        actions = actions[:, : self.horizon].float()
        decoded = {}
        for index, (key, start, dim) in enumerate(self.groups):
            low, high = getattr(self, f"low_{index}"), getattr(self, f"high_{index}")
            decoded[key] = _unnormalize(actions[..., start : start + dim].clamp(-1.0, 1.0), low, high)
        for key, state_key, kind in self.relative:
            start, dim = self.state_groups[state_key]
            reference = cast(Tensor, state)[:, start : start + dim].float()
            if kind == "non_eef":
                decoded[key] = decoded[key] + reference[:, None]
            else:
                decoded[key] = _relative_eef_to_absolute(decoded[key], reference)
        result = torch.cat([decoded[key] for key, _, _ in self.groups], dim=-1)
        if self.env_action_dim:
            result = result[..., : self.env_action_dim]
        if self.gripper is not None:
            gripper = result[..., self.gripper : self.gripper + 1]
            result = torch.cat(
                (
                    result[..., : self.gripper],
                    -torch.sign(2.0 * gripper - 1.0),
                    result[..., self.gripper + 1 :],
                ),
                dim=-1,
            )
        return result


class _ChunkActions(nn.Module):
    """The end of the chain: the denoised chunk in, the actions to play out, in robot units.

    It runs the saved postprocessor. A policy trained on actions relative to the state turns them back
    around the robot's state, which the program then takes as an input too.
    """

    def __init__(self, execution_steps: int, action_dim: int, postprocessor: PolicyProcessorPipeline):
        super().__init__()
        self.execution_steps, self.action_dim = execution_steps, action_dim
        self.postprocessor, self.mask = split_absolute_step(postprocessor, action_dim)

    @property
    def reads_state(self) -> bool:
        return self.mask is not None

    def forward(self, actions: Tensor, state: Tensor | None = None) -> Tensor:
        actions = self.postprocessor(actions[None, : self.execution_steps, : self.action_dim])
        if self.mask is not None:
            actions = to_absolute_actions(actions, cast(Tensor, state)[None], self.mask)
        return actions[0]


class _N17ChunkActions(nn.Module):
    """The end of the chain for a checkpoint that decodes with `GrootN17ActionDecodeStep`."""

    def __init__(self, execution_steps: int, action_dim: int, decode: N17ActionDecode):
        super().__init__()
        self.execution_steps, self.action_dim = execution_steps, action_dim
        self.decode = decode

    @property
    def reads_state(self) -> bool:
        return bool(self.decode.relative)

    def forward(self, actions: Tensor, state: Tensor | None = None) -> Tensor:
        state = state[None] if state is not None else None
        return self.decode(actions[None, : self.execution_steps, : self.action_dim], state)[0]


def _chunk_actions(
    execution_steps: int,
    action_dim: int,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
) -> _ChunkActions | _N17ChunkActions:
    """`_N17ChunkActions` for a checkpoint whose postprocessor decodes with N1.7's own step, else `_ChunkActions`."""
    decode = next((s for s in postprocessor.steps if isinstance(s, GrootN17ActionDecodeStep)), None)
    if decode is None:
        return _ChunkActions(execution_steps, action_dim, postprocessor)
    if not all(isinstance(s, DeviceProcessorStep) for s in postprocessor.steps if s is not decode):
        raise ValueError(
            "GrootN17ActionDecodeStep must be the postprocessor's only step besides the device's."
        )
    decode_module = N17ActionDecode(decode, _step(preprocessor, GrootN17PackInputsStep), execution_steps)
    return _N17ChunkActions(execution_steps, action_dim, decode_module)


def _prompt_ops(formalize: bool) -> list[list[str]]:
    """GR00T's processor text as prompt ops: `prepare_n1_7_language_batch`, then the chat template's tail."""
    ops = [["default", DEFAULT_TASK]]
    if formalize:
        # The encoding step fills a task the cleaning emptied with the default again, as it is.
        ops += [["lower"], ["keep", "alnum", "space", "_"], ["default", DEFAULT_TASK]]
    return [*ops, ["template", TASK_TEMPLATE]]


def _step(pipeline: PolicyProcessorPipeline, step_type: type) -> Any:
    found = next((s for s in pipeline.steps if isinstance(s, step_type)), None)
    if found is None:
        raise ValueError(f"The saved processor has no {step_type.__name__}, so it is not a GR00T N1.7 one.")
    return found


def _free_memory() -> None:
    """Return freed GPU memory to the next part."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
