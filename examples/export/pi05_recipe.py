#!/usr/bin/env python

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

"""The pi0.5 export recipe: what both backend scripts compile, and the folder they write.

As for SmolVLA, the compiled module is one action chunk of LeRobot's own code: the checkpoint's saved
preprocessor, the policy's `predict_action_chunk` and the saved postprocessor, with the starting noise
as the program's last input. One thing differs:

- The state. pi0.5 reads the normalized state as bin numbers in the prompt. The program digitizes it
  and assembles the prompt from token pieces prepared at export time. No tokenizer runs at inference.

The inputs are the robot's raw uint8 HWC cameras, float32 state and unbatched noise. Image conversion,
resize and padding, normalization and action unnormalization all run inside the program.

The checkpoint names the robot's cameras, so they come from its config, as for ACT, except the empty
cameras pi0.5 fills in itself. The folder's config names only the cameras the program takes. The
checkpoint does not record the task, so the export takes it on the command line.

The program also keeps only the rows of the 257,152-row vocabulary that the task's prompt can hold:
the task's own words, the fixed words around it, and the state's bins written as numbers. That saves
about 1 GiB, and ties the folder to the task, so `export.json` marks it `task_fixed`.

A split export (`PI05Export(args, split=True)` and `parts`) never loads the whole policy: each part
reads only its own weights from the checkpoint, so a device with less memory than the policy can
export it one part at a time.
"""

import argparse
import gc
import json
import time
from collections.abc import Iterator
from copy import copy
from pathlib import Path

import numpy as np
import torch
from act_recipe import (
    TEST_CASE,
    add_backend_args,
    gpu_processors,
    make_output_dir,
    random_robot_frame,
    to_policy_input,
)
from safetensors.numpy import save_file
from smolvla_recipe import NOISE
from torch import Tensor, nn
from transformers import DynamicCache
from transformers.utils import cached_file

from lerobot.configs import PreTrainedConfig
from lerobot.policies.common.openpi_checkpoint import load_checkpoint_modules
from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch, get_gemma_config
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, TokenizerProcessorStep
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

LANGUAGE_MODEL = "model.paligemma_with_expert.paligemma.model.language_model"
# The vision encoder, its projector and the token embeddings; the language model's lm_head is never needed.
EMBED_MODULES = (
    "model.paligemma_with_expert.paligemma.model.vision_tower",
    "model.paligemma_with_expert.paligemma.model.multi_modal_projector",
    f"{LANGUAGE_MODEL}.embed_tokens",
)
# The action expert without its lm_head, and the layers around it that `denoise_step` runs.
STEP_MODULES = (
    "model.paligemma_with_expert.gemma_expert.model",
    "model.action_in_proj",
    "model.action_out_proj",
    "model.time_mlp_in",
    "model.time_mlp_out",
)


def free_memory() -> None:
    """Return freed GPU memory, so the next part or the TensorRT build can use it."""
    gc.collect()
    torch.cuda.empty_cache()


class StatePrompt(nn.Module):
    """The fixed task and normalized state as a padded Gemma prompt."""

    def __init__(self, tokenizer: TokenizerProcessorStep, task: str, state_dim: int):
        super().__init__()
        tok = tokenizer.input_tokenizer
        clean = task.strip().replace("_", " ").replace("\n", " ")
        prefix = tok.encode(f"Task: {clean}, State:")
        pieces = [tok.encode(f" {value}", add_special_tokens=False) for value in range(-1, 256)]
        suffix = tok.encode(";\nAction: ", add_special_tokens=False)
        width = max(map(len, pieces))
        if (
            tokenizer.padding_side != "right"
            or len(prefix) + state_dim * width + len(suffix) > tokenizer.max_length
        ):
            raise ValueError("The state prompt must fit in a right-padded tokenizer window.")
        self.pad_id = tok.pad_token_id
        self.register_buffer("edges", -1 + torch.arange(256, dtype=torch.float32) / 128)
        self.register_buffer("prefix", torch.tensor(prefix))
        self.register_buffer("suffix", torch.tensor(suffix))
        self.register_buffer("pieces", torch.tensor([p + [self.pad_id] * (width - len(p)) for p in pieces]))
        self.register_buffer("lengths", torch.tensor(list(map(len, pieces))))
        self.register_buffer("piece_positions", torch.arange(width))
        self.register_buffer("positions", torch.arange(tokenizer.max_length))

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        bins = (state.flatten()[..., None] >= self.edges).sum(-1) - 1
        pieces = self.pieces[bins + 1].flatten()
        valid = (self.piece_positions < self.lengths[bins + 1, None]).flatten()
        tokens = torch.cat((self.prefix, pieces, self.suffix))
        valid = torch.cat(
            (
                torch.ones_like(self.prefix, dtype=torch.bool),
                valid,
                torch.ones_like(self.suffix, dtype=torch.bool),
            )
        )
        # Each valid token owns one output position; unused piece slots own none.
        compact = (valid.cumsum(0)[:, None] - 1 == self.positions) & valid[:, None]
        ids = (tokens[:, None] * compact).sum(0)
        mask = self.positions < valid.sum()
        return torch.where(mask, ids, self.pad_id)[None], mask[None]


class PI05Observation(nn.Module):
    """Convert and normalize the raw frame, then build its state-dependent prompt."""

    def __init__(self, frame_names: list[str], preprocessor: PolicyProcessorPipeline, prompt: StatePrompt):
        super().__init__()
        self.frame_names = frame_names
        self.preprocessor = preprocessor
        self.prompt = prompt

    def forward(self, *frame: Tensor) -> dict[str, Tensor]:
        batch = self.preprocessor(
            {name: to_policy_input(name, x) for name, x in zip(self.frame_names, frame, strict=True)}
        )
        batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK] = self.prompt(batch[OBS_STATE])
        return batch


class PI05Chunk(nn.Module):
    """One action chunk: the raw frame and noise in, unbatched actions in robot units out."""

    def __init__(
        self,
        policy: PI05Policy,
        preprocessor: PI05Observation,
        postprocessor: PolicyProcessorPipeline,
        input_names: list[str],
    ):
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.input_names = input_names

    def forward(self, *inputs: Tensor) -> Tensor:
        *observation, noise = inputs
        batch = self.preprocessor(*observation)
        actions = self.policy.predict_action_chunk(batch, noise=noise[None])
        return self.postprocessor(actions[:, : self.policy.config.n_action_steps])[0]


class PI05Prefix(nn.Module):
    """What `sample_actions` runs once per chunk: the cameras and the prompt in, the KV cache out."""

    def __init__(self, policy: PI05Policy, preprocessor: PI05Observation, input_names: list[str]):
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.input_names = input_names

    def forward(self, *observation: Tensor) -> tuple[Tensor, ...]:
        batch = self.preprocessor(*observation)
        images, img_masks = self.policy._preprocess_images(batch)
        states, state_masks = self.policy._prepare_memory_states(batch)
        model = self.policy.model
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            states,
            state_masks,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
        model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        # A program passes only tensors, so the cache leaves as each layer's keys and values.
        return prefix_pad_masks, *(tensor for keys, values, _ in past_key_values for tensor in (keys, values))


class PI05Step(nn.Module):
    """LeRobot's `denoise_step`: the prefix's mask and KV cache, the noisy actions and the time in, the velocity out."""

    def __init__(self, model: PI05Pytorch):
        super().__init__()
        self.model = model

    def forward(self, prefix_pad_masks: Tensor, *inputs: Tensor) -> Tensor:
        *cache, x_t, timestep = inputs
        past_key_values = DynamicCache(
            tuple((keys, values, None) for keys, values in zip(cache[::2], cache[1::2], strict=True))
        )
        return self.model.denoise_step(prefix_pad_masks, past_key_values, x_t[None], timestep)[0]


class PI05Actions(nn.Module):
    """The end of the chunk: the denoised actions in, the actions to play out."""

    def __init__(self, policy: PI05Policy, postprocessor: PolicyProcessorPipeline):
        super().__init__()
        self.policy = policy
        self.postprocessor = postprocessor

    def forward(self, x_0: Tensor) -> Tensor:
        action_dim = self.policy.config.output_features[ACTION].shape[0]
        return self.postprocessor(x_0[None, : self.policy.config.n_action_steps, :action_dim])[0]


class PI05Embed(nn.Module):
    """The start of `PI05Prefix`: the cameras and the prompt in, the prefix embeddings and their mask out."""

    def __init__(self, policy: PI05Policy, preprocessor: PI05Observation, input_names: list[str]):
        super().__init__()
        self.policy = policy
        self.preprocessor = preprocessor
        self.input_names = input_names

    def forward(self, *observation: Tensor) -> tuple[Tensor, Tensor]:
        batch = self.preprocessor(*observation)
        images, img_masks = self.policy._preprocess_images(batch)
        prefix_embs, prefix_pad_masks, _ = self.policy.model.embed_prefix(
            images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        )
        return prefix_embs, prefix_pad_masks


class PI05LanguageModelLayers(nn.Module):
    """Some of `PI05Prefix`'s language model layers: the hidden states in, the next hidden states and their KV cache out.

    pi0.5's prefix attends both ways inside the prompt, so the mask and positions come from the padding mask alone.
    """

    def __init__(self, model: PI05Pytorch, layers: range):
        super().__init__()
        language_model = model.paligemma_with_expert.paligemma.model.language_model
        language_model.config._attn_implementation = "eager"
        self.layers = nn.ModuleList(language_model.layers[i] for i in layers)
        self.rotary_emb = language_model.rotary_emb

    def forward(self, hidden_states: Tensor, prefix_pad_masks: Tensor) -> tuple[Tensor, ...]:
        att_masks = torch.zeros_like(prefix_pad_masks)
        attention_mask = prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, att_masks))
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        if self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
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
        return hidden_states, *(tensor for layer in own for tensor in (layer.keys, layer.values))


class CompactEmbedding(nn.Module):
    """An embedding that keeps only the rows of `token_ids`, and finds them with a constant table."""

    def __init__(self, embedding: nn.Embedding, token_ids: Tensor):
        super().__init__()
        token_ids = token_ids.to(embedding.weight.device)
        rows = torch.zeros(embedding.num_embeddings, dtype=torch.long, device=token_ids.device)
        rows[token_ids] = torch.arange(len(token_ids), device=token_ids.device)
        self.register_buffer("rows", rows)
        embedding.weight = nn.Parameter(embedding.weight.detach()[token_ids], requires_grad=False)
        embedding.num_embeddings = len(token_ids)
        self.embedding = embedding

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.embedding(self.rows[token_ids])


class PI05Export:
    """Loads a trained pi0.5 checkpoint, and writes the exported folder around a compiled program.

    With `split`, it loads only the embedding weights, and `parts` loads every later part on its own.
    """

    def __init__(self, args: argparse.Namespace, split: bool = False):
        self.output_dir = make_output_dir(args.output_dir, args.job_name)
        self.policy_path = args.policy_path
        self.backend, self.tolerance = args.backend, args.tolerance
        self.task = args.task
        if split:
            self.config = PreTrainedConfig.from_pretrained(self.policy_path)
            self.model_file = cached_file(self.policy_path, "model.safetensors")
            self.policy = self.load_part(*EMBED_MODULES)
        else:
            self.policy = PI05Policy.from_pretrained(self.policy_path).to("cuda").eval()
        self.config = self.policy.config
        preprocessor, postprocessor = self.processors()
        tokenizer = next(s for s in preprocessor.steps if isinstance(s, TokenizerProcessorStep))
        tensor_steps = PolicyProcessorPipeline(
            steps=[
                s
                for s in preprocessor.steps
                if not isinstance(s, Pi05PrepareStateTokenizerProcessorStep | TokenizerProcessorStep)
            ]
        )
        if self.config.use_visual_memory or self.config.use_proprioceptive_memory:
            raise ValueError("Raw-frame export requires a policy without observation memory.")

        config = self.policy.config
        # pi0.5 fills in its empty cameras itself, so the robot does not send them.
        empty_cameras = {f"{OBS_IMAGES}.empty_camera_{i}" for i in range(config.empty_cameras)}
        available = set(config.image_features) - empty_cameras
        cameras = available if args.cameras is None else set(args.cameras)
        if not cameras or not cameras <= available:
            raise ValueError(f"Choose cameras from {sorted(available)}.")
        self.frame = {
            name: value
            for name, value in random_robot_frame(self.policy).items()
            if name not in config.image_features or name in cameras
        }
        self.input_features = {name: config.input_features[name] for name in self.frame}
        prompt = StatePrompt(tokenizer, self.task, self.frame[OBS_STATE].size).cuda()
        names = [*self.frame, NOISE]
        self.noise = torch.randn(config.chunk_size, config.max_action_dim, device="cuda")
        # A split export cannot run the whole policy, so `parts` chains its parts' PyTorch outputs instead.
        self.expected_actions = None if split else self.rollout_actions()
        language_model = self.policy.model.paligemma_with_expert.paligemma.model.language_model
        token_ids = torch.cat((prompt.prefix, prompt.pieces.flatten(), prompt.suffix)).unique()
        language_model.embed_tokens = CompactEmbedding(language_model.embed_tokens, token_ids)
        self.inputs = (*(torch.from_numpy(x).cuda() for x in self.frame.values()), self.noise)
        observation = PI05Observation(list(self.frame), tensor_steps, prompt)
        self.module = PI05Chunk(self.policy, observation, postprocessor, names)
        self.input_names = names
        self.start = time.perf_counter()

    def release_policy(self) -> None:
        """Free the policy for the TensorRT step."""
        del self.policy, self.module
        free_memory()

    def load_part(self, *modules: str) -> PI05Policy:
        """The policy with only `modules`' weights read from the checkpoint; every other module is None."""
        return load_checkpoint_modules(PI05Policy, self.model_file, self.config, list(modules))

    def parts(self, layers_per_program: int) -> Iterator[tuple[str, nn.Module, tuple, list[str], list[str]]]:
        """Yield the chunk's programs one at a time, each with only its own weights in memory.

        Each program comes with its example inputs, input names and output names: the embeddings, the
        language model in groups of `layers_per_program` layers, one denoising step and the actions.
        Each part runs once in PyTorch to make the next part's inputs, and the test case's actions are
        those parts chained. Drop each module before taking the next, so only one part is ever loaded.
        """
        observation, names = self.inputs[:-1], self.input_names[:-1]
        postprocessor = self.module.postprocessor
        embed = PI05Embed(self.policy, self.module.preprocessor, names)
        self.release_policy()
        with torch.no_grad():
            hidden, prefix_pad_masks = embed(*observation)
        yield "embed", embed, observation, names, ["prefix_embs", "prefix_pad_masks"]
        del embed
        hidden_name, cache = "prefix_embs", {}
        depth = get_gemma_config(self.config.paligemma_variant).depth
        for group, first in enumerate(range(0, depth, layers_per_program)):
            layers = range(first, min(first + layers_per_program, depth))
            free_memory()
            layer_group = PI05LanguageModelLayers(
                self.load_part(*(f"{LANGUAGE_MODEL}.layers.{i}" for i in layers)).model, layers
            )
            inputs = (hidden, prefix_pad_masks)
            with torch.no_grad():
                hidden, *keys_values = layer_group(*inputs)
            outputs = [f"hidden_{group}", *(f"past_{kind}_{i}" for i in layers for kind in ("key", "value"))]
            cache.update(zip(outputs[1:], keys_values, strict=True))
            yield f"language_model_{group}", layer_group, inputs, [hidden_name, "prefix_pad_masks"], outputs
            del layer_group
            hidden_name = outputs[0]
        free_memory()
        step = PI05Step(self.load_part(*STEP_MODULES).model)
        cache_inputs = (prefix_pad_masks, *cache.values())
        x_t, dt = self.noise, -1.0 / self.config.num_inference_steps
        with torch.no_grad():
            for i in range(self.config.num_inference_steps):
                x_t = x_t + dt * step(*cache_inputs, x_t, torch.tensor([1.0 + i * dt], device="cuda"))
        step_inputs = (*cache_inputs, self.noise, torch.ones(1, device="cuda"))
        yield "step", step, step_inputs, ["prefix_pad_masks", *cache, "x_t", "timestep"], ["v_t"]
        del step
        free_memory()
        actions = PI05Actions(self.load_part(), postprocessor)
        with torch.no_grad():
            self.expected_actions = actions(x_t).cpu().numpy()
        yield "actions", actions, (self.noise,), ["x_t"], [ACTION]

    def denoising_programs(self) -> dict[str, tuple[nn.Module, tuple[Tensor, ...], list[str], list[str]]]:
        """The chunk as three programs, each with its example inputs, input names and output names.

        The prefix runs once per chunk, the denoising step once per Euler step and the actions at the end.
        No engine holds the whole 10-step loop.
        """
        config = self.policy.config
        if config.use_visual_memory or config.use_proprioceptive_memory:
            raise SystemExit("--step_engine does not support a policy with visual or proprioceptive memory.")
        observation, observation_names = self.inputs[:-1], self.input_names[:-1]
        prefix = PI05Prefix(self.policy, self.module.preprocessor, observation_names)
        with torch.no_grad():
            cache = prefix(*observation)
        layers = range((len(cache) - 1) // 2)
        cache_names = ["prefix_pad_masks", *(f"past_{kind}_{i}" for i in layers for kind in ("key", "value"))]
        step_inputs = (*cache, self.noise, torch.ones(1, device="cuda"))
        return {
            "prefix": (prefix, observation, observation_names, cache_names),
            "step": (PI05Step(self.policy.model), step_inputs, [*cache_names, "x_t", "timestep"], ["v_t"]),
            "actions": (
                PI05Actions(self.policy, self.module.postprocessor),
                (self.noise,),
                ["x_t"],
                [ACTION],
            ),
        }

    def write(self, program: Path | dict) -> None:
        """Save the raw test case, the policy config and `export.json` beside the program.

        `program` is the program's path or, for a chunk split into engines, each engine's file, inputs and outputs.
        """
        case = {**self.frame, NOISE: self.noise.cpu().numpy(), "expected_actions": self.expected_actions}
        save_file(case, self.output_dir / TEST_CASE)
        config = copy(self.config)
        config.input_features = self.input_features
        config.save_pretrained(self.output_dir)
        if isinstance(program, dict):
            files = {"programs": program, "num_steps": self.config.num_inference_steps}
        else:
            files = {"file": program.name}
        info = {
            "backend": self.backend,
            **files,
            "inputs": self.input_names,
            "raw_frame": True,
            "task": self.task,
            # The program keeps only the vocabulary rows this task's prompt can hold.
            "task_fixed": True,
            "noise_shape": list(self.noise.shape),
            "output": ACTION,
            "test_case": TEST_CASE,
            "tolerance": self.tolerance,
        }
        (self.output_dir / "export.json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"Wrote {self.output_dir} in {time.perf_counter() - self.start:.0f} s")

    def processors(self) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
        """The checkpoint's processors on the GPU, with the tokenizer padding every prompt to one width."""
        preprocessor, postprocessor = gpu_processors(self.policy.config, self.policy_path)
        # The prompt carries the state, so its length can change every step; the program takes one shape.
        next(s for s in preprocessor.steps if isinstance(s, TokenizerProcessorStep)).padding = "max_length"
        return preprocessor, postprocessor

    def rollout_actions(self) -> np.ndarray:
        """The chunk the PyTorch policy computes for the test frame, task and noise, as lerobot-rollout runs it."""
        preprocessor, postprocessor = self.processors()
        self.policy.reset()
        observation = prepare_observation_for_inference(dict(self.frame), torch.device("cuda"), self.task)
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(preprocessor(observation), noise=self.noise[None])
            actions = postprocessor(actions[:, : self.policy.config.n_action_steps])
        return actions[0].cpu().numpy()


def parse_args(description: str, backend: str) -> argparse.Namespace:
    """The command line every pi0.5 export script shares."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy.path", dest="policy_path", required=True, help="Trained pi0.5 checkpoint.")
    parser.add_argument("--task", required=True, help="The task the policy was trained on, as a sentence.")
    parser.add_argument(
        "--cameras", nargs="+", help="Input camera keys; omitted slots use the policy's padding."
    )
    parser.add_argument(
        "--output_dir", type=Path, help="Folder to write. Default: a new one under outputs/export."
    )
    parser.add_argument("--job_name", default=f"pi05_{backend}", help="Names the default output folder.")
    # As for SmolVLA, the backbone runs in bfloat16 and TensorRT reorders its math, so a pi0.5 engine's actions
    # drift from PyTorch's by up to about 4 on this arm, against 15 or more for wrong noise. Swapped cameras often
    # stay under 5, so the check catches gross mistakes, not small ones.
    parser.add_argument(
        "--tolerance", type=float, default=5.0, help="Largest allowed action error, robot units."
    )
    add_backend_args(parser, backend)
    if backend == "onnx_tensorrt":
        parser.add_argument(
            "--step_engine",
            action="store_true",
            help="Export the prefix, one denoising step and the actions as three engines.",
        )
    if backend == "executorch_tensorrt":
        parser.add_argument(
            "--step_engine",
            action="store_true",
            help="Export the chunk as separate programs, each loading only its own weights from the checkpoint.",
        )
        parser.add_argument(
            "--layers_per_program",
            type=int,
            default=6,
            help="Language model layers in each program with --step_engine; fewer need less memory.",
        )
        parser.add_argument(
            "--workspace_gib", type=float, help="Most scratch memory a TensorRT layer may use."
        )
        parser.add_argument(
            "--optimization_level", type=int, choices=range(6), help="Lower builds faster, with less memory."
        )
        parser.add_argument(
            "--offload_module_to_cpu",
            action="store_true",
            help="Move each program's PyTorch weights to the CPU while TensorRT builds its engine.",
        )
    return parser.parse_args()
