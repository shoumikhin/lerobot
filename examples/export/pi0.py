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

"""The pi0 recipe: one action chunk of LeRobot's own pi0, as a chain of programs.

pi0 is pi0.5's predecessor with the same vision-language prefix, so the chain is pi0.5's: the embeddings,
the language model in groups of layers, one denoising step, and the actions. The differences are that pi0
reads the task alone as its prompt, so the prompt's tokens are constants made at export time, and that it
feeds the state to the action expert, so the denoising step takes the normalized state too.
"""

import argparse
import math
from collections.abc import Iterator
from copy import copy

import torch
from common import (
    ChunkActions,
    Part,
    device,
    frame_from_policy,
    free_memory,
    hold_linear_weights_in_bfloat16,
    processors,
    to_policy_input,
    write_folder,
)
from pi05 import (
    EMBED_MODULES,
    LANGUAGE_LAYERS_PER_PROGRAM,
    LANGUAGE_MODEL,
    CompactEmbedding,
    PI05LanguageModelLayers,
)
from torch import Tensor, nn
from transformers import DynamicCache
from transformers.utils import cached_file

from lerobot.configs import PreTrainedConfig
from lerobot.policies.common.openpi_checkpoint import load_checkpoint_modules
from lerobot.policies.pi0.modeling_pi0 import PI0Policy, PI0Pytorch, get_gemma_config
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline, TokenizerProcessorStep
from lerobot.processor.pipeline import ComplementaryDataProcessorStep
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

# As for pi0.5: TensorRT rounds bfloat16 differently from PyTorch.
TOLERANCE = 5.0
STEP_MODULES = (
    "model.paligemma_with_expert.gemma_expert.model",
    "model.state_proj",
    "model.action_in_proj",
    "model.action_out_proj",
    "model.action_time_mlp_in",
    "model.action_time_mlp_out",
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True, help="The task the policy was trained on, as a sentence.")
    parser.add_argument("--cameras", nargs="+", help="The camera slots the robot fills; pi0 pads the rest.")


def part_names(args: argparse.Namespace) -> list[str]:
    depth = get_gemma_config(PreTrainedConfig.from_pretrained(args.policy_path).paligemma_variant).depth
    groups = range(math.ceil(depth / LANGUAGE_LAYERS_PER_PROGRAM))
    return ["embed", *(f"language_model_{group}" for group in groups), "step", "actions"]


class PI0Embed(nn.Module):
    """The raw frame in; the prefix's embeddings and mask, and the padded normalized state, out."""

    def __init__(
        self,
        policy: PI0Policy,
        frame_names: list[str],
        preprocessor: PolicyProcessorPipeline,
        prompt: dict[str, Tensor],
    ):
        super().__init__()
        self.policy = policy
        self.frame_names = frame_names
        self.preprocessor = preprocessor
        self.register_buffer("tokens", prompt[OBS_LANGUAGE_TOKENS])
        self.register_buffer("attention_mask", prompt[OBS_LANGUAGE_ATTENTION_MASK])

    def forward(self, *frame: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch = self.preprocessor(
            {name: to_policy_input(name, x) for name, x in zip(self.frame_names, frame, strict=True)}
        )
        # The policy's own image resize and padding, which it exposes under no public name.
        images, img_masks = self.policy._preprocess_images(batch)
        prefix_embs, prefix_pad_masks, _ = self.policy.model.embed_prefix(
            images, img_masks, self.tokens, self.attention_mask
        )
        return prefix_embs, prefix_pad_masks, self.policy.prepare_state(batch)[0]


class PI0Step(nn.Module):
    """One denoising step of LeRobot's `denoise_step`, with its Euler update, from the prefix's KV cache."""

    def __init__(self, model: PI0Pytorch, dt: float):
        super().__init__()
        self.model = model
        self.dt = dt
        if device().type == "cuda":
            hold_linear_weights_in_bfloat16(model)

    def forward(self, prefix_pad_masks: Tensor, *inputs: Tensor) -> Tensor:
        *cache, state, x_t, timestep = inputs
        past_key_values = DynamicCache(
            tuple((keys, values, None) for keys, values in zip(cache[::2], cache[1::2], strict=True))
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            velocity = self.model.denoise_step(
                state[None], prefix_pad_masks, past_key_values, x_t[None], timestep
            )
        return x_t + self.dt * velocity[0].float()


class Exporter:
    """Reads a trained pi0 checkpoint one part at a time, and writes the folder around its programs."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = PreTrainedConfig.from_pretrained(args.policy_path)
        self.config.device = str(device())
        self.model_file = cached_file(args.policy_path, "model.safetensors")
        self.policy = self.load_modules(*EMBED_MODULES)
        vlm = self.policy.model.paligemma_with_expert
        if vlm.precision == torch.bfloat16 and device().type == "cuda":
            hold_linear_weights_in_bfloat16(vlm.paligemma.model.vision_tower)
            hold_linear_weights_in_bfloat16(vlm.paligemma.model.multi_modal_projector)
        config = self.policy.config
        preprocessor, self.postprocessor = processors(config, args.policy_path)
        empty_cameras = {f"{OBS_IMAGES}.empty_camera_{i}" for i in range(config.empty_cameras)}
        available = set(config.image_features) - empty_cameras
        cameras = available if args.cameras is None else set(args.cameras)
        if not cameras or not cameras <= available:
            raise SystemExit(f"Choose --cameras from {sorted(available)}.")
        self.frame = {
            name: value
            for name, value in frame_from_policy(config).items()
            if name not in config.image_features or name in cameras
        }
        self.folder_config = copy(config)
        self.folder_config.input_features = {name: config.input_features[name] for name in self.frame}
        text = [
            s
            for s in preprocessor.steps
            if isinstance(s, TokenizerProcessorStep | ComplementaryDataProcessorStep)
        ]
        observation = prepare_observation_for_inference(dict(self.frame), device(), args.task)
        prompt = PolicyProcessorPipeline(steps=text)(observation)
        tensor_steps = PolicyProcessorPipeline(steps=[s for s in preprocessor.steps if s not in text])
        language_model = self.policy.model.paligemma_with_expert.paligemma.model.language_model
        token_ids = prompt[OBS_LANGUAGE_TOKENS].flatten().unique()
        language_model.embed_tokens = CompactEmbedding(language_model.embed_tokens, token_ids)
        self.embed = PI0Embed(self.policy, list(self.frame), tensor_steps, prompt)
        del self.policy
        self.noise = torch.randn(config.chunk_size, config.max_action_dim, device=device())
        self.dt = -1.0 / config.num_inference_steps
        self.timesteps = [1.0 + step * self.dt for step in range(config.num_inference_steps)]
        self.expected_actions = None

    def load_modules(self, *modules: str) -> PI0Policy:
        """The policy with only `modules`' weights read from the checkpoint; every other module is None."""
        return load_checkpoint_modules(PI0Policy, self.model_file, self.config, list(modules))

    def parts(self) -> Iterator[Part]:
        """The chain's parts, each run in PyTorch to make the next part's inputs, one in memory at a time."""
        frame = tuple(torch.from_numpy(x).to(device()) for x in self.frame.values())
        with torch.no_grad():
            hidden, prefix_pad_masks, state = self.embed(*frame)
        yield Part("embed", self.embed, frame, list(self.frame), ["prefix_embs", "prefix_pad_masks", "state"])
        del self.embed
        hidden_name, cache = "prefix_embs", {}
        depth = get_gemma_config(self.config.paligemma_variant).depth
        for group, first in enumerate(range(0, depth, LANGUAGE_LAYERS_PER_PROGRAM)):
            layers = range(first, min(first + LANGUAGE_LAYERS_PER_PROGRAM, depth))
            free_memory()
            module = PI05LanguageModelLayers(
                self.load_modules(*(f"{LANGUAGE_MODEL}.layers.{i}" for i in layers)).model, layers
            )
            inputs = (hidden, prefix_pad_masks)
            with torch.no_grad():
                keys_values = module(*inputs)
            cache_names = [f"past_{kind}_{i}" for i in layers for kind in ("key", "value")]
            outputs = cache_names
            if not module.last:
                outputs = [f"hidden_{group}", *cache_names]
                hidden, *keys_values = keys_values
            cache.update(zip(cache_names, keys_values, strict=True))
            yield Part(f"language_model_{group}", module, inputs, [hidden_name, "prefix_pad_masks"], outputs)
            del module
            hidden_name = outputs[0]
        free_memory()
        step = PI0Step(self.load_modules(*STEP_MODULES).model, self.dt)
        cache_inputs = (prefix_pad_masks, *cache.values(), state)
        x_t = self.noise
        with torch.no_grad():
            for timestep in self.timesteps:
                x_t = step(*cache_inputs, x_t, torch.tensor([timestep], device=device()))
        step_inputs = (*cache_inputs, self.noise, torch.ones(1, device=device()))
        yield Part(
            "step", step, step_inputs, ["prefix_pad_masks", *cache, "state", "x_t", "timestep"], ["x_next"]
        )
        del step
        free_memory()
        actions = ChunkActions(
            self.config.n_action_steps, self.config.output_features[ACTION].shape[0], self.postprocessor
        )
        raw_state = torch.from_numpy(self.frame[OBS_STATE]).to(device())
        with torch.no_grad():
            self.expected_actions = actions(x_t, raw_state).cpu().numpy()
        yield actions.part(self.noise, raw_state)

    def write(self, programs: dict[str, dict]) -> None:
        write_folder(
            self.args,
            self.folder_config,
            self.frame,
            self.noise,
            self.expected_actions,
            programs,
            steps=["step"],
            timesteps=self.timesteps,
            # The program holds the task's prompt tokens, so the folder runs only this task.
            task=self.args.task,
        )
