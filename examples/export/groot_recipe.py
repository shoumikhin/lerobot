#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""The GR00T N1.7 export recipe: what both backend scripts compile, and the folder they write.

As for SmolVLA, the compiled module is one action chunk of LeRobot's own GR00T, with the starting noise as
its last input and the checkpoint's saved postprocessor at the end. What differs is how the backbone gets
in. Qwen3-VL's own forward computes the rope index and the vision grid with Python loops over the inputs,
which no exporter can follow. NVIDIA's deployment code for GR00T (Isaac-GR00T,
scripts/deployment/export_onnx_n1d7.py) avoids them with two wrappers, ported below: the vision tower for
one fixed image grid, and the language model with the rope positions as an input.

For one task and one camera size, everything those loops compute is the same on every frame: the token
ids, the rope positions, and the rows the image features go to. So they are computed here, once, and held
as constants. The runtime resizes raw cameras with NumPy antialiased bicubic sampling. The program
takes the resized uint8 HWC cameras, float32 state and unbatched noise. Image normalization and patch
packing, state normalization and action unnormalization run inside the program. The folder records
the task it was exported for; no tokenizer runs at inference.
"""

import argparse
import gc
import json
from copy import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from act_recipe import TEST_CASE, gpu_processors, make_output_dir, to_policy_input
from safetensors.numpy import save_file
from smolvla_recipe import random_robot_frame
from torch import Tensor, nn
from transformers.feature_extraction_utils import BatchFeature

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.groot.groot_n1_7 import CategorySpecificLinear
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import GrootN17PackInputsStep, GrootN17VLMEncodeStep
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.rollout.inference.export.images import resize_images
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features

NOISE = "noise"


# From NVIDIA's export_onnx_n1d7.py: rotary with real numbers, where Qwen3-VL's own uses complex ones.
def _apply_rotary_real(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    orig_dtype = x.dtype
    x = x.float()
    cos = cos.float().unsqueeze(1)
    sin = sin.float().unsqueeze(1)
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos + rotated * sin).to(orig_dtype)


# Fix the image sequence lengths at export time instead of reading cu_seqlens inside the graph.
def vision_attention(attn: nn.Module, hidden_states: Tensor, position_embeddings, chunk_sizes: list[int]):
    seq_length = hidden_states.shape[0]
    qkv = attn.qkv(hidden_states).reshape(seq_length, 3, attn.num_heads, -1)
    q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
    cos, sin = position_embeddings
    q = _apply_rotary_real(q, cos, sin)
    k = _apply_rotary_real(k, cos, sin)
    outputs = []
    for q_c, k_c, v_c in zip(q.split(chunk_sizes), k.split(chunk_sizes), v.split(chunk_sizes), strict=True):
        q_c, k_c, v_c = (x.transpose(0, 1)[None] for x in (q_c, k_c, v_c))
        output = F.scaled_dot_product_attention(q_c, k_c, v_c, scale=attn.scaling)
        outputs.append(output[0].transpose(0, 1))
    return attn.proj(torch.cat(outputs, dim=0).reshape(seq_length, -1).contiguous())


class Qwen3VisionForExport(nn.Module):
    """NVIDIA's vision tower wrapper: the position and rotary embeddings of one image grid, computed once."""

    def __init__(self, vision_model: nn.Module, grid_thw: Tensor):
        super().__init__()
        self.chunk_sizes = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).tolist()
        self.patch_embed = vision_model.patch_embed
        self.blocks = vision_model.blocks
        self.merger = vision_model.merger
        self.deepstack_visual_indexes = vision_model.deepstack_visual_indexes
        self.deepstack_merger_list = vision_model.deepstack_merger_list
        with torch.no_grad():
            pos_embeds = vision_model.fast_pos_embed_interpolate(grid_thw)
            rotary = vision_model.rot_pos_emb(grid_thw)
            emb = torch.cat((rotary, rotary), dim=-1)
        self.register_buffer("_pos_embeds", pos_embeds.contiguous())
        self.register_buffer("_rot_cos", emb.cos().contiguous())
        self.register_buffer("_rot_sin", emb.sin().contiguous())

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        hidden_states = self.patch_embed(pixel_values) + self._pos_embeds
        position_embeddings = (self._rot_cos, self._rot_sin)
        deepstack_features = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = hidden_states + vision_attention(
                block.attn, block.norm1(hidden_states), position_embeddings, self.chunk_sizes
            )
            hidden_states = hidden_states + block.mlp(block.norm2(hidden_states))
            if layer_num in self.deepstack_visual_indexes:
                index = self.deepstack_visual_indexes.index(layer_num)
                deepstack_features.append(self.deepstack_merger_list[index](hidden_states))
        return self.merger(hidden_states), torch.stack(deepstack_features)


class LLMForExport(nn.Module):
    """NVIDIA's language model wrapper: rope positions as an input, its own causal mask and deepstack add.

    It runs the backbone's own decoder layers and rotary embedding, and returns the last layer's output
    before the final norm, which is what GR00T's action head reads.
    """

    def __init__(self, text_model: nn.Module):
        super().__init__()
        self.layers = text_model.layers
        self.rotary_emb = text_model.rotary_emb

    @staticmethod
    def _simple_causal_mask(dtype, device, batch_size, seq_len, attention_mask):
        mask_value = torch.finfo(dtype).min * 0.5
        causal_mask = torch.triu(torch.full((seq_len, seq_len), mask_value, device=device, dtype=dtype), 1)
        causal_mask = causal_mask[None, None].expand(batch_size, 1, -1, -1)
        padding_mask = (1.0 - attention_mask[:, None, None, :].to(dtype)) * mask_value
        return causal_mask + padding_mask

    @staticmethod
    def _deepstack_add(hidden_states, visual_pos_masks, visual_embeds):
        delta = torch.zeros_like(hidden_states).masked_scatter(
            visual_pos_masks.unsqueeze(-1).expand_as(hidden_states), visual_embeds
        )
        return hidden_states + delta

    def forward(self, inputs_embeds, attention_mask, position_ids, visual_pos_masks, *deepstack):
        batch_size, seq_len = inputs_embeds.shape[:2]
        attn_mask = self._simple_causal_mask(
            inputs_embeds.dtype, inputs_embeds.device, batch_size, seq_len, attention_mask
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids[0],
                past_key_values=None,
                position_embeddings=position_embeddings,
            )
            if layer_idx < len(deepstack):
                hidden_states = self._deepstack_add(
                    hidden_states, visual_pos_masks, deepstack[layer_idx].to(hidden_states.dtype)
                )
        return hidden_states


class GrootObservation(nn.Module):
    """The saved processor's tensor operations after runtime camera resizing."""

    def __init__(self, preprocessor: PolicyProcessorPipeline, frame: dict[str, np.ndarray], batch: dict):
        super().__init__()
        pack_index = next(
            i for i, s in enumerate(preprocessor.steps) if isinstance(s, GrootN17PackInputsStep)
        )
        pack = preprocessor.steps[pack_index]
        encode = next(s for s in preprocessor.steps if isinstance(s, GrootN17VLMEncodeStep))
        ip = encode.proc.image_processor
        if encode.use_albumentations or pack.video_horizon not in (None, 1):
            raise ValueError("Raw-frame export requires tensor image processing and one observation frame.")
        self.frame_names = list(frame)
        self.frame_processor = PolicyProcessorPipeline(steps=preprocessor.steps[:pack_index])
        sample = self.frame_processor({n: to_policy_input(n, torch.from_numpy(x)) for n, x in frame.items()})
        self.cameras = (
            sorted(n for n in sample if n.startswith(OBS_IMAGES))
            if not pack.video_modality_keys
            else [f"{OBS_IMAGES}.{n}" for n in pack.video_modality_keys]
        )
        if not self.cameras or not set(self.cameras) <= sample.keys():
            raise ValueError("The frame must provide the checkpoint's camera keys.")
        target_size = encode.image_target_size
        crop_fraction = encode.crop_fraction
        if crop_fraction is None and encode.image_crop_size and target_size:
            crop_fraction = encode.image_crop_size[0] / target_size[0]
        self.patch_size, self.temporal_patch_size, self.merge_size = (
            ip.patch_size,
            ip.temporal_patch_size,
            ip.merge_size,
        )
        grids = batch["image_grid_thw"].tolist()
        if len(grids) != len(self.cameras) or any(g != grids[0] for g in grids) or grids[0][0] != 1:
            raise ValueError("Raw-frame export requires equal, single-frame camera grids.")
        _, self.grid_h, self.grid_w = grids[0]
        self.image_resize = {
            "cameras": self.cameras,
            "target_size": target_size,
            "resize_edge": encode.shortest_image_edge or (target_size[0] if target_size else None),
            "crop_fraction": crop_fraction,
            "letterbox": encode.letter_box_transform,
            "image_size": [self.grid_h * ip.patch_size, self.grid_w * ip.patch_size],
        }
        self.rescale, self.normalize = ip.do_rescale, ip.do_normalize
        self.rescale_factor = ip.rescale_factor
        scale = 1 / ip.rescale_factor if self.rescale else 1
        self.register_buffer("image_mean", (torch.tensor(ip.image_mean) * scale)[None, :, None, None])
        self.register_buffer("image_std", (torch.tensor(ip.image_std) * scale)[None, :, None, None])
        dim = frame[OBS_STATE].size
        if dim > pack.max_state_dim:
            raise ValueError("The frame's state exceeds the checkpoint's state width.")
        self.max_state_dim = pack.max_state_dim
        self.normalize_state = pack.normalize_min_max and pack.stats is not None and OBS_STATE in pack.stats
        self.clip_state = pack.clip_outliers
        if self.normalize_state:
            stats = pack.stats[OBS_STATE]
            lo = torch.as_tensor(stats.get("min", torch.zeros(dim)), dtype=torch.float32).flatten()[:dim]
            hi = torch.as_tensor(stats.get("max", torch.ones(dim)), dtype=torch.float32).flatten()[:dim]
            lo, hi = F.pad(lo, (0, dim - lo.numel())), F.pad(hi, (0, dim - hi.numel()), value=1)
            span = hi - lo
            self.register_buffer("state_min", lo)
            self.register_buffer("state_nonzero", span != 0)
            self.register_buffer("state_span", torch.where(span != 0, span, torch.ones_like(span)))

    def forward(self, *frame: Tensor) -> tuple[Tensor, Tensor]:
        obs = self.frame_processor(
            {n: to_policy_input(n, x) for n, x in zip(self.frame_names, frame, strict=True)}
        )
        images = torch.cat([obs[n] for n in self.cameras])
        images = (images.clamp(0, 1) * 255).trunc()
        if self.normalize:
            images = (images - self.image_mean) / self.image_std
        elif self.rescale:
            images = images * self.rescale_factor
        p, t, m = self.patch_size, self.temporal_patch_size, self.merge_size
        # Combine channel and repeated time to stay within TensorRT's eight-dimension limit.
        patches = images[:, :, None].repeat(1, 1, t, 1, 1)
        patches = patches.reshape(len(self.cameras), 3 * t, self.grid_h // m, m, p, self.grid_w // m, m, p)
        pixels = patches.permute(0, 2, 5, 3, 6, 1, 4, 7).reshape(-1, 3 * t * p * p)
        state = obs[OBS_STATE]
        if self.normalize_state:
            state = torch.where(self.state_nonzero, 2 * (state - self.state_min) / self.state_span - 1, 0)
            if self.clip_state:
                state = state.clamp(-1, 1)
        return pixels, F.pad(state[:, None], (0, self.max_state_dim - state.shape[-1]))


class GrootChunk(nn.Module):
    """One action chunk: the raw frame and noise in, unbatched robot actions out."""

    def __init__(
        self,
        policy: GrootPolicy,
        batch: dict[str, Tensor],
        postprocessor: PolicyProcessorPipeline,
        preprocessor: GrootObservation,
    ):
        super().__init__()
        model = policy._groot_model
        backbone = model.backbone
        qwen = backbone.model.model
        self.action_head = model.action_head
        self.postprocessor = postprocessor
        self.preprocessor = preprocessor
        self.use_bf16 = policy.config.use_bf16
        self.horizon = policy._action_queue_steps
        self.action_dim = policy.config.output_features[ACTION].shape[0]

        # The constants Qwen3Backbone.forward would compute from these inputs on every frame.
        names = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
        model_input = {name: batch[name] for name in names if name in batch}
        ids, attention_mask = model_input["input_ids"], model_input["attention_mask"]
        mm_types = model_input.get("mm_token_type_ids", (ids == qwen.config.image_token_id).int())
        position_ids, _ = qwen.get_rope_index(
            input_ids=ids,
            mm_token_type_ids=mm_types,
            image_grid_thw=model_input["image_grid_thw"],
            attention_mask=attention_mask,
        )
        embodiment_id = batch["embodiment_id"]
        with torch.no_grad():
            text_embeds = backbone.language_model.get_input_embeddings()(ids)
            image_token_mask, _ = qwen.get_placeholder_mask(ids, inputs_embeds=text_embeds)
            # The action head keeps weights for every embodiment it can be trained on; the program runs one.
            for layer in self.action_head.modules():
                if isinstance(layer, CategorySpecificLinear):
                    layer.W = nn.Parameter(layer.W[embodiment_id], requires_grad=False)
                    layer.b = nn.Parameter(layer.b[embodiment_id], requires_grad=False)
        constants = {
            "text_embeds": text_embeds,
            "image_token_mask": image_token_mask,
            "attention_mask": attention_mask.long(),
            "position_ids": position_ids,
            "visual_pos_masks": image_token_mask[..., 0],
            "image_mask": ids == backbone.model.config.image_token_id,
            "backbone_attention_mask": attention_mask == 1,
            "embodiment_id": torch.zeros_like(embodiment_id),
        }
        for name, value in constants.items():
            self.register_buffer(name, value, persistent=False)
        self.vision = Qwen3VisionForExport(qwen.visual, model_input["image_grid_thw"])
        self.llm = LLMForExport(backbone.language_model)

    def forward(self, *inputs: Tensor) -> Tensor:
        *frame, noise = inputs
        pixel_values, state = self.preprocessor(*frame)
        # The same autocast GrootPolicy.predict_action_chunk runs the model under.
        with torch.autocast("cuda", torch.bfloat16, enabled=self.use_bf16):
            image_embeds, deepstack = self.vision(pixel_values)
            inputs_embeds = self.text_embeds.masked_scatter(
                self.image_token_mask, image_embeds.to(self.text_embeds.dtype)
            )
            features = self.llm(
                inputs_embeds, self.attention_mask, self.position_ids, self.visual_pos_masks, *deepstack
            )
            backbone_output = BatchFeature(
                data={
                    "backbone_features": features,
                    "backbone_attention_mask": self.backbone_attention_mask,
                    "image_mask": self.image_mask,
                }
            )
            action_input = BatchFeature(data={"state": state, "embodiment_id": self.embodiment_id})
            actions = self.action_head.get_action(backbone_output, action_input, noise=noise[None])
        return self.postprocessor(actions["action_pred"][:, : self.horizon, : self.action_dim])[0]


class GrootExport:
    """Loads a trained GR00T N1.7 checkpoint, and writes the exported folder around a compiled program."""

    def __init__(
        self,
        policy_path: str,
        dataset: str,
        dataset_root: Path | None,
        output_dir: Path | None,
        job_name: str,
    ):
        self.output_dir = make_output_dir(output_dir, job_name)
        self.policy_path = policy_path
        self.policy = GrootPolicy.from_pretrained(policy_path).to("cuda").eval()
        self.config = self.policy.config
        preprocessor, postprocessor = gpu_processors(self.policy.config, policy_path)

        metadata = LeRobotDatasetMetadata(dataset, root=dataset_root)
        self.task = str(metadata.tasks.index[0])
        self.frame = random_robot_frame(metadata.features)
        self.input_features = {
            name: feature
            for name, feature in dataset_to_policy_features(metadata.features).items()
            if name in self.frame
        }
        batch = preprocessor(self.observation())
        model = self.policy._groot_model
        self.noise = torch.randn(
            model.action_head.action_horizon, model.action_head.action_dim, device="cuda"
        )
        self.expected_actions = self.rollout_actions(preprocessor, postprocessor)
        observation = GrootObservation(preprocessor, self.frame, batch).cuda()
        self.image_resize = observation.image_resize
        self.module = GrootChunk(self.policy, batch, postprocessor, observation).eval()
        resized_frame = resize_images(self.frame, self.image_resize)
        self.inputs = (*(torch.from_numpy(x).cuda() for x in resized_frame.values()), self.noise)
        self.input_names = [*self.frame, NOISE]

    def release_policy(self) -> None:
        """Free the policy for the TensorRT step: the test case's actions are already computed."""
        del self.policy, self.module
        gc.collect()
        torch.cuda.empty_cache()

    def observation(self) -> dict[str, Tensor]:
        return prepare_observation_for_inference(dict(self.frame), torch.device("cuda"), self.task)

    def rollout_actions(
        self, preprocessor: PolicyProcessorPipeline, postprocessor: PolicyProcessorPipeline
    ) -> np.ndarray:
        """The chunk the PyTorch policy plays for the test frame, task and noise, as lerobot-rollout runs it."""
        self.policy.reset()
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(
                preprocessor(self.observation()), noise=self.noise[None]
            )
            actions = postprocessor(actions[:, : self.policy._action_queue_steps])
        return actions[0].float().cpu().numpy()

    def write(self, backend: str, program_file: str, tolerance: float) -> None:
        """Save the raw test case, policy config and `export.json` beside the program."""
        case = {**self.frame, NOISE: self.noise.cpu().numpy(), "expected_actions": self.expected_actions}
        save_file(case, self.output_dir / TEST_CASE)
        config = copy(self.config)
        config.input_features = self.input_features
        config.save_pretrained(self.output_dir)
        info = {
            "backend": backend,
            "file": program_file,
            "inputs": self.input_names,
            "raw_frame": True,
            "image_resize": self.image_resize,
            "task": self.task,
            # The task's token ids and rope positions are constants inside the program.
            "task_fixed": True,
            "noise_shape": list(self.noise.shape),
            "output": ACTION,
            "test_case": TEST_CASE,
            "tolerance": tolerance,
        }
        (self.output_dir / "export.json").write_text(json.dumps(info, indent=2) + "\n")
        print(f"Wrote {self.output_dir}")


def parse_args(description: str, backend: str) -> argparse.Namespace:
    """The command line both GR00T export scripts share."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--policy.path", dest="policy_path", required=True, help="Trained GR00T N1.7 checkpoint."
    )
    parser.add_argument(
        "--dataset.repo_id", dest="dataset", required=True, help="The dataset the policy was trained on."
    )
    parser.add_argument(
        "--dataset.root",
        dest="dataset_root",
        type=Path,
        help="Local copy of that dataset, if not on the Hub.",
    )
    parser.add_argument(
        "--output_dir", type=Path, help="Folder to write. Default: a new one under outputs/export."
    )
    parser.add_argument("--job_name", default=f"groot_{backend}", help="Names the default output folder.")
    parser.add_argument(
        "--tolerance", type=float, default=5.0, help="Largest allowed action error, robot units."
    )
    parser.add_argument(
        "--export_only",
        action="store_true",
        help="Write the folder without the engine, to build it with build_engine.py on each device.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args("Check the GR00T recipe module in eager against the PyTorch rollout chunk.", "eager")
    export = GrootExport(args.policy_path, args.dataset, args.dataset_root, args.output_dir, args.job_name)
    with torch.inference_mode():
        actual = export.module(*export.inputs).float().cpu().numpy()
    print("inputs", [(tuple(t.shape), str(t.dtype)) for t in export.inputs])
    print(
        f"EAGER_VS_ROLLOUT max_abs={np.abs(actual - export.expected_actions).max():.4g} "
        f"ref_max={np.abs(export.expected_actions).max():.4g} shape={actual.shape}"
    )
