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
as constants. The program takes the patchified images, the packed state and the noise. The checkpoint's
preprocessor, saved beside it, makes the first two from the robot's frame at rollout, and the folder
records the task it was exported for.
"""

import argparse
import json
from copy import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from act_recipe import TEST_CASE, gpu_processors, make_output_dir
from safetensors.numpy import save_file
from smolvla_recipe import random_robot_frame
from torch import Tensor, nn
from transformers.feature_extraction_utils import BatchFeature

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION
from lerobot.utils.feature_utils import dataset_to_policy_features

NOISE = "noise"
INPUTS = ["pixel_values", "state", NOISE]
STEPS = "policy_steps.json"


# From NVIDIA's export_onnx_n1d7.py: rotary with real numbers, where Qwen3-VL's own uses complex ones.
def _apply_rotary_real(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    orig_dtype = x.dtype
    x = x.float()
    cos = cos.float().unsqueeze(1)
    sin = sin.float().unsqueeze(1)
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos + rotated * sin).to(orig_dtype)


# From NVIDIA's export_onnx_n1d7.py: each image attends within its own fixed chunk of patches, where
# Qwen3-VL's own attention splits the sequence at runtime by cu_seqlens.
def _make_vision_attention_forward(attn_module: nn.Module, chunk_sizes: list[int]):
    def forward(hidden_states, cu_seqlens=None, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        seq_length = hidden_states.shape[0]
        qkv = attn_module.qkv(hidden_states).reshape(seq_length, 3, attn_module.num_heads, -1)
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        q = _apply_rotary_real(q, cos, sin)
        k = _apply_rotary_real(k, cos, sin)
        outputs = []
        for q_c, k_c, v_c in zip(
            q.split(chunk_sizes), k.split(chunk_sizes), v.split(chunk_sizes), strict=True
        ):
            q_c, k_c, v_c = q_c.transpose(0, 1), k_c.transpose(0, 1), v_c.transpose(0, 1)
            w = torch.matmul(q_c, k_c.transpose(-2, -1)) * attn_module.scaling
            w = F.softmax(w.to(torch.float32), dim=-1).to(v_c.dtype)
            outputs.append(torch.matmul(w, v_c).transpose(0, 1))
        attn_output = torch.cat(outputs, dim=0).reshape(seq_length, -1).contiguous()
        return attn_module.proj(attn_output)

    return forward


class Qwen3VisionForExport(nn.Module):
    """NVIDIA's vision tower wrapper: the position and rotary embeddings of one image grid, computed once."""

    def __init__(self, vision_model: nn.Module, grid_thw: Tensor):
        super().__init__()
        chunk_sizes = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).tolist()
        for block in vision_model.blocks:
            block.attn.forward = _make_vision_attention_forward(block.attn, chunk_sizes)
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
            hidden_states = block(hidden_states, cu_seqlens=None, position_embeddings=position_embeddings)
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


class GrootChunk(nn.Module):
    """One action chunk: the patchified images, the packed state and the noise in, robot actions out."""

    def __init__(self, policy: GrootPolicy, batch: dict[str, Tensor], postprocessor: PolicyProcessorPipeline):
        super().__init__()
        model = policy._groot_model
        backbone = model.backbone
        qwen = backbone.model.model
        self.action_head = model.action_head
        self.postprocessor = postprocessor
        self.use_bf16 = policy.config.use_bf16
        self.horizon = policy._action_queue_steps
        self.action_dim = policy.config.output_features[ACTION].shape[0]

        # The constants Qwen3Backbone.forward would compute from these inputs on every frame.
        names = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
        model_input = {name: batch[name] for name in names if name in batch}
        backbone._ensure_mm_token_type_ids(model_input)
        backbone._ensure_legacy_qwen3_position_ids(model_input)
        ids, attention_mask = model_input["input_ids"], model_input["attention_mask"]
        with torch.no_grad():
            text_embeds = backbone.language_model.get_input_embeddings()(ids)
            image_token_mask, _ = qwen.get_placeholder_mask(ids, inputs_embeds=text_embeds)
        constants = {
            "text_embeds": text_embeds,
            "image_token_mask": image_token_mask,
            "attention_mask": attention_mask.long(),
            # Qwen3Backbone adds a text row in front of the three multimodal rows; NVIDIA's wrapper takes the three.
            "position_ids": model_input["position_ids"][1:],
            "visual_pos_masks": image_token_mask[..., 0],
            "image_mask": ids == backbone.model.config.image_token_id,
            "backbone_attention_mask": attention_mask == 1,
            "embodiment_id": batch["embodiment_id"],
        }
        for name, value in constants.items():
            self.register_buffer(name, value, persistent=False)
        self.vision = Qwen3VisionForExport(qwen.visual, model_input["image_grid_thw"])
        self.llm = LLMForExport(backbone.language_model)

    def forward(self, pixel_values: Tensor, state: Tensor, noise: Tensor) -> Tensor:
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
            actions = self.action_head.get_action(backbone_output, action_input, noise=noise)
        return self.postprocessor(actions["action_pred"][:, : self.horizon, : self.action_dim])


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
        # The rollout runs the checkpoint's whole preprocessor before the program: the program takes its outputs.
        preprocessor, postprocessor = gpu_processors(self.policy.config, policy_path)
        self.steps = preprocessor

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
            1, model.action_head.action_horizon, model.action_head.action_dim, device="cuda"
        )
        # Before the module exists: it swaps the vision attention for NVIDIA's.
        self.expected_actions = self.rollout_actions(preprocessor, postprocessor)
        self.module = GrootChunk(self.policy, batch, postprocessor).eval()
        self.inputs = (batch["pixel_values"], batch["state"], self.noise)

    def observation(self) -> dict[str, Tensor]:
        return prepare_observation_for_inference(dict(self.frame), torch.device("cuda"), self.task)

    def rollout_actions(
        self, preprocessor: PolicyProcessorPipeline, postprocessor: PolicyProcessorPipeline
    ) -> np.ndarray:
        """The chunk the PyTorch policy plays for the test frame, task and noise, as lerobot-rollout runs it."""
        self.policy.reset()
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(preprocessor(self.observation()), noise=self.noise)
            actions = postprocessor(actions[:, : self.policy._action_queue_steps])
        return actions[0].float().cpu().numpy()

    def write(self, backend: str, program_file: str, tolerance: float) -> None:
        """Save the test case, the policy config, the preprocessor steps and `export.json` beside the program."""
        case = {**self.frame, NOISE: self.noise.cpu().numpy(), "expected_actions": self.expected_actions}
        save_file(case, self.output_dir / TEST_CASE)
        config = copy(self.policy.config)
        config.input_features = self.input_features
        config.save_pretrained(self.output_dir)
        self.steps.save_pretrained(self.output_dir, config_filename=STEPS)
        info = {
            "backend": backend,
            "file": program_file,
            "inputs": INPUTS,
            "text_steps": STEPS,
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args("Check the GR00T recipe module in eager against the PyTorch rollout chunk.", "eager")
    export = GrootExport(args.policy_path, args.dataset, args.dataset_root, args.output_dir, args.job_name)
    with torch.inference_mode():
        actual = export.module(*export.inputs)[0].float().cpu().numpy()
    print("inputs", [(tuple(t.shape), str(t.dtype)) for t in export.inputs])
    print(
        f"EAGER_VS_ROLLOUT max_abs={np.abs(actual - export.expected_actions).max():.4g} "
        f"ref_max={np.abs(export.expected_actions).max():.4g} shape={actual.shape}"
    )
