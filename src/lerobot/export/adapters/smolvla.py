# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""SmolVLA export adapter.

Composes PUBLIC SmolVLA modules into two exportable graphs:
  - GRAPH A (prefill): images + language + state -> prefix KV cache + pad masks
  - GRAPH B (denoise_step): noised actions + timestep + KV cache -> velocity

References (all public API, zero model edits):
  policy.model.embed_prefix        (modeling_smolvla.py:637)
  policy.model.denoise_step        (modeling_smolvla.py:883)
  policy.model.vlm_with_expert.forward  (smolvlm_with_expert.py:415)
  policy.model.vlm_with_expert.vlm.text_model.layers  (num_layers -> KV shape)

Phase 1 TODO: implement the actual graph wrappers below.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch.nn as nn

from lerobot.export.adapters.registry import register_adapter

if TYPE_CHECKING:
    from lerobot.export.cli import ExportConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: flatten / unflatten the plain-dict KV cache
# (smolvlm_with_expert.py:262-277: {layer_idx: {key_states, value_states}})
# ---------------------------------------------------------------------------


def flatten_kv_cache(kv: dict) -> list:
    """dict-of-dicts -> ordered list[Tensor] for export (write-once/read-only)."""
    out = []
    for layer_idx in sorted(kv.keys()):
        out.append(kv[layer_idx]["key_states"])
        out.append(kv[layer_idx]["value_states"])
    return out


def unflatten_kv_cache(flat: list, num_layers: int) -> dict:
    """Inverse of flatten_kv_cache."""
    kv = {}
    for i in range(num_layers):
        kv[i] = {"key_states": flat[2 * i], "value_states": flat[2 * i + 1]}
    return kv


# ---------------------------------------------------------------------------
# GRAPH A wrapper: prefill
# ---------------------------------------------------------------------------


class SmolVLAPrefill(nn.Module):
    """Wraps policy.model.embed_prefix + vlm_with_expert.forward (prefill pass).

    Inputs:  images (list[Tensor]), img_masks (list[Tensor]),
             lang_tokens (Tensor), lang_masks (Tensor), state (Tensor)
    Outputs: prefix_pad_masks (Tensor), *kv_flat (list[Tensor])

    Phase 1 TODO: implement forward().
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.m = model  # = policy.model (VLAFlowMatching)

    def forward(self, images, img_masks, lang_tokens, lang_masks, state):
        raise NotImplementedError(
            "SmolVLAPrefill.forward is a Phase 1 TODO. "
            "See doc 09 §2.3 for the composition recipe using "
            "policy.model.embed_prefix and policy.model.vlm_with_expert.forward."
        )


# ---------------------------------------------------------------------------
# GRAPH B wrapper: denoise_step
# ---------------------------------------------------------------------------


class SmolVLADenoiseStep(nn.Module):
    """Wraps policy.model.denoise_step (one Euler step).

    Inputs:  x_t (Tensor), timestep (Tensor), prefix_pad_masks (Tensor),
             *kv_flat (list[Tensor])
    Outputs: v_t (Tensor)

    Phase 1 TODO: implement forward().
    """

    def __init__(self, model: nn.Module, num_layers: int) -> None:
        super().__init__()
        self.m = model
        self.num_layers = num_layers

    def forward(self, x_t, timestep, prefix_pad_masks, *kv_flat):
        raise NotImplementedError(
            "SmolVLADenoiseStep.forward is a Phase 1 TODO. "
            "See doc 09 §2.3 for the composition recipe using "
            "policy.model.denoise_step and unflatten_kv_cache."
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_adapter
class SmolVLAExportAdapter:
    """Export adapter for the SmolVLA policy (anchor for Phase 1).

    The adapter composes ONLY public attributes of the policy:
      policy.model              (VLAFlowMatching)
      policy.config             (SmolVLAConfig)
    No model code is edited.
    """

    policy_name = "smolvla"

    def __init__(self, policy: nn.Module, cfg: ExportConfig) -> None:
        self.policy = policy
        self.cfg = cfg

    def shape_overrides(self) -> dict:
        # Force static language prefix length for export.
        return {"pad_language_to": "max_length"}

    def graphs(self, policy: nn.Module, cfg: ExportConfig) -> dict:
        """Return the two exportable graph parts.

        Phase 1 TODO: replace the placeholder modules with real ones that
        implement the prefill and denoise_step composition.
        Currently raises NotImplementedError when graph.forward() is called.
        """
        model = policy.model
        num_layers = len(model.vlm_with_expert.lm_expert.layers)

        prefill = SmolVLAPrefill(model)
        denoise = SmolVLADenoiseStep(model, num_layers)

        # Example inputs are derived from policy.config (static shapes).
        cfg_p = policy.config
        B = cfg.batch_size
        device = next(policy.parameters()).device
        import torch

        n_img = len(cfg_p.image_features)
        H, W = cfg_p.image_resolution
        img_ex = [torch.zeros(B, 3, H, W, device=device) for _ in range(n_img)]
        mask_ex = [torch.ones(B, dtype=torch.bool, device=device) for _ in range(n_img)]
        lang_len = cfg_p.tokenizer_max_length
        lang_tokens = torch.zeros(B, lang_len, dtype=torch.long, device=device)
        lang_masks = torch.ones(B, lang_len, dtype=torch.bool, device=device)
        state = torch.zeros(B, cfg_p.max_state_dim, device=device)

        prefill_inputs = ((img_ex, mask_ex, lang_tokens, lang_masks, state), {})

        # Denoise step inputs -- placeholder shapes (Phase 1 will compute prefix_len).
        prefix_len = 113  # approximate; Phase 1 computes from a real prefill pass
        x_t = torch.zeros(B, cfg_p.chunk_size, cfg_p.max_action_dim, device=device)
        timestep = torch.zeros(B, device=device)
        prefix_pad = torch.ones(B, prefix_len, dtype=torch.bool, device=device)
        kv_flat = [torch.zeros(B, 1, 1, device=device)] * (num_layers * 2)
        denoise_inputs = ((x_t, timestep, prefix_pad, *kv_flat), {})

        return {
            "prefill": (prefill, prefill_inputs),
            "denoise_step": (denoise, denoise_inputs),
        }

    def manifest_extra(self, policy: nn.Module) -> dict:
        cfg_p = policy.config
        return {
            "decode": {
                "type": "flow_matching_euler",
                "n_steps": cfg_p.num_inference_steps,
                "dt": f"-1.0/{cfg_p.num_inference_steps}",
                "noise": {"dist": "normal", "shape": [1, cfg_p.chunk_size, cfg_p.max_action_dim]},
                "x_init": "noise",
                "update": "x_t = x_t + dt * v_t",
                "action_unpad_dim": None,  # Phase 1: set from real action dim
            }
        }
