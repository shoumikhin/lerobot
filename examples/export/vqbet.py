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

"""The VQ-BeT recipe: one action chunk of LeRobot's own VQ-BeT, as one program.

The program reads the robot's last `n_obs_steps` frames, which the rollout keeps for it. VQ-BeT samples one
code per residual codebook layer with `torch.multinomial`; the program samples the same distribution by
inverse transform from the normal noise it takes as an input, like every other policy's, turned uniform by
the normal distribution function. So the program is deterministic and needs no random op.
Only the code of the current frame's action token reaches the actions, so only it is sampled.
"""

import argparse
from collections.abc import Iterator

import einops
import numpy as np
import torch
from common import NOISE, Part, device, frame_from_policy, processors, write_folder
from torch import Tensor, nn

from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_STATE

TOLERANCE = 0.5


def sample_codes(probs: Tensor, uniform: Tensor) -> Tensor:
    """One code per row of `probs`, drawn by inverse transform from `uniform`, one value per row."""
    below = (probs.cumsum(-1) < uniform[..., None]).sum(-1)
    return below.clamp(max=probs.shape[-1] - 1)


class VQBeTChunk(nn.Module):
    """One action chunk: the last frames and the uniform noise in, the actions to play out, in robot units."""

    def __init__(
        self,
        policy: VQBeTPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
    ):
        super().__init__()
        if policy.config.sequentially_select:
            raise SystemExit("VQ-BeT checkpoints that select codes sequentially are not supported.")
        self.policy = policy
        self.model = policy.vqbet
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.frame_names = list(policy.config.input_features)
        self.image_names = list(policy.config.image_features)

    def forward(self, *inputs: Tensor) -> Tensor:
        *frames, noise = inputs
        uniform = 0.5 * (1 + torch.erf(noise / 2**0.5))
        config = self.policy.config
        model = self.model
        observation = {
            name: (x.float() / 255).permute(0, 3, 1, 2) if name in self.image_names else x
            for name, x in zip(self.frame_names, frames, strict=True)
        }
        observation = self.preprocessor(observation)
        steps = config.n_obs_steps
        images = torch.stack([observation[name] for name in self.image_names], dim=-4)
        # `VQBeTModel.forward` up to the current frame's action token, for a batch of one.
        features = model.rgb_encoder(einops.rearrange(images, "s n ... -> (s n) ..."))
        rgb_tokens = model.rgb_feature_projector(einops.rearrange(features, "(s n) ... -> s n ...", s=steps))
        tokens = [rgb_tokens[:, i] for i in range(rgb_tokens.size(1))]
        tokens.append(model.state_projector(observation[OBS_STATE]))
        tokens.append(model.action_token[0].expand(steps, -1))
        tokens = einops.rearrange(torch.stack(tokens, dim=1), "n t d -> (n t) d")[None]
        future = model.action_token.repeat(1, config.n_action_pred_token - 1, 1)
        hidden = model.policy(torch.cat([tokens, future], dim=1))
        x = hidden[:, (steps - 1) * (len(config.input_features) + 1) + len(config.input_features)]
        # `VQBeTHead.forward` for that token, sampling by inverse transform.
        head = model.action_head
        layers, codes = head.vqvae_model.vqvae_num_layers, config.vqvae_n_embed
        offsets = einops.rearrange(
            head.map_to_cbet_preds_offset(x), "1 (g c wa) -> g c wa", g=layers, c=codes
        )
        logits = einops.rearrange(head.map_to_cbet_preds_bin(x), "1 (g c) -> g c", g=layers)
        centers = sample_codes(torch.softmax(logits / config.bet_softmax_temperature, dim=-1), uniform)
        offset = offsets[torch.arange(layers, device=x.device), centers].sum(0)
        latent = head.vqvae_model.get_embeddings_from_code(centers[None])
        decoded = head.vqvae_model.get_action_from_latent(latent)
        actions = decoded + einops.rearrange(offset, "(w a) -> 1 w a", w=config.action_chunk_size)
        return self.postprocessor(actions)[0]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """VQ-BeT needs nothing beyond the checkpoint."""


def part_names(args: argparse.Namespace) -> list[str]:
    return ["model"]


class Exporter:
    """Loads a trained VQ-BeT checkpoint, and writes the folder around its compiled program."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.policy = VQBeTPolicy.from_pretrained(args.policy_path).to(device()).eval()
        config = self.policy.config
        frame = frame_from_policy(config)
        # The test case starts an episode, where the history holds copies of the first frame.
        self.frame = {name: np.stack([x] * config.n_obs_steps) for name, x in frame.items()}
        self.noise = torch.randn(self.policy.vqbet.action_head.vqvae_model.vqvae_num_layers, device=device())
        self.expected_actions = None

    def parts(self) -> Iterator[Part]:
        module = VQBeTChunk(self.policy, *processors(self.policy.config, self.args.policy_path))
        inputs = (*(torch.from_numpy(x).to(device()) for x in self.frame.values()), self.noise)
        with torch.no_grad():
            self.expected_actions = module(*inputs).cpu().numpy()
        yield Part("model", module, inputs, [*self.frame, NOISE], [ACTION])

    def write(self, programs: dict[str, dict]) -> None:
        write_folder(
            self.args,
            self.policy.config,
            self.frame,
            self.noise,
            self.expected_actions,
            programs,
            n_obs_steps=self.policy.config.n_obs_steps,
        )
