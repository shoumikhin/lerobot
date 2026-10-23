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

"""The X-VLA recipe: one action chunk of LeRobot's own X-VLA, as one program.

As for SmolVLA, the program goes from the robot's frame and the noise to actions in robot units, and holds
the task's token ids as constants. X-VLA counts its present cameras and checks image value ranges with
host-side branches; the program knows its cameras and its value ranges at export time, so it runs the
same arithmetic without them.
"""

import argparse
from collections.abc import Iterator
from copy import copy

import torch
from common import (
    NOISE,
    Part,
    device,
    frame_from_policy,
    processors,
    to_policy_input,
    write_folder,
)
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.xvla.processor_xvla import (
    XVLAAddDomainIdProcessorStep,
    XVLAImageNetNormalizeProcessorStep,
    XVLAImageToFloatProcessorStep,
)
from lerobot.processor import PolicyProcessorPipeline, TokenizerProcessorStep
from lerobot.utils.constants import (
    ACTION,
    IMAGENET_STATS,
    OBS_IMAGES,
    OBS_LANGUAGE_TOKENS,
)

TOLERANCE = 5.0


class XVLAEncode(nn.Module):
    """The robot's frame in; the encoded prompt and cameras, and the padded state, out."""

    def __init__(
        self,
        policy: XVLAPolicy,
        preprocessor: PolicyProcessorPipeline,
        frame_names: list[str],
        tokens: Tensor,
    ):
        super().__init__()
        self.policy = policy
        self.model = policy.model
        # Attention as plain matrix products, as the other recipes run it. Torch-TensorRT's attention rewrite
        # breaks Florence-2's vision tower, which views the attention output with the strides SDPA returns.
        self.model.vlm.set_attn_implementation("eager")
        self.preprocessor = preprocessor
        self.frame_names = frame_names
        self.cameras = sum(name.startswith(OBS_IMAGES) for name in frame_names)
        self.register_buffer("tokens", tokens)
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_STATS["mean"])[:, None, None]
        )
        self.register_buffer("std", torch.tensor(IMAGENET_STATS["std"])[:, None, None])

    def forward(self, *frame: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        observation = {}
        for name, x in zip(self.frame_names, frame, strict=True):
            x = to_policy_input(name, x)
            observation[name] = (
                (x - self.mean) / self.std if name.startswith(OBS_IMAGES) else x
            )
        batch = self.preprocessor(observation)
        batch[OBS_LANGUAGE_TOKENS] = self.tokens
        model_inputs = self.policy._build_model_inputs(batch)
        dtype = self.model.config.dtype
        # `XVLAModel.forward_vlm` with the present cameras, which come first, known at export time.
        images = model_inputs["image_input"]
        valid = self.model.vlm.get_image_features(
            images[0, : self.cameras].to(dtype)
        ).pooler_output
        padding = valid.new_zeros((images.shape[1] - self.cameras, *valid.shape[1:]))
        features = torch.cat((valid, padding))[None]
        embeds = torch.cat(
            [features[:, 0], self.model.vlm.get_input_embeddings()(self.tokens)], dim=1
        )
        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
        encoded = self.model.vlm.language_model.encoder(
            attention_mask=mask, inputs_embeds=embeds
        )[0]
        return encoded, features[:, 1:].flatten(1, 2), model_inputs["proprio"].to(dtype)


class XVLADenoise(nn.Module):
    """`XVLAModel.generate_actions`' loop from the encoded observation and the noise; actions in robot units out."""

    def __init__(
        self, policy: XVLAPolicy, postprocessor: PolicyProcessorPipeline, domain_id: int
    ):
        super().__init__()
        self.policy = policy
        self.model = policy.model
        self.postprocessor = postprocessor
        self.register_buffer("domain_id", torch.tensor([domain_id]))

    def forward(
        self,
        vlm_features: Tensor,
        aux_visual_inputs: Tensor,
        proprio: Tensor,
        noise: Tensor,
    ) -> Tensor:
        dtype = self.model.config.dtype
        x1 = noise[None].to(dtype)
        action = torch.zeros_like(x1)
        steps = self.policy.config.num_denoising_steps
        for i in range(steps, 0, -1):
            t = torch.full((1,), i / steps, device=x1.device, dtype=dtype)
            proprio_m, x_t_m = self.model.action_space.preprocess(
                proprio, x1 * t + action * (1 - t)
            )
            action = self.model.transformer(
                domain_id=self.domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                vlm_features=vlm_features,
                aux_visual_inputs=aux_visual_inputs,
            )
        actions = self.model.action_space.postprocess(action.float())
        return self.postprocessor(actions[:, : self.policy.config.n_action_steps])[0]


class XVLAChunk(nn.Module):
    """One action chunk: the robot's frame and the noise in, the actions to play out, in robot units."""

    def __init__(self, encode: XVLAEncode, denoise: XVLADenoise):
        super().__init__()
        self.encode = encode
        self.denoise = denoise

    def forward(self, *inputs: Tensor) -> Tensor:
        *frame, noise = inputs
        return self.denoise(*self.encode(*frame), noise)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task", required=True, help="The task to export for, as a sentence."
    )
    parser.add_argument(
        "--cameras", nargs="+", help="The camera slots the robot fills. Default: all."
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        help="Run the model in this dtype. Default: the checkpoint's.",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Build the encoder and the denoising loop as two programs.",
    )


def part_names(args: argparse.Namespace) -> list[str]:
    return ["encode", "denoise"] if args.split else ["model"]


class Exporter:
    """Loads a trained X-VLA checkpoint, and writes the folder around its compiled program."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        config = PreTrainedConfig.from_pretrained(args.policy_path)
        if args.dtype is not None:
            config.dtype = getattr(torch, args.dtype)
        self.policy = (
            XVLAPolicy.from_pretrained(args.policy_path, config=config)
            .to(device())
            .eval()
        )
        config = self.policy.config
        empty = {
            f"{OBS_IMAGES}.empty_camera_{i}"
            for i in range(getattr(config, "empty_cameras", 0))
        }
        available = [name for name in config.image_features if name not in empty]
        cameras = available if args.cameras is None else args.cameras
        if not cameras or not set(cameras) <= set(available):
            raise SystemExit(f"Choose --cameras from {available}.")
        # `_prepare_images` stacks the cameras in the policy's order, so the frame keeps that order.
        self.frame = {
            name: x
            for name, x in frame_from_policy(config).items()
            if name not in config.image_features or name in cameras
        }
        self.noise = torch.randn(
            config.chunk_size, self.policy.model.dim_action, device=device()
        )
        self.expected_actions = self.rollout_actions()

    def parts(self) -> Iterator[Part]:
        preprocessor, postprocessor = processors(
            self.policy.config, self.args.policy_path
        )
        tokenizer = next(
            s for s in preprocessor.steps if isinstance(s, TokenizerProcessorStep)
        )
        domain = next(
            s for s in preprocessor.steps if isinstance(s, XVLAAddDomainIdProcessorStep)
        )
        tokens = PolicyProcessorPipeline(steps=[tokenizer])(self.observation())[
            OBS_LANGUAGE_TOKENS
        ]
        # The program converts and normalizes the images itself, without the steps' value checks.
        skipped = (
            TokenizerProcessorStep,
            XVLAImageToFloatProcessorStep,
            XVLAImageNetNormalizeProcessorStep,
        )
        tensor_steps = PolicyProcessorPipeline(
            steps=[
                s
                for s in preprocessor.steps
                if not isinstance(s, skipped + (XVLAAddDomainIdProcessorStep,))
            ]
        )
        encode = XVLAEncode(self.policy, tensor_steps, list(self.frame), tokens).to(
            device()
        )
        denoise = XVLADenoise(self.policy, postprocessor, domain.domain_id).to(device())
        frame = tuple(torch.from_numpy(x).to(device()) for x in self.frame.values())
        if not self.args.split:
            yield Part(
                "model",
                XVLAChunk(encode, denoise),
                (*frame, self.noise),
                [*self.frame, NOISE],
                [ACTION],
            )
            return
        encoded_names = ["vlm_features", "aux_visual_inputs", "proprio"]
        with torch.no_grad():
            encoded = encode(*frame)
        yield Part("encode", encode, frame, list(self.frame), encoded_names)
        yield Part(
            "denoise",
            denoise,
            (*encoded, self.noise),
            [*encoded_names, NOISE],
            [ACTION],
        )

    def observation(self) -> dict:
        return prepare_observation_for_inference(
            dict(self.frame), device(), self.args.task
        )

    def rollout_actions(self):
        """The chunk the PyTorch policy computes for the test frame, task and noise, as `lerobot-rollout` does."""
        preprocessor, postprocessor = processors(
            self.policy.config, self.args.policy_path
        )
        self.policy.reset()
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(
                preprocessor(self.observation()), noise=self.noise[None]
            )
            actions = postprocessor(actions[:, : self.policy.config.n_action_steps])
        return actions[0].float().cpu().numpy()

    def write(self, programs: dict[str, dict]) -> None:
        folder_config = copy(self.policy.config)
        folder_config.input_features = {
            name: folder_config.input_features[name] for name in self.frame
        }
        write_folder(
            self.args,
            folder_config,
            self.frame,
            self.noise,
            self.expected_actions,
            programs,
            task=self.args.task,
        )
