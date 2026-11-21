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

"""Export NAT VLA as one program: its prompt made once, its action tokens generated in a fixed loop.

NAT VLA's tokenizer step writes the state into the prompt, one state-bin token per value, and builds the prompt and
the image inputs with LFM2-VL's processor in Python, so a program cannot run it. The recipe wraps that step in
[`FixedPromptStep`]. Its per-frame inputs are computed with tensor operations exactly as the step computes them:

- the token ids: the prompt of the first eager frame, with the state's tokens written in from a table (see
  [`StatePromptTokens`]). They are added tokens, so the rest of the prompt does not depend on the state.
- the pixel values: from the processor's image processor, which runs in torch.

The model runs its vision tower on the patch grid of the first eager frame, as Python ints (see [`_ImageGrid`]).

The policy generates its action tokens with `generate` in a background thread and decodes each NAT latent as its
tokens arrive. A program can do neither, so it generates the tokens of the steps `lerobot-rollout` plays in a loop of
fixed length with a KV cache, each the action token with the highest score, then decodes them all with the NAT
tokenizer. `generate` can only pick action tokens and stops after that many, so it never ends a chunk early, and the
recipe checks that the loop generates its tokens exactly on two frames. The bundle so returns each chunk once all of
it is decoded, and does not start a chunk from the tokens of the last one (`rtc_steps`).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from lerobot.export.recipe import ActionChunkRecipe, Part, random_frame
from lerobot.processor import EnvTransition, PolicyProcessorPipeline, TransitionKey
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import _transformers_available

from ..common.export_utils import is_non_strict_exporting
from ..common.fixed_prompt import FixedPromptStep
from ..common.state_prompt import StatePromptTokens
from .processor_nat_vla import NATVLATokenizerProcessorStep, Vocabulary, state_bin_indices

if TYPE_CHECKING or _transformers_available:
    from transformers.masking_utils import create_bidirectional_mask
    from transformers.modeling_outputs import BaseModelOutputWithPooling

if TYPE_CHECKING:
    import numpy as np

    from lerobot.export.configs import ExportConfig
    from lerobot.policies import PreTrainedPolicy

    from .configuration_nat_vla import NATVLAConfig
    from .modeling_nat_vla import NATVLAPolicy


class NATVLAExportRecipe(ActionChunkRecipe):
    """[`ActionChunkRecipe`] with NAT VLA's tokenizer step made traceable and its tokens generated in a fixed loop."""

    _grid: _ImageGrid | None = None

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline,
        postprocessor: PolicyProcessorPipeline,
        cfg: ExportConfig,
    ) -> None:
        """Refuse a policy that starts each chunk from the last one's tokens, then make the test case."""
        rtc_steps = cast("NATVLAConfig", policy.config).rtc_steps
        if rtc_steps:
            raise ValueError(
                f"This NAT VLA checkpoint starts each chunk from the tokens of the last {rtc_steps} actions of the "
                "one before, generated while those actions run, and a bundle computes each chunk from its frame "
                "alone: set --policy.rtc_steps=0."
            )
        super().__init__(policy, preprocessor, postprocessor, cfg)

    def program_preprocessor(self) -> PolicyProcessorPipeline:
        """The saved preprocessor with its tokenizer step wrapped in [`FixedPromptStep`]."""
        steps = [
            FixedPromptStep(step, _FrameInputs(step))
            if isinstance(step, NATVLATokenizerProcessorStep)
            else step
            for step in self.preprocessor.steps
        ]
        return replace(self.preprocessor, steps=steps)

    def predict_chunk(self, policy: PreTrainedPolicy, batch: dict[str, Any], noise: Tensor | None) -> Tensor:
        """The actions of the tokens [`action_tokens`] generates for the steps rollout plays."""
        nat_vla = cast("NATVLAPolicy", policy)
        tokens = action_tokens(nat_vla, batch, self._steps, self._image_features(nat_vla, batch))
        return nat_vla.action_tokenizer.decode(tokens, nat_vla.action_dim)

    def parts(self) -> Iterator[Part]:
        """The default recipe's part, after checking that it generates the policy's tokens for two frames."""
        for frame in (self._case.frame, random_frame(self._features, self._generator)):
            self._check_tokens(frame)
        yield from super().parts()

    def _check_tokens(self, frame: dict[str, np.ndarray]) -> None:
        """Refuse a loop that generates other tokens than the policy's `generate` for `frame`."""
        policy = cast("NATVLAPolicy", self.policy)
        self._reset()
        with torch.no_grad():
            batch = self.preprocessor(self._observation(frame, self.task))
            stream = policy.stream_action_chunk(batch, self._steps)
            stream.thread.join()
            if stream.error is not None:
                raise stream.error
            # A stream that ended without an error holds every token it generated.
            expected = cast(Tensor, stream.decoder.tokens)[0].tolist()
            features = self._image_features(policy, batch)
            tokens = action_tokens(policy, batch, self._steps, features)[0].tolist()
        if tokens != expected:
            first = next(i for i, (a, b) in enumerate(zip(tokens, expected, strict=True)) if a != b)
            raise ValueError(
                f"The NAT VLA program generates action token {tokens[first]} where the policy generates "
                f"{expected[first]}, at position {first} of {len(expected)}, so it cannot be exported this way."
            )

    def _image_features(self, policy: NATVLAPolicy, batch: dict[str, Any]) -> list[Tensor]:
        """The images' features on the patch grid of the first eager batch."""
        if self._grid is None:
            self._grid = _ImageGrid(policy, batch)
        return self._grid(policy, batch)


def action_tokens(
    policy: NATVLAPolicy, batch: dict[str, Any], steps: int, image_features: list[Tensor]
) -> Tensor:
    """The NAT tokens of the first `steps` steps, `[batch, tokens]`, as the policy generates them.

    Each token is the action token with the highest score, given the prompt and the tokens before it, which the KV
    cache holds: what the policy's `generate` picks once its logits processor leaves only the action tokens.
    `image_features` are the projected features of each image, as `Lfm2VlModel.get_image_features` returns them.
    """
    vlm, first = policy.vlm, policy.vocabulary.first_action_token
    count = policy.action_tokenizer.token_count(steps, policy.action_dim)[0]
    attention_mask = batch["attention_mask"]
    images = BaseModelOutputWithPooling(pooler_output=image_features)
    inputs: dict[str, Any] = {"input_ids": batch["input_ids"], "mm_encoder_outputs": {"image": images}}
    cache, tokens = None, []
    for _ in range(count):
        output = vlm(
            **inputs, attention_mask=attention_mask, past_key_values=cache, use_cache=True, logits_to_keep=1
        )
        token = output.logits[:, -1, first:].argmax(-1)
        tokens.append(token)
        cache = output.past_key_values
        attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, :1])], 1)
        inputs = {"input_ids": (token + first)[:, None]}
    return torch.stack(tokens, 1)


class _ImageGrid:
    """`Lfm2VlModel.get_image_features` on one patch grid per image, held as Python ints.

    LFM2-VL reads each image's patch grid back from the `spatial_shapes` tensor, to resize its position embeddings
    and to cut its features out of the padded patches. `torch.export` turns such reads into sizes it does not know
    and cannot bound, and TensorRT cannot build an engine whose sizes have no upper bound. For one camera size the
    grid is the same on every frame, so it is read once from an eager batch, with the position embeddings the
    vision tower resizes to it, and every later eager batch is checked against it.
    """

    def __init__(self, policy: NATVLAPolicy, batch: dict[str, Any]) -> None:
        embeddings = policy.vlm.model.vision_tower.embeddings
        shapes = batch["spatial_shapes"]
        self.grid = [(height, width) for height, width in shapes.tolist()]
        side = embeddings.position_embedding_size
        table = embeddings.position_embedding.weight.reshape(side, side, -1)
        patches = batch["pixel_values"].shape[1]
        self.positions = embeddings.resize_positional_embeddings(table, shapes, patches)

    def __call__(self, policy: NATVLAPolicy, batch: dict[str, Any]) -> list[Tensor]:
        if not is_non_strict_exporting():
            self._check(batch)
        model = policy.vlm.model
        tower = model.vision_tower
        patch_embedding = tower.embeddings.patch_embedding
        pixel_values = policy.image_inputs(batch)["pixel_values"]
        hidden = patch_embedding(pixel_values.to(patch_embedding.weight.dtype)) + self.positions
        mask = create_bidirectional_mask(
            config=tower.config, inputs_embeds=hidden, attention_mask=batch["pixel_attention_mask"]
        )
        hidden = tower.encoder(inputs_embeds=hidden, attention_mask=mask).last_hidden_state
        hidden = tower.post_layernorm(hidden)
        features = []
        for image, (height, width) in zip(hidden, self.grid, strict=True):
            projected = model.multi_modal_projector(image[: height * width].reshape(1, height, width, -1))
            features.append(projected.reshape(-1, projected.shape[-1]))
        return features

    def _check(self, batch: dict[str, Any]) -> None:
        """Refuse images whose patch grid, or the patches the mask keeps, differ from the first batch's."""
        grid = [list(shape) for shape in self.grid]
        kept = [height * width for height, width in self.grid]
        if batch["spatial_shapes"].tolist() != grid or batch["pixel_attention_mask"].sum(1).tolist() != kept:
            raise ValueError(
                f"The images' patch grids are {batch['spatial_shapes'].tolist()}, where the export's first frame "
                f"had {grid}: the program runs on one grid, so every camera must keep its size."
            )


class _FrameInputs:
    """The token ids and the pixel values NAT VLA's tokenizer step computes from a frame."""

    def __init__(self, step: NATVLATokenizerProcessorStep) -> None:
        self.step = step
        # The processor the step builds its inputs with, which it keeps under no public name.
        self.processor = step._processor
        first_bin = Vocabulary(step.text_vocabulary_size, step.state_bins, action_tokens=0).first_state_bin
        self.state_tokens = [[first_bin + index] for index in range(step.state_bins)]
        self.prompt: StatePromptTokens | None = None

    def __call__(self, transition: EnvTransition) -> dict[str, Tensor]:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        # In uint8, as the step gives them to the image processor.
        frames = [
            (observation[key][0] * 255).round().clamp(0, 255).to(torch.uint8) for key in self.step.camera_keys
        ]
        inputs = {"pixel_values": self.processor.image_processor(frames, return_tensors="pt")["pixel_values"]}
        if OBS_STATE not in observation:
            return inputs
        bins = state_bin_indices(observation[OBS_STATE][0], self.step.state_bins)
        if self.prompt is None:
            if is_non_strict_exporting():
                raise ValueError("The prompt is made from an eager call before the program is traced.")
            ids = (self.step(transition)[TransitionKey.COMPLEMENTARY_DATA] or {})["input_ids"]
            self.prompt = StatePromptTokens.around(
                ids[0].tolist(),
                bins.tolist(),
                self.state_tokens,
                self.processor.tokenizer.pad_token_id,
                device=bins.device,
            )
        input_ids, _ = self.prompt(bins)
        return {"input_ids": input_ids[None], **inputs}
