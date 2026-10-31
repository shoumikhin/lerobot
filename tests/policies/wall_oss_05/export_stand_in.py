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

"""A stand-in Wall-OSS 0.5 for its export recipe, on its tokenizer step with a tiny local tokenizer."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.common.qwen_vision import is_preparing_image_grid  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.policies.utils import is_exporting  # noqa: E402
from lerobot.policies.wall_oss_05.configuration_wall_oss_05 import WallOSS05Config  # noqa: E402
from lerobot.policies.wall_oss_05.export_wall_oss_05 import WallOSS05ExportRecipe  # noqa: E402
from lerobot.policies.wall_oss_05.modeling_wall_oss_05 import WallOSS05Policy  # noqa: E402
from lerobot.policies.wall_oss_05.processor_wall_oss_05 import (  # noqa: E402
    WALL_OSS_ACTION_TOKEN,
    WALL_OSS_PROPRIO_TOKEN,
    WallOSS05PadStateProcessorStep,
    WallOSS05TaskPassthrough,
    WallOSS05TokenizerStep,
)
from lerobot.processor import (  # noqa: E402
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyProcessorPipeline,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

CHUNK = 2
SPECIAL = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
WORDS = [
    "system",
    "You",
    "are",
    "a",
    "helpful",
    "assistant",
    ".",
    "user",
    "Observation",
    ":",
    "front",
    "view",
]
WORDS += ["right", "wrist", "Instruction", "pick", "up", "the", "cube", "Predict", "next", "action", "in"]
WORDS += ["robot", "Proprioception", "-"]


def make_tokenizer(path: Path, split_words: bool):
    """A tokenizer that writes each digit of a number as its own token, as Qwen's does."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    special = [*SPECIAL, WALL_OSS_PROPRIO_TOKEN, WALL_OSS_ACTION_TOKEN]
    names = ["<|endoftext|>", "<unk>", *special, *WORDS, *map(str, range(10))]
    tokenizer = Tokenizer(models.WordLevel({name: i for i, name in enumerate(names)}, unk_token="<unk>"))
    if split_words:
        digits = pre_tokenizers.Digits(individual_digits=True)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), digits])
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<|endoftext|>",
        additional_special_tokens=special,
    ).save_pretrained(path)
    return AutoTokenizer.from_pretrained(path)


class PromptReader(WallOSS05Policy):
    """A stand-in Wall-OSS whose actions read the prompt's tokens in order, the pixels, the state and the noise.

    Like the real one, it exports only on an image grid an eager call prepared.
    """

    def __init__(self, config):
        PreTrainedPolicy.__init__(self, config)
        self.grid_prepared = False

    def predict_action_chunk(self, batch, *, noise=None):
        if is_preparing_image_grid():
            self.grid_prepared = True
        if is_exporting() and not self.grid_prepared:
            raise RuntimeError("Cannot export on a fixed image grid: no image grid was prepared.")
        if noise is None:
            noise = torch.randn(1, CHUNK, self.config.max_action_dim, device=batch[OBS_STATE].device)
        tokens, mask = batch["input_ids"], batch["attention_mask"]
        # Weighted by position among the prompt's tokens: left padding does not move it, a token out of place does.
        positions = mask.cumsum(-1) * mask
        reads = (tokens * positions).sum().float() * 1e-6 + batch["moe_token_types"].sum() * 1e-3
        reads = reads + batch["pixel_values"].mean() + batch[OBS_STATE].mean()
        return (noise + reads)[..., : self.config.output_features[ACTION].shape[0]]


def make_recipe(tmp_path, device="cpu", split_words=True, size=448) -> WallOSS05ExportRecipe:
    torch.manual_seed(0)
    from transformers import Qwen2VLImageProcessor

    cameras = ["observation.images.face_view", "observation.images.right_wrist_view"]
    config = WallOSS05Config(
        device=device,
        recipe=None,
        vlm_config={"model_type": "qwen2_5_vl"},
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            **{camera: PolicyFeature(type=FeatureType.VISUAL, shape=(3, size, size)) for camera in cameras},
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    tokenizer = WallOSS05TokenizerStep(
        processor_name="unused",
        camera_key_mapping=config.camera_key_mapping,
        chunk_size=CHUNK,
        max_state_dim=config.max_state_dim,
        action_dim=3,
    )
    tokenizer._processor = SimpleNamespace(
        tokenizer=make_tokenizer(tmp_path / "tokenizer", split_words),
        # Explicit: in transformers 5.5.4, min_pixels/max_pixels in another test change the class default.
        image_processor=Qwen2VLImageProcessor(
            size={"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
        ),
    )
    preprocessor = PolicyProcessorPipeline(
        steps=[
            AddBatchDimensionProcessorStep(),
            WallOSS05TaskPassthrough(),
            WallOSS05PadStateProcessorStep(max_state_dim=config.max_state_dim),
            tokenizer,
            DeviceProcessorStep(device=device),
        ]
    )
    postprocessor = PolicyProcessorPipeline(
        steps=[DeviceProcessorStep(device="cpu")],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device=device), seed=0, dataset=None, task="pick up the cube", tolerance=None
    )
    return WallOSS05ExportRecipe(PromptReader(config).eval(), preprocessor, postprocessor, cfg)
