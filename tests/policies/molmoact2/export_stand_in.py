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

"""A stand-in MolmoAct2 for its export recipe, on MolmoAct2's preprocessor with its own image processor."""

import re
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config  # noqa: E402
from lerobot.policies.molmoact2.export_molmoact2 import MolmoAct2ExportRecipe  # noqa: E402
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy  # noqa: E402
from lerobot.policies.molmoact2.molmoact2_hf_model.image_processing_molmoact2 import (  # noqa: E402
    MolmoAct2ImageProcessor,
)
from lerobot.policies.molmoact2.processor_molmoact2 import (  # noqa: E402
    STATE_TOKEN_PREFIX,
    make_molmoact2_pre_post_processors,
)
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

CAMERA = "observation.images.front"
CHUNK = 2
SIDE = 28
BOS, IMAGE_TOKEN = 1, 2


class Tokenizer:
    """Gives each word, punctuation mark and special token its own id, and `<state_k>` the id 1000 + k."""

    pad_token_id = 0
    eos_token, eos_token_id = "<|endoftext|>", 3

    def __init__(self):
        self.ids: dict[str, int] = {}

    def convert_tokens_to_ids(self, token):
        state = re.fullmatch(rf"{STATE_TOKEN_PREFIX}(\d+)>", token)
        return 1000 + int(state[1]) if state else self.ids.setdefault(token, 10 + len(self.ids))

    def tokenize(self, text):
        return re.findall(r"<[^<>\s]+>|\w+|[^\w\s]", text)

    def encode(self, text, add_special_tokens=False):
        return [self.convert_tokens_to_ids(token) for token in self.tokenize(text)]


class Processor:
    """Like MolmoAct2's processor: its image processor's pixels, and one image token per pooled patch."""

    def __init__(self, crop_mode):
        self.tokenizer = Tokenizer()
        size = {"height": SIDE, "width": SIDE}
        self.image_processor = MolmoAct2ImageProcessor(size=size, crop_mode=crop_mode)

    def __call__(self, text, images, **kwargs):
        vision = self.image_processor(images=images, return_tensors="pt")
        [prompt] = text
        image = [IMAGE_TOKEN] * len(vision["image_token_pooling"])
        ids = [BOS]
        for token in self.tokenizer.tokenize(prompt):
            ids += image if token == "<|image|>" else [self.tokenizer.convert_tokens_to_ids(token)]
        input_ids = torch.tensor([ids])
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), **vision}


class PromptReader(MolmoAct2Policy):
    """A stand-in MolmoAct2 whose actions read the prompt, token by token, the pixels and the noise."""

    def __init__(self, config):
        PreTrainedPolicy.__init__(self, config)

    def predict_action_chunk(self, batch, *, noise=None):
        if noise is None:
            noise = torch.randn(1, CHUNK, self.config.expected_max_action_dim, device=batch[OBS_STATE].device)
        ids, mask = batch["input_ids"], batch["attention_mask"]
        positions = torch.arange(1, ids.shape[1] + 1, device=noise.device)
        # Small enough that the actions stay inside the range the postprocessor clips them to.
        reads = (ids * positions * mask).sum().float() * 1e-6 + batch["pixel_values"].mean()
        return (noise + reads)[..., : self.config.output_features[ACTION].shape[0]]


def make_recipe(
    monkeypatch, device="cpu", width=SIDE, crop_mode="resize", state_mask=False
) -> MolmoAct2ExportRecipe:
    """The recipe on a stand-in policy. `state_mask` normalizes the state by quantiles with a gripper mask, as
    every published MolmoAct2 checkpoint's statistics do."""
    processor = "lerobot.policies.molmoact2.processor_molmoact2"
    monkeypatch.setattr(f"{processor}._resolve_checkpoint_location", lambda *args, **kwargs: "unused")
    monkeypatch.setattr(f"{processor}._load_local_molmoact2_processor", lambda path: Processor(crop_mode))
    torch.manual_seed(0)
    identity = NormalizationMode.IDENTITY
    state_mode = NormalizationMode.QUANTILES if state_mask else identity
    config = MolmoAct2Config(
        device=device,
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        normalize_gripper=not state_mask,
        normalization_mapping={"VISUAL": identity, "STATE": state_mode, "ACTION": identity},
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, SIDE, width)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    stats = None
    if state_mask:
        low, high = torch.tensor([-1.0, -0.5, 0.0]), torch.tensor([1.0, 2.0, 1.0])
        # The last value is the gripper's, which the policy reads as it is.
        stats = {OBS_STATE: {"q01": low, "q99": high, "min": low, "max": high, "mask": [True, True, False]}}
    preprocessor, postprocessor = make_molmoact2_pre_post_processors(config, stats)
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device=device), seed=0, dataset=None, task="pick up the cube", tolerance=None
    )
    return MolmoAct2ExportRecipe(PromptReader(config).eval(), preprocessor, postprocessor, cfg)
