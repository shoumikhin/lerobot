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

"""A tiny NAT VLA for its export recipe: a two-layer LFM2-VL with random weights and LFM2-VL's processor."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from transformers import AutoProcessor  # noqa: E402

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.nat_vla import NATVLAConfig, NATVLAPolicy  # noqa: E402
from lerobot.policies.nat_vla.export_nat_vla import NATVLAExportRecipe  # noqa: E402
from lerobot.policies.nat_vla.processor_nat_vla import (  # noqa: E402
    make_nat_vla_pre_post_processors,
    vlm_config_of,
)
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from tests.export.recipe_config import export_config  # noqa: E402
from tests.policies.nat_vla.test_nat_vla import (  # noqa: E402
    IDENTITY_STATS,
    TINY_VLM,
    TOKENIZER,
    tokenizer_folder,
)

CAMERA = "observation.images.front"


def make_recipe(tmp_path, device="cpu", side=64, rtc_steps=0, task="pick up the cube") -> NATVLAExportRecipe:
    """The recipe on a tiny NAT VLA whose camera images are `side` pixels square: the processor resizes 64 to 256."""
    # A folder of its own for each call, since a test may make several recipes.
    folder = Path(tempfile.mkdtemp(dir=tmp_path))
    AutoProcessor.from_pretrained("LiquidAI/LFM2-VL-450M").save_pretrained(folder / "vlm")
    (folder / "vlm" / "config.json").write_text(json.dumps(TINY_VLM))
    torch.manual_seed(0)
    config = NATVLAConfig(
        device=device,
        dtype=torch.float32,
        vlm=str(folder),
        vlm_config=vlm_config_of(folder / "vlm"),
        action_tokenizer=str(tokenizer_folder(folder / "tokenizer", TOKENIZER)),
        setup="single arm",
        control_mode="absolute joint positions",
        n_action_steps=16,
        rtc_steps=rtc_steps,
        input_features={
            CAMERA: PolicyFeature(FeatureType.VISUAL, (3, side, side)),
            OBS_STATE: PolicyFeature(FeatureType.STATE, (7,)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (7,))},
    )
    preprocessor, postprocessor = make_nat_vla_pre_post_processors(config, IDENTITY_STATS)
    cfg = export_config(task, device)
    return NATVLAExportRecipe(NATVLAPolicy(config).eval(), preprocessor, postprocessor, cfg)
