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

"""Export LingBot-VLA v2 as one program, with its vision tower on the image grid of the export's frames.

The Qwen3-VL vision tower reads the image grid on the host, which `torch.export` cannot trace, so the program
passes `predict_action_chunk` a `vision_grid` built from the grid of every camera ([`VisionGridRecipe`]).
"""

from torch import nn

from lerobot.export.recipe import VisionGridRecipe
from lerobot.policies import PreTrainedPolicy

from ..common.qwen3_vision import Qwen3VisionGrid


class LingbotVLAV2ExportRecipe(VisionGridRecipe):
    """[`VisionGridRecipe`] with LingBot-VLA v2's Qwen3-VL vision tower."""

    grid_type = Qwen3VisionGrid
    # With random weights, eager PyTorch moved its bfloat16 actions by up to 0.094 on 8 frames moved by one rounding
    # step, a correct bundle by 0.0625, and the smallest planted bug by 0.109. The margin is thin: no trained
    # checkpoint exports yet to measure it on.
    bfloat16_tolerance = 0.10

    def vision_tower(self, policy: PreTrainedPolicy) -> nn.Module:
        """The policy's Qwen vision tower."""
        return policy.model.qwenvl_with_expert.qwenvl.model.visual
