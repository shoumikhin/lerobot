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

"""A recipe that makes ACT start from a noise, as a flow-matching policy does: the noise is added to its actions.
Importing this module makes it ACT's recipe, as an `export_act.py` would.
"""

import sys

import numpy as np
from torch import Tensor

from lerobot.export.recipe import ActionChunkRecipe
from lerobot.policies import PreTrainedPolicy


class ACTNoiseRecipe(ActionChunkRecipe):
    @staticmethod
    def predict_chunk(policy: PreTrainedPolicy, batch: dict, noise: Tensor | None) -> Tensor:
        actions = ActionChunkRecipe.predict_chunk(policy, batch, None)
        return actions if noise is None else actions + noise

    def _sampled_noise_shape(self, frame: dict[str, np.ndarray]) -> tuple[int, ...]:
        return (self.policy.config.chunk_size, 1)


ACTExportRecipe = ACTNoiseRecipe
sys.modules["lerobot.policies.act.export_act"] = sys.modules[__name__]
