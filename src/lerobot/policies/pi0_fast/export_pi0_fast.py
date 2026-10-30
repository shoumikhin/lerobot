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

"""pi0-FAST is refused by lerobot-export, before its checkpoint loads.

pi0-FAST generates its actions as text tokens and turns them into numbers with the FAST tokenizer in Python: a
byte-pair decode, then an inverse discrete cosine transform. A program cannot run that decoding, so an export would
fail inside the policy, or on a checkpoint that does not start its answer with "Action:", on that assertion.
"""

from __future__ import annotations

from lerobot.export.configs import ExportConfig
from lerobot.export.recipe import ActionChunkRecipe

REASON = (
    "pi0-FAST cannot be exported: it decodes its actions from generated tokens with the FAST tokenizer in Python "
    "(byte-pair decoding and an inverse DCT), which a compiled program cannot run."
)


class PI0FastExportRecipe(ActionChunkRecipe):
    """Refuses every pi0-FAST checkpoint with the reason above."""

    @classmethod
    def load(cls, cfg: ExportConfig) -> PI0FastExportRecipe:
        raise ValueError(REASON)
