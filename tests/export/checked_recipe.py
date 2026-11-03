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

"""A recipe whose part is also checked on its own. Its plugin makes it ACT's recipe, as an `export_act.py`
would, so importing this module changes nothing.

The tiny ACT is float32, which `ProgramCheck` is not meant for: a compiler that sums in another order can miss
its floor of one rounding step. The test backend reruns the very same program, so the part matches exactly.
"""

from collections.abc import Iterator
from dataclasses import replace

from lerobot.export.recipe import ActionChunkRecipe, Part, ProgramCheck


class ACTCheckedRecipe(ActionChunkRecipe):
    def parts(self) -> Iterator[Part]:
        for part in super().parts():
            yield replace(part, check=ProgramCheck.measure(part.module, part.example_inputs))


ACTExportRecipe = ACTCheckedRecipe
