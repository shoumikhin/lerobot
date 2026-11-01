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

"""LingBot-VA's export recipe: what it refuses to export."""

from types import SimpleNamespace

import pytest

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig  # noqa: E402
from lerobot.policies.lingbot_va.export_lingbot_va import LingBotVAExportRecipe  # noqa: E402


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"camera_layout": "robotwin_tshape"}, "two streaming VAEs"),
        ({"frame_chunk_size": 1}, "frame_chunk_size is 1"),
        ({"action_per_frame": 6}, "multiple of 4"),
    ],
)
def test_export_refuses_what_it_does_not_run(change, message):
    config = LingBotVAConfig(**change)
    with pytest.raises(ValueError, match=message):
        LingBotVAExportRecipe(SimpleNamespace(config=config), None, None, None)
