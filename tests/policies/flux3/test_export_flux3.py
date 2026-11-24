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

"""The FLUX3 export recipe, on a tiny FLUX3 with a fake video autoencoder and text encoder."""

import pytest

from lerobot.export.backends.executorch_tensorrt import ExecuTorchTensorRTConfig
from tests.policies.flux3.export_stand_in import make_recipe


@pytest.mark.parametrize("tensorrt", [True, False], ids=["tensorrt", "other_backend"])
def test_flux3_turns_off_weight_sharing_only_for_tensorrt(monkeypatch, tensorrt):
    """Only with TensorRT, since only `executorch_tensorrt` reads compile options."""
    recipe = make_recipe(monkeypatch)
    if tensorrt:
        recipe.cfg.backend = ExecuTorchTensorRTConfig()
    [part] = recipe.parts()
    assert part.compile_options == ({"deduplicate_weights": False} if tensorrt else {})
