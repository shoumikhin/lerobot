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

"""Write the files of a bundle that the trace step makes: the test case, the constants and the policy config."""

from __future__ import annotations

import numpy as np
from huggingface_hub.constants import CONFIG_NAME
from safetensors.numpy import save_file

from lerobot.bundle.manifest import NOISE, POLICY_CONFIG, Bundle
from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION

from .recipe import TestCase


def write_bundle(
    bundle: Bundle,
    test_case: TestCase,
    config: PreTrainedConfig,
    constants: dict[str, np.ndarray] | None = None,
) -> None:
    """Write the test case, the constants and the policy config under the names the bundle's manifest gives.

    Args:
        bundle (`Bundle`):
            The bundle folder and the manifest it will hold.
        test_case (`TestCase`):
            The frame, noise and actions the programs must reproduce.
        config (`PreTrainedConfig`):
            The policy config, naming the frame the programs take.
        constants (`dict[str, np.ndarray]`, *optional*):
            Named arrays the programs read, written to the file the manifest names in `constants`.
    """
    folder, manifest = bundle.path, bundle.manifest
    case = {**test_case.frame, ACTION: test_case.actions}
    if test_case.noise is not None:
        case[NOISE] = test_case.noise
    # safetensors writes an array's memory in order, which scrambles a non-contiguous one.
    save_file(
        {name: np.ascontiguousarray(value) for name, value in case.items()}, folder / manifest.test_case
    )
    if manifest.constants is not None:
        save_file(
            {name: np.ascontiguousarray(value) for name, value in (constants or {}).items()},
            folder / manifest.constants,
        )
    config.save_pretrained(folder)
    (folder / CONFIG_NAME).rename(folder / POLICY_CONFIG)
