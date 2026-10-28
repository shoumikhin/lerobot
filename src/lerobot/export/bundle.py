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

import json
import posixpath
from copy import copy
from pathlib import PureWindowsPath
from typing import Any

import numpy as np
from huggingface_hub.constants import CONFIG_NAME
from safetensors.numpy import save

from lerobot.bundle.manifest import (
    ACTIONS_MASK,
    CONSTANTS,
    CONTINUOUS_ACTIONS,
    NOISE,
    POLICY_CONFIG,
    TEST_CASE,
    Bundle,
)
from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION

from .recipe import TestCase


def write_bundle(
    bundle: Bundle,
    test_case: TestCase,
    config: PreTrainedConfig,
    constants: dict[str, np.ndarray] | None = None,
) -> None:
    """Write the test case, the constants and the policy config into the bundle.

    Args:
        bundle (`Bundle`):
            The bundle folder and the manifest it will hold.
        test_case (`TestCase`):
            The frame, noise and actions the programs must reproduce.
        config (`PreTrainedConfig`):
            The policy config, naming the frame the programs take.
        constants (`dict[str, np.ndarray]`, *optional*):
            Named arrays the programs read, written to `constants.safetensors` when there are any.
    """
    folder = bundle.path
    case = {**test_case.frame, ACTION: test_case.actions}
    if test_case.noise is not None:
        case[NOISE] = test_case.noise
    if test_case.continuous_actions is not None and test_case.actions_mask is not None:
        case[CONTINUOUS_ACTIONS] = test_case.continuous_actions
        case[ACTIONS_MASK] = test_case.actions_mask
    # Contiguous, since safetensors writes an array's memory in order and so scrambles a non-contiguous one;
    # np.require, since ascontiguousarray makes a 0-d array 1-d. safetensors' save_file makes files only their
    # owner can read, so the bytes are written here, and another user can run the bundle.
    (folder / TEST_CASE).write_bytes(
        save({name: np.require(value, requirements="C") for name, value in case.items()})
    )
    if constants:
        (folder / CONSTANTS).write_bytes(
            save({name: np.require(value, requirements="C") for name, value in constants.items()})
        )
    # Without the exporter's checkpoint path, and with each other absolute path, such as a model's local copy
    # or a folder of the machine that trained the policy, cut to its last part, since the folders above it can
    # name a user. Running a bundle reads none of them.
    # Cut in the saved file, which also holds the configs a config nests, such as a backbone's.
    config = copy(config)
    config.pretrained_path = None
    config.save_pretrained(folder)
    saved = json.loads((folder / CONFIG_NAME).read_text())
    (folder / POLICY_CONFIG).write_text(json.dumps(_without_absolute_paths(saved), indent=4))
    (folder / CONFIG_NAME).unlink()


def _without_absolute_paths(value: Any) -> Any:
    """`value`, read from JSON, with each string that is an absolute path, at any depth, cut to its last part.

    A path of either system counts, since a checkpoint can be trained on one and exported on the other.
    """
    # A root, not `ntpath.isabs`, whose answer for a Windows path without a drive changed in Python 3.13.
    if isinstance(value, str) and (posixpath.isabs(value) or PureWindowsPath(value).root):
        return PureWindowsPath(value).name
    if isinstance(value, list):
        return [_without_absolute_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _without_absolute_paths(item) for key, item in value.items()}
    return value
