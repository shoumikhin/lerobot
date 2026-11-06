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

"""Write the files of a bundle that the trace step makes, beside its programs and manifest."""

from __future__ import annotations

import json
import ntpath
import os
import shutil
from copy import copy
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
from huggingface_hub.constants import CONFIG_NAME
from safetensors.numpy import save
from safetensors.torch import save as save_torch

from lerobot.bundle.manifest import NOISE, POLICY_CONFIG, Bundle
from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION

from .recipe import ActionTokenData, TaskTokenData, TestCase


def write_bundle(
    bundle: Bundle,
    test_case: TestCase,
    config: PreTrainedConfig,
    constants: dict[str, np.ndarray] | None = None,
    task_tokens: TaskTokenData | None = None,
    action_tokens: ActionTokenData | None = None,
) -> None:
    """Write the test case, the constants, the task's and the actions' files and the policy config into the bundle.

    Args:
        bundle (`Bundle`):
            The bundle folder and the manifest it will hold.
        test_case (`TestCase`):
            The frame, noise and actions the programs must reproduce.
        config (`PreTrainedConfig`):
            The policy config, naming the frame the programs take.
        constants (`dict[str, np.ndarray]`, *optional*):
            Named arrays the programs read, written to the file the manifest names in `constants`.
        task_tokens (`TaskTokenData`, *optional*):
            The tokenizer, embedding table and known tasks, written to the files the manifest names in
            `task_tokens`.
        action_tokens (`ActionTokenData`, *optional*):
            The FAST tokenizer, written to the file the manifest names in `action_tokens`.
    """
    folder, manifest = bundle.path, bundle.manifest
    case = {**test_case.frame, ACTION: test_case.actions}
    if test_case.noise is not None:
        case[NOISE] = test_case.noise
    # safetensors writes an array's memory in order, which scrambles a non-contiguous one. Its save_file makes
    # files only their owner can read, so the bytes are written here, and another user can run the bundle.
    (folder / manifest.test_case).write_bytes(
        save({name: np.ascontiguousarray(value) for name, value in case.items()})
    )
    if manifest.constants is not None:
        (folder / manifest.constants).write_bytes(
            save({name: np.ascontiguousarray(value) for name, value in (constants or {}).items()})
        )
    if manifest.task_tokens is not None and task_tokens is not None:
        shutil.copyfile(task_tokens.tokenizer, folder / manifest.task_tokens.tokenizer)
        (folder / manifest.task_tokens.embeddings).write_bytes(
            save_torch({"embeddings": task_tokens.embeddings.detach().cpu().contiguous()})
        )
        (folder / manifest.task_tokens.known_tasks).write_text(json.dumps(task_tokens.known_tasks))
    if manifest.action_tokens is not None and action_tokens is not None:
        (folder / manifest.action_tokens.tokenizer).write_text(action_tokens.tokenizer, encoding="utf-8")
    # Without the exporter's checkpoint path or another absolute path, such as a local copy of a model or a folder
    # of the machine that trained the policy, which can name a user. Running a bundle reads none of them.
    config = copy(config)
    config.pretrained_path = None
    for name, value in vars(config).items():
        setattr(config, name, _without_absolute_paths(value))
    config.save_pretrained(folder)
    (folder / CONFIG_NAME).rename(folder / POLICY_CONFIG)


def _without_absolute_paths(value: Any) -> Any:
    """`value` with each absolute path in it, also in a list, tuple or dict, cut to its last part.

    A Windows path counts too, since a checkpoint can be trained on Windows and exported elsewhere.
    """
    if isinstance(value, str | Path) and (os.path.isabs(value) or ntpath.isabs(value)):
        return type(value)(PureWindowsPath(value).name)
    if isinstance(value, list | tuple):
        return type(value)(_without_absolute_paths(item) for item in value)
    if isinstance(value, dict):
        return {key: _without_absolute_paths(item) for key, item in value.items()}
    return value
