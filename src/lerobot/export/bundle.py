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
import posixpath
import shutil
from copy import copy
from pathlib import PureWindowsPath
from typing import Any

import numpy as np
import torch
from huggingface_hub.constants import CONFIG_NAME
from safetensors.numpy import save
from safetensors.torch import save_file

from lerobot.bundle.manifest import NOISE, POLICY_CONFIG, TASK_EMBEDDINGS, TASK_LENGTH, Bundle
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
    if (manifest.task_tokens is None) != (task_tokens is None):
        raise ValueError("Give write_bundle task tokens exactly when its manifest names task token files.")
    if task_tokens is not None:
        _check_task_tokens(task_tokens, manifest.task, test_case.frame)
    case = {**test_case.frame, ACTION: test_case.actions}
    if test_case.noise is not None:
        case[NOISE] = test_case.noise
    # Contiguous, since safetensors writes an array's memory in order and so scrambles a non-contiguous one;
    # np.require, since ascontiguousarray makes a 0-d array 1-d. safetensors' save_file makes files only their
    # owner can read, so the bytes are written here, and another user can run the bundle.
    (folder / manifest.test_case).write_bytes(
        save({name: np.require(value, requirements="C") for name, value in case.items()})
    )
    if manifest.constants is not None:
        (folder / manifest.constants).write_bytes(
            save({name: np.require(value, requirements="C") for name, value in (constants or {}).items()})
        )
    if manifest.task_tokens is not None and task_tokens is not None:
        shutil.copyfile(task_tokens.tokenizer, folder / manifest.task_tokens.tokenizer)
        # Written from the table without a serialized copy in memory, which for a large vocabulary is a GiB
        # or more, then given the test case's permissions.
        embeddings = folder / manifest.task_tokens.embeddings
        save_file({"embeddings": task_tokens.embeddings.detach().cpu().contiguous()}, embeddings)
        shutil.copymode(folder / manifest.test_case, embeddings)
        (folder / manifest.task_tokens.known_tasks).write_text(json.dumps(task_tokens.known_tasks))
    if manifest.action_tokens is not None and action_tokens is not None:
        (folder / manifest.action_tokens.tokenizer).write_text(action_tokens.tokenizer, encoding="utf-8")
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


def _check_task_tokens(task_tokens: TaskTokenData, task: str | None, frame: dict[str, np.ndarray]) -> None:
    """Refuse task tokens that a bundle would first fail on when it runs, before any part is compiled."""
    if task not in task_tokens.known_tasks:
        raise ValueError(
            f"The recipe's known tasks do not hold the exported task {task!r}, so the bundle would need the "
            "tokenizers package to run its own task."
        )
    if given := sorted({TASK_EMBEDDINGS, TASK_LENGTH} & set(frame)):
        raise ValueError(
            f"The recipe's test case frame holds {given}, which the bundle makes from the task. The frame holds "
            "only what the robot sends."
        )
    if task_tokens.embeddings.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"The recipe's embedding table is {task_tokens.embeddings.dtype}; a bundle reads float32 or bfloat16."
        )


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
