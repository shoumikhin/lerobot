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

"""Write a bundle: the compiled program files are already in the folder; this adds the rest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from lerobot.configs import PreTrainedConfig
from lerobot.inference.export.bundle import MANIFEST, NOISE, ExportManifest
from lerobot.utils.constants import ACTION

from .recipe import TestCase


def write_bundle(
    folder: Path, manifest: ExportManifest, test_case: TestCase, config: PreTrainedConfig
) -> None:
    """Write the test case, the policy config and the manifest into `folder`.

    Args:
        folder (`Path`):
            The bundle folder, which already holds the program files the manifest names.
        manifest (`ExportManifest`):
            The manifest to write as `export.json`.
        test_case (`TestCase`):
            The frame, noise and actions the programs must reproduce.
        config (`PreTrainedConfig`):
            The policy config, naming the frame the programs take.
    """
    case = {**test_case.frame, ACTION: test_case.actions}
    if test_case.noise is not None:
        case[NOISE] = test_case.noise
    # safetensors writes an array's memory in order, which scrambles a non-contiguous one.
    save_file({name: np.ascontiguousarray(value) for name, value in case.items()}, folder / manifest.test)
    config.save_pretrained(folder)
    (folder / MANIFEST).write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")
