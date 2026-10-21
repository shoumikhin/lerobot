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

"""Stand-ins shared by the export engine and rollout tests."""

from pathlib import Path

import numpy as np


def write_export_folder(folder: Path, expected_actions) -> None:
    import json

    import numpy as np
    from safetensors.numpy import save_file

    frame = {"observation.state": np.arange(3, dtype=np.float32)}
    save_file(
        {**frame, "expected_actions": np.asarray(expected_actions, np.float32)}, folder / "case.safetensors"
    )
    info = {
        "route": "fake",
        "programs": {"model": {"file": "model.bin", "inputs": ["observation.state"], "outputs": ["action"]}},
        "inputs": ["observation.state"],
        "test_case": "case.safetensors",
        "tolerance": 0.5,
    }
    (folder / "export.json").write_text(json.dumps(info))


def fake_program(*inputs):
    """A program whose chunk is the state, then doubled: two actions of three joints."""
    state = inputs[0]
    return np.stack([state, 2 * state])
