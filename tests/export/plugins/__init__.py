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

"""Plugins for the export tests. Each folder holds a module and the record of an installed package of the same
name, so that `lerobot-export`, which looks for plugins among the installed packages, finds it once the folder
is on `PYTHONPATH`.
"""

import os
from pathlib import Path

import pytest


def use_plugins(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Let the export's steps, which run in processes of their own, find the test plugins `names`."""
    paths = [str(Path(__file__).parent / name) for name in names]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [*paths, os.environ.get("PYTHONPATH")])))
