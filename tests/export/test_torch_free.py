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

"""The parts of a bundle that a process without PyTorch loads."""

import pytest

from tests.torch_free import assert_runs_without_torch

BUNDLE_MODULES = [
    "lerobot.bundle.manifest",
    "lerobot.bundle.runtime",
    "lerobot.bundle.executorch",
    "lerobot.bundle.engine",
    "lerobot.bundle.task_tokens",
]


def test_the_bundle_modules_import_without_torch():
    """The `lerobot-export` parent process and a process that only runs a bundle load these, never PyTorch."""
    assert_runs_without_torch("\n".join(f"import {name}" for name in BUNDLE_MODULES))


def test_executorch_loads_without_torch_once_prepared():
    """ExecuTorch imports PyTorch unless told to return its own results, which the runtime's `prepare` does."""
    pytest.importorskip("executorch.runtime")
    assert_runs_without_torch(
        """
        from lerobot.bundle import executorch

        executorch.prepare()
        import executorch.runtime
        """
    )
