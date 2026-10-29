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

import os
from pathlib import Path

import pytest

from tests.export.numpy_bundle import make_numpy_bundle
from tests.torch_free import assert_runs_without_torch

REPO = Path(__file__).parents[2]

BUNDLE_MODULES = [
    "lerobot.bundle.manifest",
    "lerobot.bundle.runtime",
    "lerobot.bundle.executorch",
    "lerobot.bundle.engine",
]


def test_the_bundle_modules_import_without_torch():
    """The `lerobot-export` parent process and a process that only runs a bundle load these, never PyTorch."""
    assert_runs_without_torch("\n".join(f"import {name}" for name in BUNDLE_MODULES))


@pytest.mark.executorch
def test_executorch_loads_without_torch_once_prepared():
    """ExecuTorch imports PyTorch unless told to return its own results, which the runtime's `prepare` does."""
    pytest.importorskip("executorch.runtime")
    assert_runs_without_torch(
        """
        from lerobot.bundle.runtime import prepare_runtime

        prepare_runtime("executorch", [])
        import executorch.runtime
        """
    )


def test_the_bundle_check_runs_without_torch(tmp_path, monkeypatch):
    """`python -m lerobot.bundle` loads the programs and replays the test case without PyTorch."""
    bundle = make_numpy_bundle(tmp_path / "bundle")
    # The check does not look in the current folder, so the test runtime's module comes from PYTHONPATH.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))
    assert_runs_without_torch(
        """
        import runpy
        import sys

        sys.argv = ["lerobot.bundle", sys.argv[1]]
        runpy.run_module("lerobot.bundle", run_name="__main__")
        """,
        str(bundle),
    )
