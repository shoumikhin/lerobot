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

import subprocess
import sys

# A script prefix that refuses to load PyTorch, for tests that check a process never imports it. The spec is
# still found, so code that only checks whether torch is installed keeps working. Each load attempt is
# recorded, so an import wrapped in try/except still fails the test.
BLOCK_TORCH = """
import importlib
import importlib.abc
import importlib.machinery
import sys
import traceback

attempts = []


class RefuseLoader(importlib.abc.Loader):
    def exec_module(self, module):
        attempts.append("".join(traceback.format_stack()))
        raise RuntimeError("torch must not be imported here")


class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "torch":
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        spec.loader = RefuseLoader()
        return spec


sys.meta_path.insert(0, BlockTorch())
"""

BUNDLE_MODULES = [
    "lerobot.bundle.manifest",
    "lerobot.bundle.runtime",
    "lerobot.bundle.executorch",
    "lerobot.bundle.task_tokens",
]


def test_the_bundle_modules_import_without_torch():
    """The `lerobot-export` parent process and a process that only runs programs load these, never PyTorch."""
    script = BLOCK_TORCH + "\n".join(
        [*(f"import {name}" for name in BUNDLE_MODULES), "assert not attempts, attempts[0]"]
    )
    subprocess.run([sys.executable, "-c", script], check=True)
