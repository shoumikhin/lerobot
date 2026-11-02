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

"""Loads ExecuTorch for the export tests the way a bundle does."""

import warnings

from tests.export.backends import EXECUTORCH

# ExecuTorch picks the type of its results once, when it loads, and a bundle loads it after `prepare()`.
# Called before any test runs, so a test that runs a bundle here does not depend on which test loaded
# ExecuTorch first.
if EXECUTORCH:
    from lerobot.bundle.executorch import prepare

    try:
        prepare()
    except ImportError as error:
        # A broken install fails the tests that need ExecuTorch, not the collection of every export test.
        warnings.warn(str(error), stacklevel=1)
