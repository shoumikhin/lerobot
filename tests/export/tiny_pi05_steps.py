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

"""Importing this makes pi0.5 tiny for the rest of the process. The export's steps import it, so they build
the same tiny pi0.5 the test saved; the test process itself uses `use_tiny_pi05` with `monkeypatch`."""

from types import SimpleNamespace

from tests.export.tiny_pi05 import use_tiny_pi05

use_tiny_pi05(SimpleNamespace(setattr=setattr))
