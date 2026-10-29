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

"""In which order a bundle gets its runtime ready."""

from pathlib import Path

import pytest

from lerobot.bundle import runtime as bundle_runtime
from lerobot.bundle.runtime import RUNTIMES, Runtime, prepare_runtime


@pytest.fixture
def events(monkeypatch):
    """What happens, in order: runtime preparation, module imports, program loads."""
    happened: list[str] = []
    saved = dict(RUNTIMES)
    monkeypatch.setattr(
        bundle_runtime.importlib, "import_module", lambda name: happened.append(f"import {name}")
    )
    yield happened
    RUNTIMES.clear()
    RUNTIMES.update(saved)


def _runtime(events: list[str], name: str) -> Runtime:
    return Runtime(
        load=lambda path, data: events.append(f"load {name}"),
        prepare=lambda: events.append(f"prepare {name}"),
    )


def test_a_runtime_lerobot_has_prepares_before_its_modules(events):
    RUNTIMES["fake"] = _runtime(events, "fake")
    prepare_runtime("fake", ["delegate"])
    assert events == ["prepare fake", "import delegate"]


def test_a_runtime_a_module_registers_prepares_before_it_loads(events, monkeypatch):
    def import_registering(name):
        events.append(f"import {name}")
        RUNTIMES["fake"] = _runtime(events, "fake")

    monkeypatch.setattr(bundle_runtime.importlib, "import_module", import_registering)
    prepare_runtime("fake", ["provider"]).load(Path("model.bin"), None)
    assert events == ["import provider", "prepare fake", "load fake"]
