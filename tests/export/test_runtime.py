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

"""How a bundle finds its runtime, and in which order it gets the runtime ready."""

import sys
from pathlib import Path

import pytest

from lerobot.bundle import runtime as bundle_runtime
from lerobot.bundle.runtime import RUNTIMES, Runtime, prepare_runtime, register_runtime


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


def test_a_registered_runtime_prepares_before_its_modules(events):
    register_runtime("fake", _runtime(events, "fake"))
    prepare_runtime("fake", ["delegate"])
    assert events == ["prepare fake", "import delegate"]


def test_a_runtime_a_module_registers_prepares_before_it_loads(events, monkeypatch):
    def import_registering(name):
        events.append(f"import {name}")
        register_runtime("fake", _runtime(events, "fake"))

    monkeypatch.setattr(bundle_runtime.importlib, "import_module", import_registering)
    prepare_runtime("fake", ["provider"]).load(Path("model.bin"), None)
    assert events == ["import provider", "prepare fake", "load fake"]


def test_the_bundle_folder_is_off_the_search_path_until_the_runtime_is_ready(events, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    before = list(sys.path)
    register_runtime("fake", Runtime(lambda path, data: None, lambda: events.append(list(sys.path))))

    def fail(name):
        raise ImportError(name)

    monkeypatch.setattr(bundle_runtime.importlib, "import_module", fail)
    with pytest.raises(ImportError, match=r"module 'delegate', which loads its programs \(delegate\)"):
        prepare_runtime("fake", ["delegate"], folder=tmp_path)
    assert events == [before[1:]]
    assert sys.path == before


def test_an_unknown_runtime_names_the_ones_there_are(events):
    with pytest.raises(ValueError, match=r"Unknown runtime 'missing'.*\['executorch'"):
        prepare_runtime("missing", [])


def test_a_runtime_name_registers_once(events):
    with pytest.raises(ValueError, match="already registered"):
        register_runtime("executorch", Runtime(load=lambda path, data: None))
