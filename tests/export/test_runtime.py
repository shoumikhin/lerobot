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

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lerobot.bundle import executorch, runtime as bundle_runtime
from lerobot.bundle.manifest import MANIFEST
from lerobot.bundle.runtime import RUNTIMES, Runtime, prepare_runtime, register_runtime
from tests.export.numpy_bundle import make_numpy_bundle


@pytest.fixture
def events(monkeypatch):
    """What happens, in order: plugin scans, runtime preparation, module imports, program loads."""
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
        configure=lambda options: events.append(f"configure {options}"),
    )


def test_a_runtime_lerobot_does_not_have_comes_from_a_runtime_package(events, monkeypatch):
    def scan(kinds):
        events.append(f"scan {kinds}")
        register_runtime("fake", _runtime(events, "fake"))

    monkeypatch.setattr(bundle_runtime, "register_third_party_plugins", scan)
    prepare_runtime("fake", ["delegate"]).load(Path("model.bin"), None)
    assert events == ["scan ('runtime',)", "prepare fake", "import delegate", "load fake"]


def test_a_runtime_lerobot_has_prepares_before_its_modules_without_a_scan(events, monkeypatch):
    monkeypatch.setattr(bundle_runtime, "register_third_party_plugins", lambda kinds: events.append("scan"))
    register_runtime("fake", _runtime(events, "fake"))
    # Its options apply after the modules import and before the programs load.
    prepare_runtime("fake", ["delegate"], {"Accelerator": {"shared": True}}).load(Path("model.bin"), None)
    assert events == [
        "prepare fake",
        "import delegate",
        "configure {'Accelerator': {'shared': True}}",
        "load fake",
    ]


def test_a_runtime_a_module_registers_prepares_before_it_loads(events, monkeypatch):
    monkeypatch.setattr(bundle_runtime, "register_third_party_plugins", lambda kinds: None)

    def import_registering(name):
        events.append(f"import {name}")
        register_runtime("fake", _runtime(events, "fake"))

    monkeypatch.setattr(bundle_runtime.importlib, "import_module", import_registering)
    prepare_runtime("fake", ["provider"], {"Accelerator": {"shared": True}}).load(Path("model.bin"), None)
    assert events == [
        "import provider",
        "prepare fake",
        "configure {'Accelerator': {'shared': True}}",
        "load fake",
    ]


def test_the_bundle_folder_is_off_the_search_path_until_the_runtime_is_ready(events, monkeypatch, tmp_path):
    monkeypatch.setattr(bundle_runtime, "register_third_party_plugins", lambda kinds: None)
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


def test_a_runtime_package_the_bundle_ships_does_not_load(tmp_path):
    """The installed runtime packages are looked for with the bundle folder off the search path."""
    bundle = make_numpy_bundle(tmp_path / "bundle")
    manifest = json.loads((bundle / MANIFEST).read_text())
    (bundle / MANIFEST).write_text(json.dumps({**manifest, "runtime": "shipped", "runtime_imports": []}))
    ran = tmp_path / "ran"
    (bundle / "lerobot_runtime_shipped.py").write_text(f"open({str(ran)!r}, 'w').close()\n")
    (bundle / "lerobot_runtime_shipped-1.0.dist-info").mkdir()
    (bundle / "lerobot_runtime_shipped-1.0.dist-info" / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: lerobot_runtime_shipped\nVersion: 1.0\n"
    )
    code = "from lerobot.bundle.engine import BundleEngine\nfrom lerobot.bundle.manifest import Bundle\n"
    code += "BundleEngine(Bundle.read('.'))"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=bundle, capture_output=True, text=True, timeout=300
    )
    assert not ran.exists()
    assert "Unknown runtime 'shipped'" in result.stderr


def test_an_unknown_runtime_names_the_ones_there_are(events, monkeypatch):
    monkeypatch.setattr(bundle_runtime, "register_third_party_plugins", lambda kinds: None)
    with pytest.raises(ValueError, match=r"Unknown runtime 'missing'.*\['executorch'.*lerobot_runtime_"):
        prepare_runtime("missing", [])


def test_a_runtime_name_registers_once(events):
    with pytest.raises(ValueError, match="already registered"):
        register_runtime("executorch", Runtime(load=lambda path, data: None))


def test_a_runtime_that_takes_no_options_refuses_a_bundle_that_sets_some(events, monkeypatch):
    monkeypatch.setattr(bundle_runtime, "register_third_party_plugins", lambda kinds: None)
    register_runtime("fake", Runtime(load=lambda path, data: None))
    with pytest.raises(ValueError, match="sets options of the 'fake' runtime, which takes none"):
        prepare_runtime("fake", [], {"Accelerator": {"shared": True}})


def test_executorch_gives_each_backend_its_options_only_when_it_has_every_backend(monkeypatch):
    calls = []
    registry = SimpleNamespace(
        is_available=lambda backend: backend != "NoSuchBackend",
        set_option=lambda backend, values: calls.append((backend, values)),
    )
    runtime = SimpleNamespace(get=lambda: SimpleNamespace(backend_registry=registry))
    monkeypatch.setitem(
        sys.modules, "executorch.runtime", SimpleNamespace(ExecuTorchResult=object, Runtime=runtime)
    )
    options = {"TensorRTBackend": {"use_shared_activation_scratch": True}, "XnnpackBackend": {"x": 2}}
    with pytest.raises(ValueError, match=r"this machine does not have: \['NoSuchBackend'\]"):
        executorch.configure({**options, "NoSuchBackend": {}})
    assert calls == []
    executorch.configure(options)
    assert calls == list(options.items())


def test_an_executorch_older_than_1_6_gets_the_install_hint(monkeypatch):
    # Its `executorch.runtime` imports, without the result type that ExecuTorch 1.6 added.
    runtime = SimpleNamespace(get=lambda: None)
    monkeypatch.setitem(sys.modules, "executorch.runtime", SimpleNamespace(Runtime=runtime))
    with pytest.raises(ImportError, match="ExecuTorch 1.6"):
        executorch.ExecuTorchProgram(Path("model.pte"))
    with pytest.raises(ImportError, match="ExecuTorch 1.6"):
        executorch.configure({"TensorRTBackend": {"use_shared_activation_scratch": True}})
