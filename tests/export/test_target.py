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

"""The checks that a bundle's programs can load here, which run before any program loads."""

import importlib.metadata
import json
from contextlib import nullcontext
from dataclasses import asdict

import pytest

from lerobot.bundle import target
from lerobot.bundle.engine import BundleEngine
from lerobot.bundle.manifest import MANIFEST, Bundle, Target
from tests.export.numpy_bundle import make_numpy_bundle
from tests.export.torch_backend import TorchTestBackend

ORIN = Target("cuda", "Orin", "8.7")


@pytest.mark.parametrize(
    ("built_for", "gpus", "check"),
    [
        # A bundle for any device needs no GPU.
        (None, [], nullcontext()),
        (ORIN, [("Orin", "8.7")], nullcontext()),
        (ORIN, [], pytest.raises(RuntimeError, match="sees none.*CUDA_VISIBLE_DEVICES")),
        (
            ORIN,
            [("NVIDIA A100", "8.0")],
            pytest.raises(
                RuntimeError, match=r"built for Orin \(compute capability 8.7\).*A100.*export the policy"
            ),
        ),
        (Target("npu", "Ascend", "1.0"), [], pytest.raises(ValueError, match="Upgrade LeRobot")),
    ],
)
def test_a_bundle_runs_only_on_the_kind_of_device_it_was_built_for(monkeypatch, built_for, gpus, check):
    monkeypatch.setattr(target, "cuda_gpus", lambda: gpus)
    with check:
        target.check_target(built_for)


def test_another_gpu_of_the_same_compute_capability_is_warned(monkeypatch, caplog):
    monkeypatch.setattr(target, "cuda_gpus", lambda: [("Orin", "8.7")])
    target.check_target(ORIN)
    assert "built on" not in caplog.text
    monkeypatch.setattr(target, "cuda_gpus", lambda: [("NVIDIA Jetson AGX Orin", "8.7")])
    target.check_target(ORIN)
    assert "built on Orin, and the GPU here is NVIDIA Jetson AGX Orin" in caplog.text


@pytest.mark.parametrize(("platform", "library"), [("linux", "libcuda.so.1"), ("win32", "nvcuda.dll")])
def test_the_gpus_come_from_the_nvidia_driver_of_the_platform(monkeypatch, platform, library):
    opened = []

    def no_driver(name):
        opened.append(name)
        raise OSError(name)

    monkeypatch.setattr(target.sys, "platform", platform)
    monkeypatch.setattr(target.ctypes, "CDLL", no_driver)
    assert target.cuda_gpus() == []
    assert opened == [library]


class FakeDriver:
    """The calls `cuda_gpus` makes to the NVIDIA driver, for GPUs of these names and compute capabilities."""

    def __init__(self, gpus, init=0, failing_attribute=None):
        self.gpus, self.init, self.failing_attribute = gpus, init, failing_attribute

    def cuInit(self, flags):  # noqa: N802
        return self.init

    def cuDeviceGetCount(self, count):  # noqa: N802
        count._obj.value = len(self.gpus)
        return 0

    def cuDeviceGet(self, device, index):  # noqa: N802
        device._obj.value = index
        return 0

    def cuDeviceGetAttribute(self, value, attribute, device):  # noqa: N802
        if attribute == self.failing_attribute:
            return 1
        major, minor = self.gpus[device.value][1].split(".")
        value._obj.value = int(major if attribute == 75 else minor)
        return 0

    def cuDeviceGetName(self, name, length, device):  # noqa: N802
        name.value = self.gpus[device.value][0].encode()
        return 0


@pytest.mark.parametrize(
    ("driver", "check"),
    [
        (FakeDriver([("Orin", "8.7"), ("NVIDIA A100", "8.0")]), nullcontext()),
        # The driver works, but CUDA_VISIBLE_DEVICES leaves this process no GPU.
        (FakeDriver([], init=100), nullcontext()),
        (FakeDriver([], init=803), pytest.raises(RuntimeError, match="CUDA error 803")),
        (
            FakeDriver([("Orin", "8.7")], failing_attribute=75),
            pytest.raises(RuntimeError, match="CUDA error 1"),
        ),
    ],
)
def test_the_gpus_are_read_from_the_driver_and_a_driver_error_is_refused(monkeypatch, driver, check):
    monkeypatch.setattr(target.ctypes, "CDLL", lambda name: driver)
    with check:
        assert target.cuda_gpus() == driver.gpus


def test_a_missing_runtime_package_is_refused_and_another_version_warned(monkeypatch, caplog):
    monkeypatch.setattr(target, "runtime_versions", lambda modules: {"executorch": "1.6.1"})
    target.check_runtime_versions({"executorch": "1.6.1"})
    assert "exported with" not in caplog.text
    target.check_runtime_versions({"executorch": "1.6.0"})
    assert "exported with executorch 1.6.0, and this machine has 1.6.1" in caplog.text
    with pytest.raises(ImportError, match="exported with not_a_module 11.4, which is not installed"):
        target.check_runtime_versions({"not_a_module": "11.4"})


def test_a_runtime_module_that_must_match_is_refused_at_another_version(monkeypatch):
    monkeypatch.setattr(
        target, "runtime_versions", lambda modules: {"executorch": "1.6.1", "tensorrt": "11.3.0.1"}
    )
    target.check_runtime_versions({"executorch": "1.6.0", "tensorrt": "11.3.0.1"}, exact=["tensorrt"])
    with pytest.raises(
        RuntimeError, match=r"exported with tensorrt 11.4.0.106, and this machine has 11.3.0.1"
    ):
        target.check_runtime_versions({"executorch": "1.6.1", "tensorrt": "11.4.0.106"}, exact=["tensorrt"])


def test_a_module_that_imports_without_a_package_is_warned_not_refused(tmp_path, monkeypatch, caplog):
    (tmp_path / "source_tree_module.py").write_text("")
    monkeypatch.syspath_prepend(tmp_path)
    target.check_runtime_versions({"source_tree_module": "1.0"})
    assert "No installed package here gives the version of the source_tree_module module" in caplog.text


def test_the_installed_packages_are_read_once_for_all_the_modules(monkeypatch):
    scans = []
    scan = importlib.metadata.packages_distributions
    monkeypatch.setattr(importlib.metadata, "packages_distributions", lambda: scans.append(1) or scan())
    target.check_runtime_versions({name: importlib.metadata.version(name) for name in ("numpy", "pytest")})
    assert len(scans) == 1


def test_the_versions_of_installed_modules_are_recorded():
    assert target.runtime_versions(["numpy", "not_a_module"]) == {
        "numpy": importlib.metadata.version("numpy")
    }


def test_the_engine_checks_the_device_before_any_program_loads(tmp_path, monkeypatch):
    bundle = make_numpy_bundle(tmp_path / "bundle")
    manifest = json.loads((bundle / MANIFEST).read_text())
    (bundle / MANIFEST).write_text(json.dumps({**manifest, "target": asdict(ORIN)}))
    (bundle / "model.npy").unlink()
    monkeypatch.setattr(target, "cuda_gpus", lambda: [])
    with pytest.raises(RuntimeError, match="sees none"):
        BundleEngine(Bundle.read(bundle))


@pytest.mark.parametrize(
    ("recorded", "check"),
    [
        (
            {"runtime_versions": {"not_a_module": "1.0"}},
            pytest.raises(ImportError, match="exported with not_a_module 1.0, which is not installed"),
        ),
        (
            {"runtime_versions": {"numpy": "0.1"}, "exact_runtime_modules": ["numpy"]},
            pytest.raises(RuntimeError, match="exported with numpy 0.1, and this machine has"),
        ),
    ],
)
def test_the_engine_checks_the_runtime_packages_before_any_program_loads(tmp_path, recorded, check):
    bundle = make_numpy_bundle(tmp_path / "bundle")
    manifest = json.loads((bundle / MANIFEST).read_text())
    (bundle / MANIFEST).write_text(json.dumps({**manifest, **recorded}))
    (bundle / "model.npy").unlink()
    with check:
        BundleEngine(Bundle.read(bundle))


@pytest.mark.parametrize(("device", "recorded"), [("cpu", None), ("cuda", ORIN)])
def test_a_backend_records_the_gpu_it_builds_on(monkeypatch, device, recorded):
    monkeypatch.setattr(TorchTestBackend, "device", device)
    monkeypatch.setattr(
        target.ctypes, "CDLL", lambda name: FakeDriver([("Orin", "8.7"), ("NVIDIA A100", "8.0")])
    )
    assert TorchTestBackend().target() == recorded
    # What the export records, the engine reads the same way on the same GPU.
    target.check_target(recorded)
