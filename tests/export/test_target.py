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

import json
from dataclasses import asdict

import pytest
import torch

from lerobot.bundle import target
from lerobot.bundle.engine import BundleEngine
from lerobot.bundle.manifest import MANIFEST, Bundle, Target
from tests.export.numpy_bundle import make_numpy_bundle
from tests.export.torch_backend import TorchTestBackend

ORIN = Target("cuda", "Orin", "8.7")


def test_a_bundle_for_any_device_needs_no_gpu(monkeypatch):
    monkeypatch.setattr(target, "cuda_gpus", lambda: [])
    target.check_target(None)


def test_a_gpu_bundle_runs_on_the_kind_of_gpu_that_built_it(monkeypatch):
    monkeypatch.setattr(target, "cuda_gpus", lambda: [("Orin", "8.7")])
    target.check_target(ORIN)


def test_a_gpu_bundle_refuses_a_machine_without_a_gpu(monkeypatch):
    monkeypatch.setattr(target, "cuda_gpus", lambda: [])
    with pytest.raises(RuntimeError, match="sees none.*CUDA_VISIBLE_DEVICES"):
        target.check_target(ORIN)


def test_a_gpu_bundle_refuses_another_kind_of_gpu(monkeypatch):
    monkeypatch.setattr(target, "cuda_gpus", lambda: [("NVIDIA A100", "8.0")])
    with pytest.raises(
        RuntimeError, match=r"built for Orin \(compute capability 8.7\).*A100.*export the policy"
    ):
        target.check_target(ORIN)


def test_a_device_this_version_cannot_check_is_refused():
    with pytest.raises(ValueError, match="Upgrade LeRobot"):
        target.check_target(Target("npu", "Ascend", "1.0"))


def test_a_missing_runtime_package_is_refused_and_another_version_warned(monkeypatch, caplog):
    monkeypatch.setattr(target, "installed_version", {"executorch": "1.6.1"}.get)
    target.check_runtime_versions({"executorch": "1.6.1"})
    assert "exported with" not in caplog.text
    target.check_runtime_versions({"executorch": "1.6.0"})
    assert "exported with executorch 1.6.0, and this machine has 1.6.1" in caplog.text
    with pytest.raises(ImportError, match="exported with tensorrt 11.4, which is not installed"):
        target.check_runtime_versions({"tensorrt": "11.4"})


def test_the_versions_of_installed_modules_are_recorded():
    assert target.runtime_versions(["numpy", "not_a_module"]) == {"numpy": target.installed_version("numpy")}


def test_the_engine_checks_the_device_before_any_program_loads(tmp_path, monkeypatch):
    bundle = make_numpy_bundle(tmp_path / "bundle")
    manifest = json.loads((bundle / MANIFEST).read_text())
    (bundle / MANIFEST).write_text(json.dumps({**manifest, "target": asdict(ORIN)}))
    (bundle / "model.npy").unlink()
    monkeypatch.setattr(target, "cuda_gpus", lambda: [])
    with pytest.raises(RuntimeError, match="sees none"):
        BundleEngine(Bundle.read(bundle))


def test_a_manifest_refuses_a_target_key_it_does_not_know(tmp_path):
    bundle = make_numpy_bundle(tmp_path / "bundle")
    manifest = json.loads((bundle / MANIFEST).read_text())
    (bundle / MANIFEST).write_text(json.dumps({**manifest, "target": {**asdict(ORIN), "memory": "8"}}))
    with pytest.raises(ValueError, match=r"uses \['memory'\].*Upgrade LeRobot"):
        Bundle.read(bundle)


@pytest.mark.parametrize(("device", "recorded"), [("cpu", None), ("cuda", ORIN)])
def test_a_backend_records_the_gpu_it_builds_on(monkeypatch, device, recorded):
    monkeypatch.setattr(TorchTestBackend, "device", device)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 7))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "Orin")
    assert TorchTestBackend().target() == recorded
