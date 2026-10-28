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

"""What each export backend tells the user to install."""

import email.message
import importlib.metadata

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from lerobot.export import backends  # noqa: F401  # registers the backends
from lerobot.export.configs import ExportBackendConfig

# LeRobot's own, not the ones tests and plugins register.
BACKENDS = {
    name: backend
    for name, backend in ExportBackendConfig.get_known_choices().items()
    if backend.__module__.startswith("lerobot.")
}


def _brought_by(requirement: Requirement) -> set[str] | None:
    """The packages that installing `requirement` brings, its own dependencies and those of its extras, read from
    its installed metadata; `None` if it is not installed, or is installed without one of the extras."""
    try:
        metadata = importlib.metadata.metadata(requirement.name)
    except importlib.metadata.PackageNotFoundError:
        return None
    if not requirement.extras <= set(metadata.get_all("Provides-Extra") or []):
        return None
    extras = ("", *requirement.extras)
    return {
        canonicalize_name(dependency.name)
        for dependency in map(Requirement, metadata.get_all("Requires-Dist") or [])
        if dependency.marker is None or any(dependency.marker.evaluate({"extra": extra}) for extra in extras)
    }


def _providers(module: str) -> set[str]:
    """The packages that can provide `module`, read for its top-level name from the installed ones."""
    top = module.partition(".")[0]
    return {canonicalize_name(name) for name in importlib.metadata.packages_distributions().get(top, [top])}


def _not_brought(packages: tuple[str, ...], modules: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """The modules no package of `packages` brings, and the packages whose dependencies could not be read."""
    requirements = [Requirement(package) for package in packages]
    brought = {canonicalize_name(requirement.name) for requirement in requirements}
    unread = []
    for requirement in requirements:
        if requirement.extras:
            dependencies = _brought_by(requirement)
            if dependencies is None:
                unread.append(requirement.name)
            else:
                brought |= dependencies
    return [module for module in modules if not _providers(module) & brought], unread


# Torch-TensorRT, whose extras bring the TensorRT runtime import, is installed only where exports run.
@pytest.mark.executorch
@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_the_install_command_brings_each_module_a_bundle_imports(name):
    """A bundle imports its backend's `runtime_imports` before its programs load, so the install command must
    bring them, by name or through an extra of a package it names, or the export compiles and then its own
    check fails on the missing module."""
    backend = BACKENDS[name]
    missing, unread = _not_brought(backend.packages, backend.runtime_imports)
    if missing and unread:
        pytest.skip(f"{', '.join(unread)} is not installed with its extras, so what they bring is unknown")
    assert not missing


def test_the_check_reads_dependencies_without_a_marker_and_modules_inside_a_package(monkeypatch):
    def metadata(name):
        if name != "accel":
            raise importlib.metadata.PackageNotFoundError(name)
        message = email.message.Message()
        message["Provides-Extra"] = "runtime"
        message["Requires-Dist"] = "accel-kernels"
        message["Requires-Dist"] = 'accel-loader; extra == "runtime"'
        message["Requires-Dist"] = 'accel-docs; extra == "docs"'
        return message

    monkeypatch.setattr(importlib.metadata, "metadata", metadata)
    monkeypatch.setattr(importlib.metadata, "packages_distributions", dict)
    modules = ("accel_kernels", "accel_loader.plugin", "executorch.kernels.quantized")
    assert _not_brought(("accel[runtime]", "executorch>=1.6"), modules) == ([], [])
    assert _not_brought(("accel[runtime]",), ("accel_docs",)) == (["accel_docs"], [])
    assert _not_brought(("accel[docs, other]",), ("accel_docs",)) == (["accel_docs"], ["accel"])
