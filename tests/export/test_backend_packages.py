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

import pytest
from packaging.requirements import Requirement

from lerobot.export import backends  # noqa: F401  # registers the backends
from lerobot.export.configs import ExportBackendConfig

# LeRobot's own, not the ones tests and plugins register.
BACKENDS = {
    name: backend
    for name, backend in ExportBackendConfig.get_known_choices().items()
    if backend.__module__.startswith("lerobot.")
}


@pytest.mark.parametrize("name", sorted(BACKENDS))
def test_the_install_command_names_the_package_of_each_module_a_bundle_imports(name):
    """A bundle imports its backend's `runtime_imports` before its programs load, so the install command must
    bring them, or the export compiles and then its own check fails on the missing module."""
    backend = BACKENDS[name]
    installed = {Requirement(package).name for package in backend.packages}
    assert {module.replace("_", "-") for module in backend.runtime_imports} <= installed
