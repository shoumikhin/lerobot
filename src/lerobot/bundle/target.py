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

"""What a bundle's programs were built with and for, and the checks that this machine can load them.

The checks run before any program loads, because a program built for another GPU or another version of its
runtime may fail with a message that does not say why, or crash the process.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import logging
from collections.abc import Mapping, Sequence

from .manifest import Target

logger = logging.getLogger(__name__)

# The CUDA driver API attributes of a GPU's compute capability.
_COMPUTE_CAPABILITY_MAJOR = 75
_COMPUTE_CAPABILITY_MINOR = 76


def installed_version(module: str) -> str | None:
    """The version of the installed package that provides the Python module `module`, if any."""
    distributions = importlib.metadata.packages_distributions().get(module.split(".")[0])
    return importlib.metadata.version(distributions[0]) if distributions else None


def runtime_versions(modules: Sequence[str]) -> dict[str, str]:
    """The installed version of each of `modules`, for a bundle to record."""
    return {module: version for module in modules if (version := installed_version(module)) is not None}


def check_runtime_versions(recorded: Mapping[str, str]) -> None:
    """Refuse a bundle whose runtime packages are missing, and warn about ones of another version."""
    for module, version in recorded.items():
        installed = installed_version(module)
        if installed is None:
            raise ImportError(
                f"This bundle was exported with {module} {version}, which is not installed. Install it the "
                "way the export guide says for the backend that wrote the bundle: "
                "https://huggingface.co/docs/lerobot/export"
            )
        if installed != version:
            logger.warning(
                "This bundle was exported with %s %s, and this machine has %s. If it does not load, install "
                "%s %s, or export the policy again on this machine.",
                module,
                version,
                installed,
                module,
                version,
            )


def cuda_gpus() -> list[tuple[str, str]]:
    """The name and compute capability of each GPU this process can use, in CUDA's order.

    Asks the NVIDIA driver directly, so it needs no PyTorch, and honors `CUDA_VISIBLE_DEVICES`.
    """
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return []
    count = ctypes.c_int()
    if driver.cuInit(0) != 0 or driver.cuDeviceGetCount(ctypes.byref(count)) != 0:
        return []
    gpus = []
    for index in range(count.value):
        device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        driver.cuDeviceGet(ctypes.byref(device), index)
        driver.cuDeviceGetAttribute(ctypes.byref(major), _COMPUTE_CAPABILITY_MAJOR, device)
        driver.cuDeviceGetAttribute(ctypes.byref(minor), _COMPUTE_CAPABILITY_MINOR, device)
        driver.cuDeviceGetName(name, len(name), device)
        gpus.append((name.value.decode(), f"{major.value}.{minor.value}"))
    return gpus


def check_target(target: Target | None) -> None:
    """Refuse a bundle built for a device this machine does not have. `None` runs anywhere."""
    if target is None:
        return
    if target.device != "cuda":
        raise ValueError(
            f"This bundle was built for a {target.device} device, which this version of LeRobot cannot check. "
            "Upgrade LeRobot."
        )
    gpus = cuda_gpus()
    built_for = f"{target.gpu} (compute capability {target.compute_capability})"
    if not gpus:
        raise RuntimeError(
            f"This bundle runs on an NVIDIA GPU, {built_for}, but this process sees none. Check the NVIDIA "
            "driver and CUDA_VISIBLE_DEVICES."
        )
    name, capability = gpus[0]
    if capability != target.compute_capability:
        raise RuntimeError(
            f"This bundle was built for {built_for}, but the GPU here is {name} (compute capability "
            f"{capability}). Its programs run only on the kind of GPU that built them: export the policy again "
            "on this machine."
        )
