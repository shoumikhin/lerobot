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
import importlib.util
import logging
import sys
from collections.abc import Collection, Iterable, Mapping

from .manifest import Target

logger = logging.getLogger(__name__)

# The CUDA driver API attributes of a GPU's compute capability.
_COMPUTE_CAPABILITY_MAJOR = 75
_COMPUTE_CAPABILITY_MINOR = 76
# What `cuInit` returns when the driver works but this process may use no GPU.
_CUDA_ERROR_NO_DEVICE = 100


def runtime_versions(modules: Iterable[str]) -> dict[str, str]:
    """The version of the installed package that provides each of the Python `modules`, for those that have one."""
    # One scan for all the modules: a scan reads every installed package, which takes a good part of a second.
    distributions = importlib.metadata.packages_distributions()
    return {
        module: importlib.metadata.version(found[0])
        for module in modules
        if (found := distributions.get(module.split(".")[0]))
    }


def check_runtime_versions(recorded: Mapping[str, str], exact: Collection[str] = ()) -> None:
    """Refuse a bundle whose runtime modules are missing, and warn about ones of another version.

    A module in `exact` of another version is refused, since the programs load only at the version that built
    them.
    """
    versions = runtime_versions(recorded)
    for module, version in recorded.items():
        installed = versions.get(module)
        if installed is None and importlib.util.find_spec(module) is not None:
            # A module can import without a package that gives its version, such as a source tree on the path.
            logger.warning(
                "This bundle was exported with %s %s. No installed package here gives the version of the %s "
                "module, so LeRobot cannot check it.",
                module,
                version,
                module,
            )
        elif installed is None:
            raise ImportError(
                f"This bundle was exported with {module} {version}, which is not installed. Install it the "
                "way the export guide says for the backend that wrote the bundle: "
                "https://huggingface.co/docs/lerobot/export"
            )
        if installed != version and module in exact:
            raise RuntimeError(
                f"This bundle was exported with {module} {version}, and this machine has {installed}. Its "
                f"programs load only with {module} {version}: install it, or export the policy again on this "
                "machine."
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
        driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
    except OSError:
        return []
    initialized = driver.cuInit(0)
    if initialized == _CUDA_ERROR_NO_DEVICE:
        return []
    _check(initialized)
    count = ctypes.c_int()
    _check(driver.cuDeviceGetCount(ctypes.byref(count)))
    gpus = []
    for index in range(count.value):
        device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        _check(driver.cuDeviceGet(ctypes.byref(device), index))
        _check(driver.cuDeviceGetAttribute(ctypes.byref(major), _COMPUTE_CAPABILITY_MAJOR, device))
        _check(driver.cuDeviceGetAttribute(ctypes.byref(minor), _COMPUTE_CAPABILITY_MINOR, device))
        _check(driver.cuDeviceGetName(name, len(name), device))
        gpus.append((name.value.decode(), f"{major.value}.{minor.value}"))
    return gpus


def _check(result: int) -> None:
    """Refuse a failed call to the NVIDIA driver, whose `result` is a CUDA error code."""
    if result != 0:
        raise RuntimeError(
            f"The NVIDIA driver failed with CUDA error {result} while LeRobot read this machine's GPUs. Check "
            "the NVIDIA driver and CUDA_VISIBLE_DEVICES."
        )


def check_target(target: Target | None) -> None:
    """Refuse a bundle built for a device this machine does not have.

    A bundle that records no device (`None`) is not checked.
    """
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
    if name != target.gpu:
        # Programs of the same compute capability usually load here, but TensorRT promises an engine only on
        # the GPU model that built it, so the test case check that runs before the robot moves decides.
        logger.warning(
            "This bundle was built on %s, and the GPU here is %s, of the same compute capability. Its test case "
            "checks the actions before the robot moves; export the policy on this machine if they differ.",
            target.gpu,
            name,
        )
