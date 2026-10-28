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

"""What the backends whose programs ExecuTorch runs have in common."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..configs import ExportBackendConfig

if TYPE_CHECKING:
    from executorch.exir import ExecutorchBackendConfig

EXECUTORCH = "executorch>=1.6.0,<1.7.0"


@dataclass
class ExecuTorchConfig(ExportBackendConfig):
    """The base of the backends that compile into ExecuTorch programs. Not a `--backend.type` itself."""

    runtime = "executorch"
    file_suffix = ".pte"
    packages = (EXECUTORCH,)


def gpu_memory_config() -> ExecutorchBackendConfig:
    """Keep a program's inputs and outputs in GPU memory, so a program reads an earlier one's outputs in place."""
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
    from executorch.exir.passes.propagate_device_pass import PropagateDeviceConfig

    return ExecutorchBackendConfig(
        propagate_device_config=PropagateDeviceConfig(
            skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
        ),
        enable_non_cpu_memory_planning=True,
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=False),
    )
