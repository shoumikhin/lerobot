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

"""ExecuTorch programs that run on a Qualcomm Hexagon NPU through QNN."""

from __future__ import annotations

import ctypes
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..configs import ExportBackendConfig
from .executorch import ExecuTorchConfig

if TYPE_CHECKING:
    from torch.export import ExportedProgram


@ExportBackendConfig.register_subclass("executorch_qnn")
@dataclass
class ExecuTorchQNNConfig(ExecuTorchConfig):
    """ExecuTorch programs whose work runs on the Hexagon NPU of a Snapdragon chip, through Qualcomm's QNN.

    Each program is built for one chip, `soc`. ExecuTorch's Linux x86_64 package runs the programs on QNN's
    simulator of that NPU, so a bundle can be exported and checked on a PC. To export and to run there, the
    libraries of Qualcomm's QNN SDK must be on `LD_LIBRARY_PATH` before Python starts: its
    `lib/x86_64-linux-clang` and LLVM 14's `lib/x86_64-unknown-linux-gnu`, whose libc++ it needs. Setting
    `QNN_SDK_ROOT` to the SDK's folder is recommended: ExecuTorch reads the SDK's version from it to choose
    how it lowers some operations. The NPU computes in float16, so a float32 policy needs a larger
    `--tolerance` than its default.
    """

    device = "cpu"

    soc: str | None = None
    """The Snapdragon chip the programs run on, such as `SM8650` (Snapdragon 8 Gen 3) or `QCS9100`; a wrong
    name gets the list of the chips it can be. Required: a program runs only on chips with the same Hexagon
    version, so name the robot's own chip. Float16 needs Hexagon v69 or newer, so older chips, such as the
    QCM6490, are refused."""

    def __post_init__(self) -> None:
        """Refuse a machine that cannot export QNN programs, or a chip they cannot run on, before tracing."""
        if sys.platform != "linux" or platform.machine() != "x86_64":
            raise ValueError(
                "executorch_qnn exports only on Linux x86_64, whose ExecuTorch package has QNN's compiler. "
                "Export on such a PC, which checks the bundle on QNN's simulator."
            )
        # Before importing ExecuTorch, which an older PyTorch may not load.
        self.check_torch()
        try:
            from executorch.backends.qualcomm.serialization.qc_schema import HtpArch
            from executorch.backends.qualcomm.utils.utils import get_soc_to_htp_arch_map
        except ImportError as error:
            raise ValueError(
                f"executorch_qnn needs ExecuTorch with its QNN backend ({error}). ExecuTorch's Linux x86_64 "
                "package has it: pip install 'lerobot[export]'. See "
                "https://huggingface.co/docs/lerobot/export."
            ) from error
        chips = sorted(chip for chip, arch in get_soc_to_htp_arch_map().items() if arch >= HtpArch.V69)
        if self.soc not in chips:
            raise ValueError(
                f"--backend.soc must be one of {chips}, the chips whose Hexagon NPU computes in float16 (v69 "
                f"or newer), not {self.soc!r}."
            )
        try:
            ctypes.CDLL("libQnnHtp.so")
        except OSError as error:
            raise ValueError(
                f"executorch_qnn cannot load Qualcomm's QNN libraries ({error}). Put the SDK's "
                "lib/x86_64-linux-clang and LLVM 14's lib/x86_64-unknown-linux-gnu on LD_LIBRARY_PATH before "
                "Python starts. See https://huggingface.co/docs/lerobot/export."
            ) from error

    def compile(self, program: ExportedProgram, path: Path) -> None:
        """Lower `program` to QNN in float16 and save it as an ExecuTorch program."""
        from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
        from executorch.backends.qualcomm.utils.utils import (
            generate_htp_compiler_spec,
            generate_qnn_executorch_compiler_spec,
            get_qnn_context_binary_alignment,
            to_edge_transform_and_lower_to_qnn,
        )
        from executorch.exir import ExecutorchBackendConfig
        from executorch.exir.passes.constant_prop_pass import constant_prop_pass

        specs = generate_qnn_executorch_compiler_spec(
            soc_model=QcomChipset[self.soc], backend_options=generate_htp_compiler_spec(use_fp16=True)
        )
        # QNN runs its own passes around its partitioner, so the generic `lower` does not apply; the values
        # computed from constants alone are folded first, as `lower` does.
        edge = to_edge_transform_and_lower_to_qnn(
            constant_prop_pass(program.run_decompositions({})), None, specs
        )
        compiled = edge.to_executorch(
            ExecutorchBackendConfig(segment_alignment=get_qnn_context_binary_alignment())
        )
        with path.open("wb") as file:
            compiled.write_to_file(file)
