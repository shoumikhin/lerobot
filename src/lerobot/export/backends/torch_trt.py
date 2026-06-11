# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""torch-trt AOTI backend stub.

Phase 2 TODO: implement TRT-engine export using torch_tensorrt.

Requires a separate export venv with torch 2.12+ and torch-tensorrt 2.12.
See src/lerobot/export/requirements-export-trt.txt for setup instructions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.export.adapters.base import ExportAdapter
    from lerobot.export.cli import ExportConfig


def run_torch_trt_export(adapter: ExportAdapter, cfg: ExportConfig, output_dir: Path) -> None:
    """Phase 2 TODO: torch-tensorrt AOTI export -> .pt2 with TRT engines."""
    # torch_tensorrt is imported LAZILY here (never at module top level) so that
    # 'import lerobot.export' succeeds on the default install without torch-trt installed.
    raise NotImplementedError(
        "The torch-trt backend is a Phase 2 TODO.\n\n"
        "It requires torch 2.12+ and torch-tensorrt 2.12, which cannot be installed in the\n"
        "default lerobot venv (torch<2.12.0 / cu128 index). Set up the dedicated export venv:\n\n"
        "    python -m venv ~/lerobot-export-env\n"
        "    source ~/lerobot-export-env/bin/activate\n"
        "    pip install -r src/lerobot/export/requirements-export-trt.txt\n"
        "    pip install 'lerobot[smolvla,export]'\n\n"
        "Then re-run lerobot-export with --backend=torch-trt."
    )
