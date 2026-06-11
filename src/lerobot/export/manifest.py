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
"""Manifest writer for lerobot export bundles.

Produces manifest.json: the self-contained recipe a C++ runner needs to
load and execute the exported graphs without calling back into Python.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from lerobot.export.cli import ExportConfig

logger = logging.getLogger(__name__)


def write_manifest(
    output_dir: Path,
    policy_name: str,
    backend: str,
    artifact_paths: dict[str, str],
    manifest_extra: dict,
    cfg: ExportConfig,
) -> Path:
    """Write manifest.json into output_dir.

    Args:
        output_dir: bundle root directory.
        policy_name: lerobot policy type name (e.g. 'smolvla').
        backend: one of 'aoti', 'torch-trt', 'torch-trt-et'.
        artifact_paths: mapping graph_name -> filename (relative to output_dir).
        manifest_extra: decode recipe + preprocess metadata from the adapter.
        cfg: export config (for recording batch_size, etc.).

    Returns:
        Path to the written manifest.json.
    """
    import lerobot

    toolchain: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torch_tensorrt": _try_import_version("torch_tensorrt"),
        "executorch": _try_import_version("executorch"),
    }

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "policy": {
            "name": policy_name,
            "lerobot_version": getattr(lerobot, "__version__", "unknown"),
            "source_checkpoint": cfg.policy.path,
        },
        "backend": backend,
        "toolchain": toolchain,
        "export_config": {
            "batch_size": cfg.batch_size,
            "device": cfg.policy.device,
        },
        "graphs": {
            name: {"file": path} for name, path in artifact_paths.items()
        },
        # Preprocessing, postprocessing, and decode recipe from the adapter.
        # Phase 1: adapters will populate these from policy.config + processor metadata.
        "preprocess": manifest_extra.get("preprocess", {}),
        "postprocess": manifest_extra.get("postprocess", {}),
        "decode": manifest_extra.get("decode", {}),
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written to %s", manifest_path)
    return manifest_path


def _try_import_version(pkg: str) -> str | None:
    try:
        m = __import__(pkg)
        return getattr(m, "__version__", "unknown")
    except ImportError:
        return None
