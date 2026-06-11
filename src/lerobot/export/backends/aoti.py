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
"""AOTI backend: vanilla AOTInductor export to .pt2 bundles.

Works on the DEFAULT lerobot install (torch 2.11+cu128 or newer).
Uses torch.export.export and torch._inductor.aoti_compile_and_package.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from lerobot.export.manifest import write_manifest

if TYPE_CHECKING:
    from lerobot.export.adapters.base import ExportAdapter
    from lerobot.export.cli import ExportConfig

logger = logging.getLogger(__name__)


def _export_one_graph(
    name: str,
    module: nn.Module,
    example_inputs: tuple,
    output_dir: Path,
) -> Path:
    """Export a single nn.Module to a .pt2 AOTI package.

    Args:
        name: graph name (used as file stem, e.g. 'prefill').
        module: the nn.Module to export (must be traceable with torch.export).
        example_inputs: (args_tuple, kwargs_dict) for tracing.
        output_dir: directory to write the .pt2 file into.

    Returns:
        Path to the written .pt2 file.
    """
    args, kwargs = example_inputs
    pt2_path = str(output_dir / f"{name}.pt2")

    logger.info("Exporting graph %r via torch.export.export ...", name)
    with torch.no_grad():
        ep = torch.export.export(module, args=args, kwargs=kwargs or {})

    logger.info("Compiling and packaging %r -> %s ...", name, pt2_path)
    torch._inductor.aoti_compile_and_package(ep, package_path=pt2_path)

    logger.info("Written: %s", pt2_path)
    return Path(pt2_path)


def run_aoti_export(adapter: ExportAdapter, cfg: ExportConfig, output_dir: Path) -> None:
    """Run the full AOTI export for all graphs declared by the adapter."""
    graphs = adapter.graphs(adapter.policy, cfg)

    artifact_paths: dict[str, str] = {}
    for graph_name, (module, example_inputs) in graphs.items():
        pt2_path = _export_one_graph(graph_name, module, example_inputs, output_dir)
        artifact_paths[graph_name] = pt2_path.name

    manifest_extra = adapter.manifest_extra(adapter.policy)
    write_manifest(
        output_dir=output_dir,
        policy_name=adapter.policy_name,
        backend="aoti",
        artifact_paths=artifact_paths,
        manifest_extra=manifest_extra,
        cfg=cfg,
    )
    logger.info("AOTI export complete. Files in %s:", output_dir)
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            logger.info("  %s", f.relative_to(output_dir))
