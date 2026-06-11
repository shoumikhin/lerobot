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
"""ExportAdapter base class and ExampleInputs type alias."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from lerobot.export.cli import ExportConfig

# Type alias: a tuple of positional args + a dict of keyword args for torch.export.export.
ExampleInputs = tuple[tuple[Any, ...], dict[str, Any]]


@runtime_checkable
class ExportAdapter(Protocol):
    """Protocol every per-policy export adapter must satisfy.

    An adapter is ONLY responsible for:
      1. Declaring which PUBLIC policy modules to export (as thin nn.Module wrappers).
      2. Providing example inputs for tracing.
      3. Declaring the decode-loop recipe for the manifest.

    It MUST NOT edit any model code or import from modeling_*.py internal symbols.
    """

    policy_name: str
    """Must match the lerobot policy registry key (e.g. 'smolvla', 'pi0')."""

    def graphs(
        self,
        policy: nn.Module,
        cfg: ExportConfig,
    ) -> dict[str, tuple[nn.Module, ExampleInputs]]:
        """Return the exportable graph parts.

        Each entry: graph_name -> (module, (args_tuple, kwargs_dict)).
        The module wraps public policy methods; example inputs drive shape-tracing.
        """
        ...

    def manifest_extra(self, policy: nn.Module) -> dict:
        """Return decode-loop recipe and preprocessing metadata for manifest.json."""
        ...

    def shape_overrides(self) -> dict:
        """Config field overrides to apply before tracing (e.g. pad_language_to='max_length')."""
        return {}
