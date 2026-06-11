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
"""Adapter registry: maps policy names to ExportAdapter classes.

Adapters auto-register on first import via @register_adapter.
The registry is keyed by the same policy name used by lerobot's get_policy_class().
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

import torch.nn as nn

from lerobot.export.adapters.base import ExportAdapter

if TYPE_CHECKING:
    from lerobot.export.cli import ExportConfig

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[ExportAdapter]] = {}

# Map policy names to their adapter module paths (lazy import to avoid heavy deps).
_ADAPTER_MODULES: dict[str, str] = {
    "smolvla": "lerobot.export.adapters.smolvla",
    # Phase 1: add pi0, act, diffusion, etc.
}


def register_adapter(cls: type) -> type:
    """Class decorator that registers an ExportAdapter subclass."""
    name = getattr(cls, "policy_name", None)
    if not name:
        raise TypeError(f"ExportAdapter subclass {cls.__name__} must define policy_name")
    _REGISTRY[name] = cls
    return cls


def get_adapter(policy_name: str, policy: nn.Module, cfg: ExportConfig) -> ExportAdapter:
    """Return a (lazily imported) adapter instance for the given policy type."""
    if policy_name not in _REGISTRY:
        module_path = _ADAPTER_MODULES.get(policy_name)
        if module_path is None:
            raise NotImplementedError(
                f"No ExportAdapter registered for policy type {policy_name!r}. "
                f"Available adapters: {sorted(_ADAPTER_MODULES.keys())}. "
                "To add support for a new policy, create "
                "src/lerobot/export/adapters/<policy_name>.py and decorate the "
                "adapter class with @register_adapter."
            )
        importlib.import_module(module_path)

    adapter_cls = _REGISTRY[policy_name]
    return adapter_cls(policy, cfg)
