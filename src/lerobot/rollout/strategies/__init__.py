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

"""Rollout strategies — public API re-exports."""

from typing import TYPE_CHECKING

from lerobot.utils.import_utils import lazy_exports

from .base import BaseStrategy
from .core import (
    RolloutStrategy,
    estimate_max_episode_seconds,
    safe_push_to_hub,
    send_next_action,
)
from .factory import create_strategy

# The recording strategies import torch and the dataset stack, so each is imported the first time it is used.
if TYPE_CHECKING:
    from .dagger import DAggerEvents, DAggerPhase, DAggerStrategy
    from .episodic import EpisodicStrategy
    from .highlight import HighlightStrategy
    from .sentry import SentryStrategy
else:
    __getattr__, __dir__ = lazy_exports(
        __name__,
        {
            "DAggerEvents": ".dagger.DAggerEvents",
            "DAggerPhase": ".dagger.DAggerPhase",
            "DAggerStrategy": ".dagger.DAggerStrategy",
            "EpisodicStrategy": ".episodic.EpisodicStrategy",
            "HighlightStrategy": ".highlight.HighlightStrategy",
            "SentryStrategy": ".sentry.SentryStrategy",
        },
    )

__all__ = [
    "BaseStrategy",
    "DAggerEvents",
    "DAggerPhase",
    "DAggerStrategy",
    "HighlightStrategy",
    "EpisodicStrategy",
    "RolloutStrategy",
    "SentryStrategy",
    "create_strategy",
    "estimate_max_episode_seconds",
    "safe_push_to_hub",
    "send_next_action",
]
