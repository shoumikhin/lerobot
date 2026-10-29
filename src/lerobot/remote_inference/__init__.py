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

"""Exclusive asynchronous policy serving over Zenoh."""

from typing import TYPE_CHECKING

from lerobot.utils.import_utils import lazy_exports

from .configs import ExecutionConfig, LanguageConfig, ModelConfig, ServerConfig
from .deployment import artifact_identity, load_deployment
from .protocol import PROTOCOL_VERSION, AdmissionDeniedError, ErrorCode, ProtocolError
from .server import PolicyServer, SessionWorker

# The robot side imports torch, so each is imported the first time it is used; a server of a bundle never does.
if TYPE_CHECKING:
    from .client import RemoteClient, RequestCancelled
    from .engine import RemoteInferenceEngine
else:
    __getattr__, __dir__ = lazy_exports(
        __name__,
        {
            "RemoteClient": ".client.RemoteClient",
            "RequestCancelled": ".client.RequestCancelled",
            "RemoteInferenceEngine": ".engine.RemoteInferenceEngine",
        },
    )

__all__ = [
    "PROTOCOL_VERSION",
    "AdmissionDeniedError",
    "ErrorCode",
    "ExecutionConfig",
    "LanguageConfig",
    "ModelConfig",
    "PolicyServer",
    "ProtocolError",
    "RemoteClient",
    "RemoteInferenceEngine",
    "RequestCancelled",
    "ServerConfig",
    "SessionWorker",
    "artifact_identity",
    "load_deployment",
]
