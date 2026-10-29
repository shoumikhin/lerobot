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

"""Serve a bundle written by `lerobot-export`, without importing PyTorch."""

from __future__ import annotations

import time

from lerobot.inference import (
    ActionChunk,
    ActionProvenance,
    ExecutionMode,
    FeatureSpec,
    ObservationSnapshot,
    PolicyCapabilities,
)
from lerobot.inference.bundle import BundleInferenceEngine


class BundleRunner:
    """The runner of a deployment whose model is a bundle: one run of its programs per chunk.

    A bundle runs as it was compiled, so it serves plain chunks of its own length and no text.

    Args:
        engine (`BundleInferenceEngine`):
            The loaded bundle, already checked against its test case.
        action_interval (`float`):
            Seconds between two actions of a chunk.
        features (`tuple[FeatureSpec, ...]`):
            The observation features the server takes, named as the bundle names them.
        action_feature (`FeatureSpec`):
            The action the server returns.
    """

    def __init__(
        self,
        engine: BundleInferenceEngine,
        *,
        action_interval: float,
        features: tuple[FeatureSpec, ...],
        action_feature: FeatureSpec,
    ) -> None:
        """Refuse features or actions that differ from what the bundle takes and returns."""
        engine.check_robot({feature.name: feature.shape for feature in features}, list(action_feature.names))
        self.engine = engine
        self.capabilities = PolicyCapabilities(
            modes=(ExecutionMode.CHUNK,),
            prediction_steps=engine.chunk_length,
            execution_steps=engine.chunk_length,
            action_interval=action_interval,
            features=features,
            action_feature=action_feature,
        )

    def predict(
        self, observation: ObservationSnapshot, *, provenance: ActionProvenance | None = None, **_: object
    ) -> ActionChunk:
        """Run the bundle on a fresh snapshot and return its chunk of actions.

        An empty task stands for the bundle's own. A plain chunk reads no continuation, so the server's other
        arguments are not used.
        """
        if observation.task and (error := self.engine.text_input_error(observation.task, instruction=True)):
            raise ValueError(error)
        started = time.perf_counter()
        actions = self.engine.run_chunk(observation.features, observation.task or self.engine.task)
        return ActionChunk(
            model_actions=None,
            canonical_actions=actions,
            provenance=provenance
            or ActionProvenance(
                capture_time=observation.capture_time,
                task=observation.task,
                task_version=observation.task_version,
                observation_id=observation.observation_id,
            ),
            execution_steps=len(actions),
            server_durations={"policy": time.perf_counter() - started},
        )

    def query(self, observation: ObservationSnapshot, *, kind: str, text: str) -> str:
        """A bundle holds no text model."""
        raise ValueError("This deployment serves a bundle, which answers no text queries.")

    def reset(self, *, full: bool = True) -> None:
        """Drop what the engine holds between chunks."""
        self.engine.reset()
