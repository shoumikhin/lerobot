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

"""Serve a bundle written by `lerobot-export`."""

from __future__ import annotations

import logging
import time

from lerobot.bundle.engine import BundleEngine
from lerobot.inference import (
    ActionChunk,
    ActionProvenance,
    ExecutionMode,
    FeatureSpec,
    ObservationSnapshot,
    PolicyCapabilities,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from .protocol import ErrorCode, ProtocolError

logger = logging.getLogger(__name__)


class BundleRunner:
    """The runner of a deployment whose model is a bundle: one run of its programs per chunk.

    A bundle runs as it was compiled, so it serves plain chunks of its own length and answers no text.

    Args:
        engine (`BundleEngine`):
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
        engine: BundleEngine,
        *,
        action_interval: float,
        features: tuple[FeatureSpec, ...],
        action_feature: FeatureSpec,
    ) -> None:
        """Refuse features or actions that differ from what the bundle takes and returns."""
        if not action_feature.names:
            raise ValueError("A bundle needs the names of its action components in action_feature.")
        # Named components already make the feature one-dimensional.
        if action_feature.name != ACTION or action_feature.dtype != "float32":
            raise ValueError(f"A bundle returns float32 actions named {ACTION!r}.")
        served = {feature.name: tuple(feature.shape) for feature in features}
        if wrong := [
            f"{name} {shape} (features give {served.get(name)})"
            for name, shape in engine.frame_shapes.items()
            if served.get(name) != shape
        ]:
            raise ValueError(
                f"The bundle reads {'; '.join(wrong)}. Give the server's features the bundle's names and "
                "shapes."
            )
        names = list(action_feature.names)
        if engine.action_names is not None and names != engine.action_names:
            raise ValueError(
                f"The bundle returns the actions {engine.action_names}, not {names}. Set "
                "action_feature.names to them."
            )
        if len(names) != engine.action_dim:
            raise ValueError(
                f"The bundle returns {engine.action_dim} action components, not {len(names)}. Name that "
                "many in action_feature.names."
            )
        dtypes = engine.frame_dtypes
        if unread := sorted({feature.name for feature in features} - set(dtypes)):
            raise ValueError(f"The bundle does not read the features {unread}. Serve only the ones it takes.")
        if wrong := [
            f"{feature.name} as {dtypes[feature.name]} (not {feature.dtype})"
            for feature in features
            if feature.dtype != dtypes[feature.name]
        ]:
            raise ValueError(f"The bundle takes {'; '.join(wrong)}.")
        # A bundle takes the state in the order of its actions when both name the same joints.
        state = next((feature for feature in features if feature.name == OBS_STATE), None)
        if (
            state is not None
            and set(state.names) == set(action_feature.names)
            and state.names != action_feature.names
        ):
            raise ValueError("State and action component order must be aligned.")
        self.engine = engine
        self._default_task = engine.task
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
        """Run the bundle on one observation and return its chunk.

        An empty task stands for the one the bundle was exported for. A task the bundle cannot run or
        prepare is refused as a request error: the bundle keeps its task, so the server keeps the session. A
        plain chunk reads no continuation, so the server's other arguments are not used.
        """
        task = observation.task or self._default_task
        if task != self.engine.task:
            try:
                self.engine.set_task(task)
            except ValueError as exc:
                raise ProtocolError(ErrorCode.UNSUPPORTED, str(exc)) from exc
            except RuntimeError as exc:
                # Such as a failed tokenizer process, whose whole error output the message holds.
                logger.warning("The bundle could not take the task %r: %s", task, exc)
                raise ProtocolError(
                    ErrorCode.UNSUPPORTED, "The server could not prepare this task; its log says why."
                ) from exc
        started = time.perf_counter()
        actions = self.engine.predict_chunk(observation.features)
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
        """Drop what the engine keeps between chunks, on a full reset only.

        The server's invalidate (`full=False`), for a pause or a new instruction, keeps the carried values, as
        `PolicyRunner` keeps a policy's planner memory. The server runs whole chunks, so no action is queued.
        """
        if full:
            self.engine.reset()
