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

"""`lerobot-policy-server` serving a bundle from `lerobot-export`, run by the NumPy test runtime."""

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.bundle.manifest import MANIFEST
from lerobot.inference import FeatureSpec, ObservationSnapshot
from lerobot.policies import RTCConfig
from lerobot.remote_inference.configs import ExecutionConfig, LanguageConfig, ModelConfig, ServerConfig
from lerobot.remote_inference.deployment import load_deployment
from lerobot.transport.zenoh import ZenohConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.export.numpy_bundle import make_numpy_bundle

SEMANTICS = "test-units-v1"
MOTORS = ("motor_1.pos", "motor_2.pos", "motor_3.pos")
STATE = np.array([1.0, 2.0, 3.0], np.float32)


@pytest.fixture
def bundle(tmp_path):
    """A bundle whose chunk is the state, then twice the state."""
    return make_numpy_bundle(tmp_path / "bundle")


def server_config(bundle: Path, **changes) -> ServerConfig:
    settings = {
        "deployment": "bundle",
        "model": ModelConfig(str(bundle)),
        "semantics": SEMANTICS,
        "features": [FeatureSpec(OBS_STATE, (3,), "float32", names=MOTORS, semantics=SEMANTICS)],
        "action_feature": FeatureSpec(ACTION, (3,), "float32", names=MOTORS, semantics=SEMANTICS),
        "zenoh": ZenohConfig(listen_endpoints=["tcp/127.0.0.1:7447"]),
    }
    return ServerConfig(**(settings | changes))


def observation(task: str = "") -> ObservationSnapshot:
    return ObservationSnapshot({OBS_STATE: STATE}, 0.0, task)


def test_a_bundle_serves_its_chunk(bundle):
    runner, identity = load_deployment(server_config(bundle))
    assert runner.capabilities.prediction_steps == runner.capabilities.execution_steps == 2
    assert identity.startswith("sha256:")
    np.testing.assert_array_equal(runner.predict(observation()).canonical_actions, [STATE, 2 * STATE])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "execution": ExecutionConfig(
                    supported_modes=["chunk", "rtc_guided"], rtc=RTCConfig(enabled=True, mode="guided")
                )
            },
            "plain chunks only",
        ),
        ({"execution": ExecutionConfig(n_action_steps=1)}, "execution.n_action_steps"),
        ({"language": LanguageConfig(enabled=True)}, "no text model"),
    ],
)
def test_a_bundle_refuses_settings_that_change_the_policy(bundle, changes, message):
    with pytest.raises(ValueError, match=message):
        load_deployment(server_config(bundle, **changes))


def test_a_bundle_that_keeps_past_frames_is_refused(bundle):
    manifest = json.loads((bundle / MANIFEST).read_text())
    (bundle / MANIFEST).write_text(json.dumps({**manifest, "history": {OBS_STATE: 2}}))
    with pytest.raises(ValueError, match="past control ticks"):
        load_deployment(server_config(bundle))


def test_a_bundle_with_first_runs_is_refused(bundle):
    manifest = json.loads((bundle / MANIFEST).read_text())
    (bundle / MANIFEST).write_text(json.dumps({**manifest, "first_runs": [manifest["runs"]]}))
    with pytest.raises(ValueError, match="first chunks after a reset"):
        load_deployment(server_config(bundle))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"features": [FeatureSpec(OBS_STATE, (4,), "float32", semantics=SEMANTICS)]},
            "does not match the exported policy.*Give the robot's observations the bundle's names",
        ),
        (
            {"features": [FeatureSpec(OBS_STATE, (3,), "float32", names=MOTORS[::-1], semantics=SEMANTICS)]},
            "order must be aligned",
        ),
        (
            {"action_feature": FeatureSpec(ACTION, (3,), "float64", names=MOTORS, semantics=SEMANTICS)},
            "float32 actions",
        ),
        (
            {"action_feature": FeatureSpec(ACTION, (3,), "float32", semantics=SEMANTICS)},
            "names of its action",
        ),
        (
            {
                "features": [
                    FeatureSpec(OBS_STATE, (3,), "float32", names=MOTORS, semantics=SEMANTICS),
                    FeatureSpec(
                        "observation.images.top", (8, 8, 3), "uint8", kind="rgb", semantics=SEMANTICS
                    ),
                ]
            },
            r"does not read the features \['observation.images.top'\]",
        ),
        (
            {"features": [FeatureSpec(OBS_STATE, (3,), "float64", names=MOTORS, semantics=SEMANTICS)]},
            r"takes observation.state as float32 \(not float64\)",
        ),
    ],
)
def test_a_bundle_refuses_a_contract_it_cannot_serve(bundle, changes, message):
    with pytest.raises(ValueError, match=message):
        load_deployment(server_config(bundle, **changes))
