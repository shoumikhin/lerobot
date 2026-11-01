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
from uuid import uuid4

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from lerobot.bundle.manifest import MANIFEST, TEST_CASE, TOKENIZER
from lerobot.inference import FeatureSpec, ObservationSnapshot
from lerobot.policies import RTCConfig
from lerobot.remote_inference import deployment
from lerobot.remote_inference.configs import ExecutionConfig, LanguageConfig, ModelConfig, ServerConfig
from lerobot.remote_inference.deployment import load_deployment
from lerobot.remote_inference.protocol import Envelope, ErrorCode, MessageType
from lerobot.remote_inference.server import SessionWorker
from lerobot.transport.zenoh import ZenohConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.export.numpy_bundle import make_numpy_bundle
from tests.export.test_engine import _carry_bundle
from tests.export.test_task_tokens import task_bundle
from tests.remote_inference.helpers import admit_session, assert_error

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


def observation(
    worker: SessionWorker, identity: str, session: str, task: str, sequence: int, state: np.ndarray = STATE
) -> Envelope:
    body = {
        "artifact_identity": identity,
        "features": {OBS_STATE: state},
        "task": task,
        "task_version": 0,
        "observation_id": uuid4().hex,
        "capture_time": 0.0,
        "delay": 0,
        "sequence": sequence,
    }
    return Envelope(MessageType.OBSERVATION, worker.instance_id, session, 0, uuid4().hex, body)


def test_a_bundle_serves_its_chunk(bundle):
    runner, identity = load_deployment(server_config(bundle))
    assert runner.capabilities.prediction_steps == runner.capabilities.execution_steps == 2
    assert identity.startswith("sha256:")
    chunk = runner.predict(ObservationSnapshot({OBS_STATE: STATE}, 0.0, ""))
    np.testing.assert_array_equal(chunk.canonical_actions, [STATE, 2 * STATE])
    assert chunk.execution_steps == 2


def test_a_served_bundle_refuses_another_task_and_keeps_the_session(bundle):
    """Another task than the bundle's is a request error, and the session still serves the bundle's task."""
    manifest = json.loads((bundle / MANIFEST).read_text())
    (bundle / MANIFEST).write_text(json.dumps({**manifest, "task": "Pick up the red cube."}))
    runner, identity = load_deployment(server_config(bundle))
    worker = SessionWorker(runner, deployment="bundle", artifact_identity=identity, semantics=SEMANTICS)
    try:
        session = admit_session(worker)
        refused = worker.submit(observation(worker, identity, session, "pick up the red cube", 1)).result(2)
        assert_error(refused, ErrorCode.UNSUPPORTED)
        reply = worker.submit(observation(worker, identity, session, "Pick up the red cube.", 2)).result(2)
        assert reply.message_type is MessageType.ACTION
        np.testing.assert_array_equal(reply.body["canonical_actions"], [STATE, 2 * STATE])
    finally:
        worker.close()


def test_a_task_the_server_cannot_tokenize_is_refused_and_keeps_the_session(tmp_path):
    path, _ = task_bundle(tmp_path)
    (path / TOKENIZER).write_text("not a tokenizer")
    motors = MOTORS[:2]
    features = [FeatureSpec(OBS_STATE, (2,), "float32", names=motors, semantics=SEMANTICS)]
    action_feature = FeatureSpec(ACTION, (2,), "float32", names=motors, semantics=SEMANTICS)
    runner, identity = load_deployment(server_config(path, features=features, action_feature=action_feature))
    worker = SessionWorker(runner, deployment="bundle", artifact_identity=identity, semantics=SEMANTICS)
    try:
        session = admit_session(worker)
        state = np.array([1.0, 2.0], np.float32)
        refused = worker.submit(observation(worker, identity, session, "pick the box", 1, state)).result(10)
        assert_error(refused, ErrorCode.UNSUPPORTED)
        # The tokenizer's error output stays in the server's log.
        assert "Traceback" not in refused.body["message"]
        reply = worker.submit(observation(worker, identity, session, "pick the cube", 2, state)).result(2)
        assert reply.message_type is MessageType.ACTION
    finally:
        worker.close()


def test_an_invalidate_keeps_what_a_served_bundle_carries(tmp_path):
    """The server's invalidate (`full=False`) keeps the carried values, as a policy keeps its planner."""
    runner, _ = load_deployment(server_config(_carry_bundle(tmp_path / "carry").path))
    ones = ObservationSnapshot({OBS_STATE: np.ones(3, np.float32)}, 0.0, "")
    runner.predict(ones)
    runner.reset(full=False)
    np.testing.assert_array_equal(runner.predict(ones).canonical_actions, [[2.0, 2.0, 2.0]])
    runner.reset(full=True)
    np.testing.assert_array_equal(runner.predict(ones).canonical_actions, [[1.0, 1.0, 1.0]])


def test_a_served_bundle_starts_without_what_its_warmup_carried(tmp_path, monkeypatch):
    ones = ObservationSnapshot({OBS_STATE: np.ones(3, np.float32)}, 0.0, "")
    # Warmup frames that change the carried total, unlike the zeros the server sends.
    monkeypatch.setattr(deployment, "_warmup_observation", lambda cfg, task: ones)
    runner, _ = load_deployment(server_config(_carry_bundle(tmp_path / "carry").path))
    np.testing.assert_array_equal(runner.predict(ones).canonical_actions, [[1.0, 1.0, 1.0]])


def test_a_served_bundle_answers_no_text(bundle):
    runner, _ = load_deployment(server_config(bundle))
    with pytest.raises(ValueError, match="answers no text queries"):
        runner.query(ObservationSnapshot({OBS_STATE: STATE}, 0.0, ""), kind="subtask", text="What next?")


def test_a_bundle_refuses_action_names_other_than_its_own(tmp_path):
    bundle = make_numpy_bundle(tmp_path / "bundle", action_names=list(MOTORS[::-1]))
    with pytest.raises(
        ValueError, match=r"returns the actions \['motor_3.pos', 'motor_2.pos', 'motor_1.pos'\]"
    ):
        load_deployment(server_config(bundle))


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


def test_a_bundle_from_the_hub_is_refused(bundle, monkeypatch):
    # A Hub repo that gains an export.json would otherwise start running its programs at the next restart.
    monkeypatch.setattr(deployment, "resolve_artifact", lambda repo_or_path, revision: bundle)
    with pytest.raises(ValueError, match="only from a local folder"):
        load_deployment(server_config(bundle, model=ModelConfig("user/bundle")))


def test_a_bundle_that_keeps_past_frames_is_refused(bundle):
    manifest = json.loads((bundle / MANIFEST).read_text())
    (bundle / MANIFEST).write_text(json.dumps({**manifest, "history": {OBS_STATE: 2}}))
    with pytest.raises(ValueError, match="past control ticks"):
        load_deployment(server_config(bundle))


def test_a_bundle_that_keeps_only_the_current_frame_is_served(bundle):
    # A policy exported with n_obs_steps=1 lists its keys with one frame each.
    manifest = json.loads((bundle / MANIFEST).read_text())
    (bundle / MANIFEST).write_text(json.dumps({**manifest, "history": {OBS_STATE: 1}}))
    case = load_file(bundle / TEST_CASE)
    save_file({**case, OBS_STATE: case[OBS_STATE][None]}, bundle / TEST_CASE)
    runner, _ = load_deployment(server_config(bundle))
    chunk = runner.predict(ObservationSnapshot({OBS_STATE: STATE}, 0.0, ""))
    np.testing.assert_array_equal(chunk.canonical_actions, [STATE, 2 * STATE])


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
            r"reads observation.state \(3,\) \(features give \(4,\)\).*the bundle's names and shapes",
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
            {
                "action_feature": FeatureSpec(
                    "action.joints", (3,), "float32", names=MOTORS, semantics=SEMANTICS
                )
            },
            "float32 actions named 'action'",
        ),
        (
            {"action_feature": FeatureSpec(ACTION, (3,), "float32", semantics=SEMANTICS)},
            "names of its action",
        ),
        (
            {"action_feature": FeatureSpec(ACTION, (2,), "float32", names=MOTORS[:2], semantics=SEMANTICS)},
            "returns 3 action components, not 2",
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
