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
import signal
import subprocess
import sys
import time
from pathlib import Path

import draccus
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
from tests.remote_inference.network_helpers import free_port

REPO = Path(__file__).parents[2]
SEMANTICS = "test-units-v1"
MOTORS = ("motor_1.pos", "motor_2.pos", "motor_3.pos")
STATE = np.array([1.0, 2.0, 3.0], np.float32)

# `lerobot-policy-server --config_path=argv[1]`.
SERVE = """
import sys

from lerobot.scripts import lerobot_policy_server

sys.argv = ["lerobot-policy-server", f"--config_path={sys.argv[1]}"]
lerobot_policy_server.main()
"""

# `lerobot-rollout --inference.type=remote` on a mock robot, against the server at argv[1].
REMOTE_ROLLOUT = f"""
import json
import sys

from tests.mocks.mock_robot import MockRobot

sent = []
send_action = MockRobot.send_action
MockRobot.send_action = lambda self, action: sent.append(dict(action)) or send_action(self, action)
# The remote client holds the robot in place while it waits, which the mock robot can do.
MockRobot.supports_position_hold = True

from lerobot.scripts import lerobot_rollout

sys.argv = [
    "lerobot-rollout",
    "--inference.type=remote",
    f"--inference.endpoint={{sys.argv[1]}}",
    "--inference.deployment=bundle",
    "--inference.semantics={SEMANTICS}",
    "--robot.type=mock_robot",
    "--robot.random_values=false",
    "--robot.static_values=[1, 2, 3]",
    "--strategy.type=base",
    "--duration=1",
    "--fps=30",
]
lerobot_rollout.main()
print(json.dumps(sent))
"""


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


def test_a_bundle_runs_its_own_task_and_refuses_another(bundle):
    manifest = json.loads((bundle / MANIFEST).read_text())
    (bundle / MANIFEST).write_text(json.dumps({**manifest, "task": "pick the cube"}))
    runner, _ = load_deployment(server_config(bundle))
    for task in ("", "pick the cube"):
        runner.predict(observation(task))
    with pytest.raises(ValueError, match="runs only the task it was exported for"):
        runner.predict(observation("stack the cups"))


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
            {"action_feature": FeatureSpec("actions", (3,), "float32", names=MOTORS, semantics=SEMANTICS)},
            "float32 actions named 'action'",
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


def test_remote_rollout_plays_the_chunks_of_a_served_bundle(bundle, tmp_path):
    """A robot client drives a mock robot with the chunks of a bundle that the server runs."""
    pytest.importorskip("zenoh")
    pytest.importorskip("msgpack")
    endpoint = f"tcp/127.0.0.1:{free_port()}"
    config = tmp_path / "server.json"
    served = server_config(bundle, zenoh=ZenohConfig(listen_endpoints=[endpoint]))
    config.write_text(json.dumps(draccus.encode(served)))
    log = tmp_path / "server.log"
    with log.open("w") as output:
        server = subprocess.Popen(
            [sys.executable, "-c", SERVE, str(config)],
            stdout=output,
            stderr=subprocess.STDOUT,
            cwd=REPO,
        )
        try:
            deadline = time.monotonic() + 120
            while "Policy server ready:" not in log.read_text():
                assert server.poll() is None and time.monotonic() < deadline, log.read_text()
                time.sleep(0.05)
            client = subprocess.run(
                [sys.executable, "-c", REMOTE_ROLLOUT, endpoint],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=REPO,
            )
            assert client.returncode == 0, client.stderr[-4000:]
            server.send_signal(signal.SIGTERM)
            assert server.wait(timeout=20) == 0, log.read_text()[-4000:]
        finally:
            if server.poll() is None:
                server.kill()
                server.wait(timeout=10)
    sent = json.loads(client.stdout.strip().splitlines()[-1])
    assert list(sent[0]) == list(MOTORS)
    rows = np.array([list(action.values()) for action in sent])
    # Only the bundle doubles the state. Before and after, the robot holds or returns to its first position.
    assert (rows == 2 * STATE).all(axis=1).any()
    assert all(np.allclose(row, STATE) or np.allclose(row, 2 * STATE) for row in rows)
