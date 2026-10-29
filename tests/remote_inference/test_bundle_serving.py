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

"""`lerobot-policy-server` serving a bundle from `lerobot-export`, without importing PyTorch."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from lerobot.inference import FeatureSpec
from lerobot.policies import RTCConfig
from lerobot.remote_inference import ExecutionConfig, LanguageConfig, ModelConfig, ServerConfig
from lerobot.remote_inference.deployment import load_deployment
from lerobot.transport.zenoh import ZenohConfig
from lerobot.utils.import_utils import is_package_available
from tests.remote_inference.network_helpers import free_port
from tests.test_torch_free_imports import (
    ACTION_NAMES,
    BLOCK_TORCH,
    NUMPY_RUNTIME,
    STATE,
    write_bundle,
)

SEMANTICS = "test-units-v1"
REPO = Path(__file__).resolve().parents[2]

# `lerobot-policy-server` on the bundle in argv[1], listening on argv[2], in a process where torch refuses to load.
SERVE_WITHOUT_TORCH = (
    BLOCK_TORCH
    + f"""
from lerobot.inference import FeatureSpec
from lerobot.remote_inference import ModelConfig, ServerConfig
from lerobot.scripts import lerobot_policy_server
from lerobot.transport.zenoh import ZenohConfig

names = {tuple(ACTION_NAMES)!r}
lerobot_policy_server.serve(
    ServerConfig(
        deployment="bundle", model=ModelConfig(sys.argv[1]), semantics="{SEMANTICS}",
        features=[FeatureSpec("observation.state", (3,), "float32", names=names, semantics="{SEMANTICS}")],
        action_feature=FeatureSpec("action", (3,), "float32", names=names, semantics="{SEMANTICS}"),
        zenoh=ZenohConfig(listen_endpoints=[sys.argv[2]]),
    )
)
if attempts:
    sys.exit(attempts[0])
assert "torch" not in sys.modules
"""
)

# `lerobot-rollout --inference.type=remote` on a mock robot, against the server at argv[1].
REMOTE_ROLLOUT = f"""
import json
import sys

sys.path.insert(0, {str(REPO)!r})
from tests.mocks.mock_robot import MockRobot

sent = []
send_action = MockRobot.send_action
MockRobot.send_action = lambda self, action: sent.append(dict(action)) or send_action(self, action)
# The remote client holds the robot in place while it waits, which the mock robot can do.
MockRobot.supports_position_hold = True

from lerobot.scripts import lerobot_rollout

sys.argv = [
    "lerobot-rollout", "--inference.type=remote", f"--inference.endpoint={{sys.argv[1]}}",
    "--inference.deployment=bundle", "--inference.semantics={SEMANTICS}", "--robot.type=mock_robot",
    "--robot.random_values=false", "--robot.static_values=[1,2,3]", "--strategy.type=base",
    "--duration=1", "--fps=30",
]
lerobot_rollout.main()
print(json.dumps(sent))
"""


@pytest.fixture
def numpy_bundle(tmp_path, monkeypatch):
    """A bundle whose program is a NumPy function, registered by a module on the import path."""
    (tmp_path / "numpy_runtime.py").write_text(NUMPY_RUNTIME)
    monkeypatch.syspath_prepend(str(tmp_path))
    return write_bundle(tmp_path / "bundle", "numpy_test", ["numpy_runtime"], "model.bin")


def executorch_bundle(folder: Path) -> Path:
    import torch
    from executorch.exir import to_edge

    class TwoActions(torch.nn.Module):
        def forward(self, state):
            return torch.stack([state, 2 * state])

    program = to_edge(torch.export.export(TwoActions(), (torch.from_numpy(STATE),))).to_executorch()
    folder.mkdir()
    (folder / "model.pte").write_bytes(program.buffer)
    return write_bundle(folder, "executorch", [], "model.pte")


def server_config(bundle: Path, **kwargs) -> ServerConfig:
    names = tuple(ACTION_NAMES)
    return ServerConfig(
        deployment="bundle",
        model=ModelConfig(str(bundle)),
        semantics=SEMANTICS,
        features=[FeatureSpec("observation.state", (3,), "float32", names=names, semantics=SEMANTICS)],
        action_feature=FeatureSpec("action", (3,), "float32", names=names, semantics=SEMANTICS),
        zenoh=ZenohConfig(listen_endpoints=["tcp/127.0.0.1:0"]),
        **kwargs,
    )


@pytest.mark.dataset
@pytest.mark.parametrize(
    "runtime",
    [
        "numpy",
        pytest.param(
            "executorch",
            marks=pytest.mark.skipif(
                not is_package_available("executorch"), reason="executorch not installed"
            ),
        ),
    ],
)
def test_remote_rollout_plays_a_bundle_served_without_torch(tmp_path, runtime):
    """A robot client drives a mock robot with the chunks of a bundle that a torch-free server runs."""
    pytest.importorskip("zenoh")
    pytest.importorskip("msgpack")
    pytest.importorskip("datasets")
    if runtime == "numpy":
        (tmp_path / "numpy_runtime.py").write_text(NUMPY_RUNTIME)
        bundle = write_bundle(tmp_path / "bundle", "numpy_test", ["numpy_runtime"], "model.bin")
    else:
        bundle = executorch_bundle(tmp_path / "bundle")
    endpoint = f"tcp/127.0.0.1:{free_port()}"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(REPO)])}
    log = tmp_path / "server.log"
    with log.open("w") as output:
        server = subprocess.Popen(
            [sys.executable, "-c", SERVE_WITHOUT_TORCH, str(bundle), endpoint],
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            deadline = time.monotonic() + 60
            while "Policy server ready:" not in log.read_text():
                assert server.poll() is None and time.monotonic() < deadline, log.read_text()
                time.sleep(0.05)
            client = subprocess.run(
                [sys.executable, "-c", REMOTE_ROLLOUT, endpoint],
                capture_output=True,
                text=True,
                timeout=120,
            )
            assert client.returncode == 0, client.stderr[-4000:]
            server.send_signal(signal.SIGTERM)
            assert server.wait(timeout=20) == 0, log.read_text()[-4000:]
        finally:
            if server.poll() is None:
                server.kill()
                server.wait(timeout=10)
    sent = json.loads(client.stdout.strip().splitlines()[-1])
    assert list(sent[0]) == ACTION_NAMES
    rows = np.array([list(action.values()) for action in sent])
    # Only the bundle doubles the state. Before and after, the robot holds or returns to its first position.
    assert (rows == 2 * STATE).all(axis=1).any()
    assert all(np.allclose(row, STATE) or np.allclose(row, 2 * STATE) for row in rows)


@pytest.mark.parametrize(
    ("kwargs", "message"),
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
def test_bundle_refuses_settings_that_change_the_policy(numpy_bundle, kwargs, message):
    with pytest.raises(ValueError, match=message):
        load_deployment(server_config(numpy_bundle, **kwargs))


def test_bundle_refuses_features_it_does_not_take(numpy_bundle):
    cfg = server_config(numpy_bundle)
    cfg.features = [FeatureSpec("observation.state", (4,), "float32", semantics=SEMANTICS)]
    with pytest.raises(ValueError, match="does not match the exported policy"):
        load_deployment(cfg)


def test_bundle_serves_its_chunk(numpy_bundle):
    runner, identity = load_deployment(server_config(numpy_bundle))
    assert runner.capabilities.prediction_steps == runner.capabilities.execution_steps == 2
    assert identity.startswith("sha256:")
