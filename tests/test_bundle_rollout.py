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

"""`lerobot-rollout --bundle.path`: a bundle on a robot, without PyTorch."""

import os
import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from lerobot.bundle import rollout as bundle_rollout
from lerobot.bundle.manifest import TEST_CASE
from lerobot.bundle.rollout import BundleConfig, BundleRolloutConfig, rollout
from lerobot.scripts import lerobot_rollout_entry
from lerobot.utils.constants import OBS_STATE
from tests.export.numpy_bundle import RecordingRobotConfig, make_numpy_bundle, read_log
from tests.torch_free import assert_runs_without_torch

REPO = Path(__file__).parents[1]
MOTORS = ["motor_1.pos", "motor_2.pos", "motor_3.pos"]


def _install(monkeypatch, *plugins: str) -> None:
    """Make the test plugins `plugins` installed packages for the commands a test starts."""
    folders = [str(REPO / "tests" / "export" / "plugins" / name) for name in plugins]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [*folders, os.environ.get("PYTHONPATH")])))


@pytest.fixture(autouse=True)
def no_return_wait(monkeypatch):
    monkeypatch.setattr("lerobot.utils.robot_utils.precise_sleep", lambda _: None)


def _config(tmp_path, bundle, **changes) -> BundleRolloutConfig:
    robot = RecordingRobotConfig(log=str(tmp_path / "robot.jsonl"), static_values=[1.0, 2.0, 3.0])
    values = {"fps": 200.0, "duration": 0.05, "return_to_initial_position": False, **changes}
    return BundleRolloutConfig(bundle=BundleConfig(bundle), robot=robot, **values)


def _actions(tmp_path) -> list[dict]:
    return [value for event, value in read_log(tmp_path / "robot.jsonl") if event == "action"]


@pytest.mark.parametrize(
    ("plugins", "robot_package"),
    [
        # The named package adds the robot, so the installed plugin, which imports PyTorch, does not load.
        (["torch_robot"], ["--robot.discover_packages_path=tests.export.recording_robot"]),
        # No loaded package has the robot, so the installed plugin that adds it loads.
        (["recording_robot"], []),
    ],
)
def test_the_command_runs_a_bundle_without_torch(tmp_path, monkeypatch, plugins, robot_package):
    _install(monkeypatch, *plugins)
    bundle = make_numpy_bundle(tmp_path / "bundle")
    log = tmp_path / "robot.jsonl"
    assert_runs_without_torch(
        """
        import json, sys
        from lerobot.scripts.lerobot_rollout_entry import main

        sys.argv = ["lerobot-rollout", *sys.argv[1:]]
        main()
        events = [json.loads(line)[0] for line in open(sys.argv[-1].split("=", 1)[1])]
        assert events[0] == "connect" and events[-1] == "disconnect" and "action" in events, events
        """,
        f"--bundle.path={bundle}",
        *robot_package,
        "--robot.type=recording_robot",
        "--robot.static_values=[1.0, 2.0, 3.0]",
        "--fps=100",
        "--duration=0.1",
        f"--robot.log={log}",
        cwd=REPO,
    )


def test_discovering_a_robot_package_of_lerobot_imports_no_torch(tmp_path):
    """The package registers the robot; its kinematics processor, which imports PyTorch, is not loaded."""
    assert_runs_without_torch(
        """
        import sys
        from lerobot.scripts.lerobot_rollout_entry import main

        sys.argv = ["lerobot-rollout", *sys.argv[1:]]
        try:
            main()
        except SystemExit as error:
            assert error.code == 0, error.code
        """,
        f"--bundle.path={make_numpy_bundle(tmp_path / 'bundle')}",
        "--robot.discover_packages_path=lerobot.robots.so_follower",
        "--help",
        cwd=REPO,
    )


def test_an_installed_robot_plugin_does_not_load_for_a_robot_lerobot_has(tmp_path, monkeypatch):
    """A robot plugin may import PyTorch, so it loads only when the robot may come from it."""
    _install(monkeypatch, "torch_robot")
    assert_runs_without_torch(
        """
        import sys
        from lerobot.scripts.lerobot_rollout_entry import main

        sys.argv = ["lerobot-rollout", *sys.argv[1:]]
        try:
            main()
        except SystemExit as error:
            assert error.code == 0, error.code
        """,
        f"--bundle.path={make_numpy_bundle(tmp_path / 'bundle')}",
        "--robot.type=so101_follower",
        "--help",
        cwd=REPO,
    )


def test_bundle_mode_loads_no_plugin_but_robots_and_cameras(tmp_path):
    with pytest.raises(ValueError, match="robot and camera plugins only"):
        bundle_rollout.main(
            [f"--bundle.path={make_numpy_bundle(tmp_path / 'b')}", "--policy.discover_packages_path=x"]
        )


def test_actions_follow_the_joint_names_the_bundle_records(tmp_path):
    names = list(reversed(MOTORS))
    rollout(_config(tmp_path, make_numpy_bundle(tmp_path / "bundle", action_names=names)))
    # The state is read in the bundle's joint order (3, 2, 1), and each action is named the same way.
    assert _actions(tmp_path)[0] == {"motor_3.pos": 3.0, "motor_2.pos": 2.0, "motor_1.pos": 1.0}
    assert _actions(tmp_path)[1] == {"motor_3.pos": 6.0, "motor_2.pos": 4.0, "motor_1.pos": 2.0}


def test_interpolation_sends_commands_between_actions(tmp_path):
    rollout(_config(tmp_path, make_numpy_bundle(tmp_path / "bundle"), interpolation_multiplier=2))
    assert _actions(tmp_path)[:3] == [
        dict(zip(MOTORS, [1.0, 2.0, 3.0], strict=True)),
        dict(zip(MOTORS, [1.5, 3.0, 4.5], strict=True)),
        dict(zip(MOTORS, [2.0, 4.0, 6.0], strict=True)),
    ]


def test_a_bundle_that_misses_its_test_case_never_connects(tmp_path):
    with pytest.raises(RuntimeError, match="does not reproduce its test case"):
        rollout(_config(tmp_path, make_numpy_bundle(tmp_path / "bundle", expected_offset=1.0)))
    assert read_log(tmp_path / "robot.jsonl") == []


def test_a_robot_that_does_not_match_never_connects(tmp_path):
    config = _config(tmp_path, make_numpy_bundle(tmp_path / "bundle", n_motors=2))
    with pytest.raises(ValueError, match="does not match the exported policy"):
        rollout(config)
    assert read_log(tmp_path / "robot.jsonl") == []


def test_a_robot_frame_of_another_dtype_stops_the_run_before_any_action(tmp_path):
    bundle = make_numpy_bundle(tmp_path / "bundle")
    case = load_file(bundle / TEST_CASE)
    save_file({**case, OBS_STATE: case[OBS_STATE].astype(np.float64)}, bundle / TEST_CASE)
    with pytest.raises(ValueError, match=r"takes observation.state as float64 \(the robot gives float32\)"):
        rollout(_config(tmp_path, bundle, return_to_initial_position=True))
    assert read_log(tmp_path / "robot.jsonl") == [["connect", None], ["disconnect", None]]


def test_non_finite_actions_stop_the_run_and_disconnect_without_moving(tmp_path):
    config = _config(tmp_path, make_numpy_bundle(tmp_path / "bundle"), return_to_initial_position=True)
    config.robot.static_values = [float("nan"), 2.0, 3.0]
    with pytest.raises(RuntimeError, match="NaN or infinite"):
        rollout(config)
    assert read_log(tmp_path / "robot.jsonl") == [["connect", None], ["disconnect", None]]


def test_a_stop_during_inference_sends_nothing_after_it(tmp_path, monkeypatch):
    stop = Event()
    monkeypatch.setattr(
        bundle_rollout, "ProcessSignalHandler", lambda **_: SimpleNamespace(shutdown_event=stop)
    )
    run_chunk = bundle_rollout.BundleEngine._run_chunk
    chunks = []

    def stop_while_running(engine, frame, noise=None):
        actions = run_chunk(engine, frame, noise)
        if engine.task == "stop now":
            chunks.append(actions)
            # The first chunk is the test case, run while the bundle loads; the second is the first live one.
            if len(chunks) == 2:
                stop.set()
        return actions

    monkeypatch.setattr(bundle_rollout.BundleEngine, "_run_chunk", stop_while_running)
    rollout(_config(tmp_path, make_numpy_bundle(tmp_path / "bundle"), duration=0, task="stop now"))
    assert len(chunks) == 2
    assert _actions(tmp_path) == []
    assert read_log(tmp_path / "robot.jsonl")[-1] == ["disconnect", None]


def test_a_clean_stop_returns_the_robot_to_where_it_started(tmp_path):
    rollout(_config(tmp_path, make_numpy_bundle(tmp_path / "bundle"), return_to_initial_position=True))
    assert _actions(tmp_path)[-1] == dict(zip(MOTORS, [1.0, 2.0, 3.0], strict=True))
    assert read_log(tmp_path / "robot.jsonl")[-1] == ["disconnect", None]


def test_the_config_refuses_a_folder_that_is_not_a_bundle(tmp_path):
    with pytest.raises(ValueError, match="not a bundle written by lerobot-export"):
        _config(tmp_path, tmp_path)


def test_bundle_mode_loads_robot_plugins_only_for_a_robot_no_loaded_package_has(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(bundle_rollout, "register_third_party_plugins", lambda kinds=(): loaded.append(kinds))
    monkeypatch.setattr(bundle_rollout, "rollout", lambda cfg: None)
    bundle = f"--bundle.path={make_numpy_bundle(tmp_path / 'bundle')}"
    config = tmp_path / "rollout.yaml"
    config.write_text("robot:\n  type: mock_robot\n")
    bundle_rollout.main([bundle, "--robot.type=mock_robot"])
    bundle_rollout.main([bundle, f"--config_path={config}"])
    with pytest.raises(SystemExit):
        bundle_rollout.main([bundle, "--robot.type=robot_of_a_plugin"])
    assert loaded == [("camera",), ("robot", "camera"), ("robot", "camera")]


def test_bundle_mode_refuses_the_options_it_does_not_have(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle_rollout, "register_third_party_plugins", lambda kinds=(): None)
    bundle = make_numpy_bundle(tmp_path / "bundle")
    with pytest.raises(SystemExit):
        bundle_rollout.main([f"--bundle.path={bundle}", "--robot.type=mock_robot", "--dataset.repo_id=a/b"])


@pytest.mark.dataset
@pytest.mark.parametrize(
    ("argv", "mode"),
    [
        (["--bundle.path=b", "--robot.type=mock_robot"], "bundle"),
        (["--robot.type=mock_robot", "--bundle.path", "b", "--help"], "bundle"),
        (["--policy.path=p", "--robot.type=mock_robot"], "policy"),
        (["--bundle.pathX=b", "--help"], "policy"),
        (["--bundle.pat=b", "--help"], "policy"),
        (["--bundle=b"], "policy"),
    ],
)
def test_the_command_picks_the_mode_from_bundle_path_only(monkeypatch, argv, mode):
    chosen = []
    monkeypatch.setattr(sys, "argv", ["lerobot-rollout", *argv])
    monkeypatch.setattr("lerobot.bundle.rollout.main", lambda: chosen.append(("bundle", sys.argv[1:])))
    monkeypatch.setattr(
        "lerobot.scripts.lerobot_rollout.main", lambda: chosen.append(("policy", sys.argv[1:]))
    )
    lerobot_rollout_entry.main()
    assert chosen == [(mode, argv)]


def test_help_with_a_bundle_path_is_the_bundle_mode_help(monkeypatch, capsys):
    monkeypatch.setattr(bundle_rollout, "register_third_party_plugins", lambda kinds=(): None)
    monkeypatch.setattr(sys, "argv", ["lerobot-rollout", "--bundle.path=b", "--help"])
    with pytest.raises(SystemExit) as stop:
        lerobot_rollout_entry.main()
    assert stop.value.code == 0
    help_text = capsys.readouterr().out
    assert "--bundle.path" in help_text and "--policy.path" not in help_text


@pytest.mark.dataset
def test_the_policy_mode_points_a_bundle_to_the_bundle_mode(tmp_path, monkeypatch):
    from lerobot.rollout import RolloutConfig
    from tests.mocks.mock_robot import MockRobotConfig

    bundle = make_numpy_bundle(tmp_path / "bundle")
    monkeypatch.setattr(sys, "argv", ["lerobot-rollout", f"--policy.path={bundle}"])
    with pytest.raises(ValueError, match="lerobot-rollout --bundle.path="):
        RolloutConfig(robot=MockRobotConfig())
