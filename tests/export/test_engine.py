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

"""The engine that runs a bundle, on bundles of small modules run by the test runtime."""

import json

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from lerobot.inference.export import MANIFEST, ExportBundle, ExportInferenceEngine
from lerobot.utils.constants import OBS_STATE
from tests.export.torch_backend import make_bundle

FRAME = {OBS_STATE: np.array([1.0, 2.0, 3.0], np.float32)}


class Chunk(nn.Module):
    """Three actions from a state: the state scaled by 1, 2 and 3."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([[1.0], [2.0], [3.0]]))

    def forward(self, state: Tensor) -> Tensor:
        return state[None] * self.scale


def test_engine_plays_a_chunk_then_runs_again(tmp_path):
    engine = ExportInferenceEngine(make_bundle(tmp_path / "bundle", Chunk(), FRAME))
    actions = [engine.get_action({OBS_STATE: np.ones(3, np.float32)}) for _ in range(4)]
    np.testing.assert_array_equal(np.stack(actions), [[1, 1, 1], [2, 2, 2], [3, 3, 3], [1, 1, 1]])


def test_engine_refuses_a_bundle_that_misses_its_test_case(tmp_path):
    with pytest.raises(RuntimeError, match="does not reproduce its test case"):
        ExportInferenceEngine(make_bundle(tmp_path / "bundle", Chunk(), FRAME, expected_offset=1.0))


def test_engine_refuses_non_finite_actions(tmp_path):
    engine = ExportInferenceEngine(make_bundle(tmp_path / "bundle", Chunk(), FRAME))
    with pytest.raises(RuntimeError, match="NaN or infinite"):
        engine.get_action({OBS_STATE: np.array([np.nan, 0.0, 0.0], np.float32)})


def test_engine_runs_only_the_task_it_was_exported_for(tmp_path):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME, task="pick")
    with pytest.raises(ValueError, match="runs only the task"):
        ExportInferenceEngine(bundle, task="place")
    engine = ExportInferenceEngine(bundle)
    assert engine.task == "pick"
    assert not engine.set_task("place")


def test_engine_refuses_a_robot_that_does_not_match(tmp_path):
    engine = ExportInferenceEngine(make_bundle(tmp_path / "bundle", Chunk(), FRAME))
    engine.check_robot({OBS_STATE: (3,)}, ["a", "b", "c"])
    with pytest.raises(ValueError, match=r"observation.state \(3,\) \(the robot gives \(4,\)\)"):
        engine.check_robot({OBS_STATE: (4,)}, ["a", "b", "c"])
    with pytest.raises(ValueError, match="3 action components"):
        engine.check_robot({OBS_STATE: (3,)}, ["a", "b"])


@pytest.mark.parametrize(
    ("change", "error"),
    [({"format_version": 2}, "format version 2"), ({"steps": ["model"]}, r"\['steps'\]")],
)
def test_bundle_refuses_a_manifest_it_cannot_run(tmp_path, change, error):
    bundle = make_bundle(tmp_path / "bundle", Chunk(), FRAME)
    manifest = json.loads((bundle.path / MANIFEST).read_text())
    (bundle.path / MANIFEST).write_text(json.dumps({**manifest, **change}))
    with pytest.raises(ValueError, match=error):
        ExportBundle.read(bundle.path)
