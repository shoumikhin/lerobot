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

"""The arithmetic behind the README's results: time per chunk and action error."""

import importlib
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "export"


@pytest.fixture
def examples(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    return importlib.import_module


def test_time_per_chunk_is_the_median_of_the_run_medians(examples):
    benchmark = examples("benchmark")
    # A slow outlier chunk moves no median; the slow run is outvoted by the other two.
    chunk_ms = np.array([[1.0, 2.0, 100.0], [3.0, 3.0, 3.0], [50.0, 60.0, 70.0]])
    assert benchmark.time_per_chunk(chunk_ms) == 3.0


def test_action_error_takes_each_frames_largest_difference(examples):
    benchmark = examples("benchmark")
    reference = np.zeros((20, 2, 3))
    actions = reference.copy()
    actions[:, 1, 2] = 0.5  # every frame is off by 0.5 on one joint of one action
    actions[7, 0, 0] = 10.0  # and one frame by 10
    mean, p95 = benchmark.action_error(actions, reference)
    assert mean == pytest.approx((19 * 0.5 + 10.0) / 20)
    assert p95 == pytest.approx(0.5 + 0.05 * (10.0 - 0.5))
