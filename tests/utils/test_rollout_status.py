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

"""The rollout status logs run without PyTorch; `tests/test_rollout.py` covers what they report."""

from tests.torch_free import assert_runs_without_torch


def test_status_logs_run_without_torch():
    assert_runs_without_torch(
        """
        import logging
        import time

        from lerobot.utils.cycle_timer import CycleTimer
        from lerobot.utils.latency import LatencyWindow
        from lerobot.utils.rollout_status import RolloutStatusReporter

        logger = logging.getLogger("lerobot.utils.rollout_status")
        logger.setLevel(logging.INFO)
        lines = []
        logger.addFilter(lambda record: lines.append(record.getMessage()))
        latency = LatencyWindow()
        latency.configure(0.02)
        reporter = RolloutStatusReporter(latency, 0.02, device=None)
        timer = CycleTimer(200)
        for _ in range(20):
            timer.tick()
            reporter.report_if_due(timer)
            latency.record(0.003, time.monotonic())
            timer.wait()
        assert lines and all(line.startswith("Rollout: loop ") for line in lines), lines
        assert any("infer 3.0 ms" in line for line in lines), lines
        """
    )
