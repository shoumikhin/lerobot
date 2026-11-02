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

"""The dispatch helpers run a NumPy engine without PyTorch; `tests/inference` covers their contract."""

from tests.torch_free import assert_runs_without_torch


def test_dispatch_tick_runs_a_numpy_engine_without_torch():
    assert_runs_without_torch(
        """
        import numpy as np

        from lerobot.utils import ActionInterpolator
        from lerobot.utils.action_dispatch import dispatch_tick


        class Engine:
            def __init__(self):
                self.actions = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
                self.dispatched = 0

            def dispatch_allowed(self):
                return True

            def acknowledge_hold(self):
                pass

            def get_action(self, obs_frame):
                return self.actions.pop(0)

            def record_dispatch(self):
                self.dispatched += 1


        engine, sent = Engine(), []
        interpolator = ActionInterpolator(multiplier=2)
        for _ in range(3):
            dispatch_tick(engine, interpolator, {}, send=sent.append)
        assert [list(target) for target in sent] == [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], sent
        assert engine.dispatched == 3
        """
    )
