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


"""Check a bundle on this machine: `python -m lerobot.bundle <bundle folder>`.

It loads the programs and replays the test case as `lerobot-rollout` does before the robot moves, and exits
with an error if the actions differ by more than the bundle's tolerance.
"""

import sys

from lerobot.utils import init_logging

from .engine import BundleEngine
from .manifest import Bundle

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m lerobot.bundle <bundle folder>")
    init_logging()
    BundleEngine(Bundle.read(sys.argv[1]))
