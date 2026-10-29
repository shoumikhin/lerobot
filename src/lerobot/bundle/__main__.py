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


"""Check a bundle on this machine: `python -m lerobot.bundle <bundle folder>`, as `lerobot-check-bundle` does.

Prefer `lerobot-check-bundle` inside a bundle's folder: `python -m` puts the current folder first on the module
search path before Python imports `lerobot` and parts of the standard library, so a .py file shipped with the
bundle could load in place of one of them.
"""

import os
import sys

if __name__ == "__main__":
    # Otherwise a file the bundle ships would load in place of a module this check imports, such as NumPy.
    # Under `python -P` the first entry is not the current folder but the first of PYTHONPATH, which stays.
    if not sys.flags.safe_path and sys.path[0] in ("", os.getcwd()):
        sys.path.pop(0)
    from lerobot.scripts.lerobot_check_bundle import main

    main()
