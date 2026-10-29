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

"""The `lerobot-rollout` command: choose the mode, then load only what that mode needs.

- With `--bundle.path`: run a bundle written by `lerobot-export`, see `lerobot.bundle.rollout`. It never imports
  PyTorch, so this module imports nothing before the mode is known. `--bundle.path=<folder> --help` lists its
  options.
- Without it: run a PyTorch policy, see `lerobot.scripts.lerobot_rollout`.

Either mode gets the arguments unchanged. A `--bundle.path` set only in a `--config_path` file does not select
the bundle mode.
"""

import argparse


def main() -> None:
    """Entry point of `lerobot-rollout`."""
    # No help and no abbreviations: every argument, `--help` included, belongs to the chosen mode.
    parser = argparse.ArgumentParser(prog="lerobot-rollout", add_help=False, allow_abbrev=False)
    parser.add_argument("--bundle.path", dest="bundle_path")
    args, _ = parser.parse_known_args()
    if args.bundle_path is None:
        from lerobot.scripts.lerobot_rollout import main as policy_main

        policy_main()
    else:
        from lerobot.bundle.rollout import main as bundle_main

        bundle_main()


if __name__ == "__main__":
    main()
