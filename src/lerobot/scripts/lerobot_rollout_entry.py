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

- Without `--mode`: run a PyTorch policy, see `lerobot.scripts.lerobot_rollout`. The arguments reach it
  unchanged.
- `--mode=bundle`: run a bundle written by `lerobot-export`, see `lerobot.bundle.rollout`. It never imports
  PyTorch, so this module imports nothing before the mode is known. `--mode=bundle --help` lists its options.
"""

import argparse


def main() -> None:
    """Entry point of `lerobot-rollout`."""
    # No help and no abbreviations: every other argument, `--help` included, belongs to the chosen mode.
    parser = argparse.ArgumentParser(prog="lerobot-rollout", add_help=False, allow_abbrev=False)
    parser.add_argument("--mode", choices=["bundle"])
    args, rest = parser.parse_known_args()
    if args.mode == "bundle":
        from lerobot.bundle.rollout import main as bundle_main

        bundle_main(rest)
    else:
        from lerobot.scripts.lerobot_rollout import main as policy_main

        policy_main()


if __name__ == "__main__":
    main()
