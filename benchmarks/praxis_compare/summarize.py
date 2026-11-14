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

"""Print each side's per-pass medians and the median of those medians, from the logs run_ab.sh wrote.

usage: python summarize.py <out dir> <passes> <side> [<side> ...]

Exits non-zero when a side has fewer than <passes> clean passes, or when a pass records no sampler settings or
the passes ran different ones.
"""

import json
import statistics
import sys
from pathlib import Path


def main() -> None:
    out, passes, sides = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
    ok = True
    samplers: dict[str, list[str]] = {}
    for side in sides:
        medians = []
        for i in range(1, passes + 1):
            log = out / f"{side}_pass{i}.log"
            lines = (
                [ln for ln in log.read_text().splitlines() if ln.startswith("SUMMARY ")]
                if log.exists()
                else []
            )
            if lines:
                summary = json.loads(lines[-1][len("SUMMARY ") :])
                medians.append(summary["median_ms"])
                if not summary.get("sampler"):
                    print(f"{side} pass {i} records no sampler settings")
                    ok = False
                samplers.setdefault(json.dumps(summary.get("sampler"), sort_keys=True), []).append(
                    f"{side} {i}"
                )
        line = f"{side:10s} {len(medians)}/{passes} passes, per-pass medians {medians} ms"
        print(line + (f", median of medians {statistics.median(medians)} ms" if medians else ""))
        ok &= len(medians) == passes
    if len(samplers) > 1:
        print("the passes ran different sampler settings:", json.dumps(samplers))
        ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
