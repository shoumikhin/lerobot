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

"""Print each side's passes and the median of the valid passes' medians, from the logs run_ab.sh wrote.

usage: python summarize.py <out dir> <passes> <side> [<side> ...]

A pass is invalid when its median is more than 1.25 times the median of the same side's other passes' medians, or
when a warm-up chunk after the first takes more than 10 s. Other work on a shared machine, or a first pass that
fills cold compile caches, can stretch a pass like that. The headline leaves invalid passes out; the 90th percentile is printed but never excludes a pass.
Exits non-zero when a side has fewer than <passes> clean passes or no valid pass, or when a pass records no sampler
settings or the passes ran different ones.
"""

import json
import statistics
import sys
from pathlib import Path


def invalid_reasons(median: float, others: list[float], warmup_ms: list[float]) -> list[str]:
    reasons = []
    if others and median > 1.25 * statistics.median(others):
        reasons.append(f"median above 1.25x the other passes' {statistics.median(others)} ms")
    late = [ms for ms in warmup_ms[1:] if ms > 10000]
    if late:
        reasons.append(f"warm-up chunk of {late[0] / 1000:.1f} s after the first")
    return reasons


def main() -> None:
    out, passes, sides = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
    ok = True
    samplers: dict[str, list[str]] = {}
    for side in sides:
        summaries = {}
        for i in range(1, passes + 1):
            log = out / f"{side}_pass{i}.log"
            lines = (
                [ln for ln in log.read_text().splitlines() if ln.startswith("SUMMARY ")]
                if log.exists()
                else []
            )
            if lines:
                summary = json.loads(lines[-1][len("SUMMARY ") :])
                summaries[i] = summary
                if not summary.get("sampler"):
                    print(f"{side} pass {i} records no sampler settings")
                    ok = False
                samplers.setdefault(json.dumps(summary.get("sampler"), sort_keys=True), []).append(
                    f"{side} {i}"
                )
        valid = []
        for i, summary in summaries.items():
            median = summary["median_ms"]
            others = [s["median_ms"] for k, s in summaries.items() if k != i]
            reasons = invalid_reasons(median, others, summary.get("warmup_ms") or [])
            if not reasons:
                valid.append(median)
            verdict = "invalid: " + "; ".join(reasons) if reasons else "valid"
            print(f"{side:10s} pass {i}: median {median} ms, p90 {summary.get('p90_ms')} ms, {verdict}")
        line = f"{side:10s} {len(summaries)}/{passes} passes, valid medians {valid} ms"
        print(
            line
            + (f", median of medians {round(statistics.median(valid), 2)} ms" if valid else ", no valid pass")
        )
        ok &= len(summaries) == passes and bool(valid)
    if len(samplers) > 1:
        print("the passes ran different sampler settings:", json.dumps(samplers))
        ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
