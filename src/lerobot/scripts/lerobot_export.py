#!/usr/bin/env python

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

"""Compile a trained policy into a bundle that `lerobot-rollout` runs instead of the PyTorch policy.

Export on the kind of board that will run the policy, then run the bundle with `lerobot-rollout`:

::

    lerobot-export \\
        --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model \\
        --backend.type=executorch_tensorrt

    lerobot-rollout \\
        --bundle.path=outputs/export/act_executorch_tensorrt \\
        --robot.type=so101_follower --robot.port=/dev/ttyACM0 --duration=30

Before writing the bundle, the export replays its test case through the compiled programs, as rollout
does before the robot moves, and writes nothing if the actions differ from PyTorch's by more than the
tolerance.

This process never imports PyTorch. It traces the policy in one process, compiles each part in a process
of its own, and checks the bundle in another (see `lerobot.export.worker`), so each step starts with the
memory the one before it used given back.
"""

import json
import logging
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from lerobot.bundle.manifest import Bundle, BundleManifest
from lerobot.utils import init_logging

logger = logging.getLogger(__name__)


def export(args: Sequence[str]) -> Path:
    """Write the bundle the `lerobot-export` command line `args` asks for, and return its folder.

    `args` is the command line, such as `["--policy.path=...", "--backend.type=executorch_tensorrt"]`.
    """
    with tempfile.TemporaryDirectory(prefix="lerobot-export-") as requests:
        work = Path(requests)
        traced = _run_step(work, "trace", {"step": "trace", "args": list(args)})
        if traced is None:
            # The command line asked for help, which the trace step printed.
            raise SystemExit(0)
        folder, output_dir = Path(traced["folder"]), Path(traced["output_dir"])
        try:
            manifest = BundleManifest.from_dict(traced["manifest"])
            for name, files in manifest.programs.items():
                request = {
                    "step": "compile",
                    "backend": traced["backend"],
                    "program": str(Path(traced["traced"]) / f"{name}.pt2"),
                    "path": str(folder / files.file),
                }
                files.data = cast(dict[str, Any], _run_step(work, f"compile-{name}", request))["data"]
            shutil.rmtree(traced["traced"])
            Bundle(folder, manifest).write()
            logger.info("Checking the bundle")
            code = subprocess.run([sys.executable, "-m", "lerobot.bundle", str(folder)]).returncode
            if code:
                reason = (
                    f"crashed with {signal.Signals(-code).name}"
                    if code < 0
                    else "failed its check (see above)"
                )
                raise RuntimeError(f"The bundle {reason} on this machine, so it was not written.")
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            shutil.rmtree(traced["traced"], ignore_errors=True)
            raise
    folder.rename(output_dir)
    logger.info("Wrote %s", output_dir)
    return output_dir


def _run_step(work: Path, name: str, request: dict[str, Any]) -> dict[str, Any] | None:
    """Run one step of the export in a new process, and return its result, or `None` if it wrote none."""
    request_file = work / f"{name}.json"
    result_file = work / f"{name}.result.json"
    request_file.write_text(json.dumps({**request, "result": str(result_file)}))
    code = subprocess.run([sys.executable, "-m", "lerobot.export.worker", str(request_file)]).returncode
    if code:
        reason = f"crashed with {signal.Signals(-code).name}" if code < 0 else "failed (see above)"
        raise RuntimeError(f"The {name} step {reason}, so the bundle was not written.")
    return json.loads(result_file.read_text()) if result_file.exists() else None


def main() -> None:
    """CLI entry point for `lerobot-export`."""
    init_logging()
    try:
        export(sys.argv[1:])
    except RuntimeError as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
