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
        --robot.type=so101_follower --robot.port=/dev/ttyACM0 \\
        --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
        --duration=30

Before writing the bundle, the export replays its test case through the compiled programs, as rollout
does before the robot moves, and writes nothing if the actions differ from PyTorch's by more than the
tolerance. A recipe may also have some parts checked on their own, each compiled program on the inputs the part
got in PyTorch.

This process never imports PyTorch. It traces the policy in one process, compiles each program in a process of
its own (on a CUDA device, for a policy that takes a noise, also the one that draws it), checks each part the
recipe asks for in one more process each (see `lerobot.export.worker`), then checks the bundle in another
(`python -m lerobot.bundle`), so each step starts with the memory the one before it used given back.
"""

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, cast

from lerobot.bundle.manifest import Bundle, BundleManifest
from lerobot.utils import init_logging
from lerobot.utils.import_utils import PLUGIN_DISCOVERY_SUFFIX, parse_plugin_args

logger = logging.getLogger(__name__)

# How often a step that is still running says so.
HEARTBEAT_SECONDS = 60


def export(args: Sequence[str]) -> Path:
    """Write the bundle the `lerobot-export` command line `args` asks for, and return its folder.

    `args` is the command line, such as `["--policy.path=...", "--backend.type=executorch_tensorrt"]`.
    """
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="lerobot-export-") as requests:
        work = Path(requests)
        folders = work / "folders.json"
        try:
            # Not `_step`: the trace step logs each part it traces, and `--help` should print only the help.
            with _heartbeat("Tracing the policy"):
                traced = _run_step(
                    work, "trace", {"step": "trace", "args": list(args), "folders": str(folders)}
                )
            if traced is None:
                # The command line asked for help, which the trace step printed.
                raise SystemExit(0)
            folder, output_dir = Path(traced["folder"]), Path(traced["output_dir"])
            # The trace step loaded these from the command line; a compile step needs the backend's.
            plugins = [
                path
                for key, path in parse_plugin_args(PLUGIN_DISCOVERY_SUFFIX, args).items()
                if key.startswith("backend.")
            ]
            manifest = BundleManifest.from_dict(traced["manifest"])
            steps = len(manifest.programs) + len(traced["checks"]) + 1
            logger.info("Traced the policy in %.0f s", time.monotonic() - start)
            step = 0
            for name, files in manifest.programs.items():
                step += 1
                request = {
                    "step": "compile",
                    "backend": traced["backend"],
                    "program": str(Path(traced["traced"]) / f"{name}.pt2"),
                    "path": str(folder / files.file),
                    "plugins": plugins,
                    "log_level": traced["log_level"],
                }
                with _step(f"[{step}/{steps}] Compiling {name} with {traced['backend']['type']}"):
                    files.data = cast(dict[str, Any], _run_step(work, f"compile-{name}", request))["data"]
                if name in traced["checks"]:
                    step += 1
                    check = {
                        "step": "check",
                        "backend": traced["backend"],
                        "check": str(Path(traced["traced"]) / f"{name}.check.safetensors"),
                        "path": request["path"],
                        "data": None if files.data is None else str(folder / files.data),
                        "plugins": plugins,
                        "log_level": traced["log_level"],
                    }
                    with _step(f"[{step}/{steps}] Checking the compiled {name} on its own inputs"):
                        _run_step(work, f"check-{name}", check)
            shutil.rmtree(traced["traced"])
            Bundle(folder, manifest).write()
            with _step(f"[{steps}/{steps}] Checking the bundle"):
                # -P: nothing loads from the current folder, which can hold any .py file.
                code = subprocess.run([sys.executable, "-P", "-m", "lerobot.bundle", str(folder)]).returncode
            if code:
                reason = f"crashed with {_signal_name(-code)}" if code < 0 else "failed its check (see above)"
                raise RuntimeError(f"The bundle {reason} on this machine, so it was not written.")
            # A rename onto an empty folder would replace it without a word, and one onto a link fails.
            if os.path.lexists(output_dir):
                raise RuntimeError(
                    f"{output_dir} was made while the export ran, so the bundle was not written."
                )
            folder.rename(output_dir)
        except BaseException:
            # A killed trace step could not remove its folders itself. It names them before it makes any, so
            # one killed while it wrote their names made none.
            with suppress(FileNotFoundError, ValueError):
                _remove_made(json.loads(folders.read_text()))
            raise
    logger.info("Wrote %s in %.0f s", output_dir, time.monotonic() - start)
    return output_dir


def _remove_made(folders: dict[str, list[str]]) -> None:
    """Remove the folders the trace step made beside the output folder, then the parents it made that are empty."""
    for path in folders["made"]:
        shutil.rmtree(path, ignore_errors=True)
    for path in folders["parents"]:
        # One that holds anything else, such as another export's bundle, stays.
        with suppress(OSError):
            Path(path).rmdir()


@contextmanager
def _heartbeat(doing: str) -> Iterator[None]:
    """Log every `HEARTBEAT_SECONDS` while the block runs that it is still `doing`, so a long step is not silent.

    A plain line each time, never a progress bar redrawn in place, which a log file or CI would show garbled.
    """
    start = time.monotonic()
    done = threading.Event()

    def beat() -> None:
        while not done.wait(HEARTBEAT_SECONDS):
            logger.info("%s: still running after %.0f s", doing, time.monotonic() - start)

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join()


@contextmanager
def _step(doing: str) -> Iterator[None]:
    """Log `doing` when the block starts, that it still runs while it does, and how long it took once done."""
    logger.info(doing)
    start = time.monotonic()
    with _heartbeat(doing):
        yield
    logger.info("%s: done in %.0f s", doing, time.monotonic() - start)


def _run_step(work: Path, name: str, request: dict[str, Any]) -> dict[str, Any] | None:
    """Run one step of the export in a new process, and return its result, or `None` if it wrote none."""
    request_file = work / f"{name}.json"
    result_file = work / f"{name}.result.json"
    request_file.write_text(json.dumps({**request, "result": str(result_file)}))
    # -P: nothing loads from the current folder, which can hold any .py file.
    code = subprocess.run([sys.executable, "-P", "-m", "lerobot.export.worker", str(request_file)]).returncode
    if code:
        reason = f"crashed with {_signal_name(-code)}" if code < 0 else "failed (see above)"
        raise RuntimeError(f"The {name} step {reason}, so the bundle was not written.")
    return json.loads(result_file.read_text()) if result_file.exists() else None


def _signal_name(number: int) -> str:
    try:
        return signal.Signals(number).name
    except ValueError:
        # Such as a Linux real-time signal, which has no name in Python.
        return f"signal {number}"


def main() -> None:
    """CLI entry point for `lerobot-export`."""
    init_logging()
    # SIGTERM (a `kill`, a timeout, a stopped container) and SIGHUP (a closed terminal or ssh session) would
    # end this process at once; as an exit each stops the running step and removes the unfinished folders, as
    # Ctrl-C does. A signal already ignored, such as SIGHUP under nohup, stays ignored. Windows has no SIGHUP.
    for stop in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if stop is not None and signal.getsignal(stop) is not signal.SIG_IGN:
            signal.signal(stop, lambda number, _: sys.exit(128 + number))
    try:
        export(sys.argv[1:])
    except RuntimeError as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
