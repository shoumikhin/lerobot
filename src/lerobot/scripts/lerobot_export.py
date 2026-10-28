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

"""Compile a trained policy into a bundle that `lerobot-rollout` runs without PyTorch.

Export on the kind of board that will run the policy, then run the bundle like a checkpoint:

::

    lerobot-export \\
        --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model \\
        --backend.type=executorch_tensorrt

    lerobot-rollout \\
        --policy.path=outputs/export/act_executorch_tensorrt \\
        --robot.type=so101_follower --robot.port=/dev/ttyACM0 --duration=30

Before writing the bundle, the export replays its test case through the compiled programs, as rollout
does before the robot moves, and writes nothing if the actions differ from PyTorch's by more than the
tolerance.
"""

import gc
import logging
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

from lerobot.configs import PreTrainedConfig, parser
from lerobot.export import backends  # noqa: F401
from lerobot.export.configs import ExportConfig
from lerobot.utils import init_logging, register_third_party_plugins

logger = logging.getLogger(__name__)


def compile_bundle(cfg: ExportConfig, folder: Path) -> None:
    """Compile the policy's parts into `folder` and write the rest of the bundle there."""
    from lerobot.export.bundle import write_bundle
    from lerobot.export.recipe import make_recipe
    from lerobot.inference.export import ExportManifest, ProgramSpec
    from lerobot.policies import get_policy_class, make_pre_post_processors

    config = cast(PreTrainedConfig, cfg.policy)
    config.device = cfg.backend.device
    # How the policy runs, not how it trained, decides what compiles it.
    if hasattr(config, "compile_model"):
        config.compile_model = False
    policy = get_policy_class(config.type).from_pretrained(str(config.pretrained_path), config=config)
    policy = policy.to(config.device).eval()
    overrides = {"device_processor": {"device": config.device}}
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=config.pretrained_path,
        preprocessor_overrides=overrides,
        postprocessor_overrides=overrides,
    )
    recipe = make_recipe(policy, preprocessor, postprocessor, cfg)
    programs = []
    for part in recipe.parts():
        file = f"{part.name}{cfg.backend.file_suffix}"
        logger.info("Compiling %s with %s", part.name, cfg.backend.type)
        cfg.backend.compile(part, folder / file)
        programs.append(ProgramSpec(part.name, file, part.input_names, part.output_names))
    test_case = recipe.test_case()
    manifest = ExportManifest(
        backend=cfg.backend.type,
        runtime=cfg.backend.runtime,
        runtime_imports=list(cfg.backend.runtime_imports),
        programs=programs,
        noise_shape=None if test_case.noise is None else list(test_case.noise.shape),
        task=recipe.task,
        tolerance=recipe.tolerance,
    )
    write_bundle(folder, manifest, test_case, recipe.config)


@parser.wrap()
def export(cfg: ExportConfig) -> Path:
    """Write the bundle, and return its folder."""
    import torch

    init_logging()
    output_dir = cast(Path, cfg.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        compile_bundle(cfg, folder)
        # The check loads the programs in a new process, as rollout does, so free what compiling held.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Checking the bundle")
        if subprocess.run([sys.executable, "-m", "lerobot.inference.export", str(folder)]).returncode:
            raise RuntimeError(
                "The bundle failed its check on this machine (see above), so it was not written."
            )
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    folder.rename(output_dir)
    logger.info("Wrote %s", output_dir)
    return output_dir


def main() -> None:
    """CLI entry point for `lerobot-export`."""
    register_third_party_plugins()
    cast(Callable[[], Path], export)()


if __name__ == "__main__":
    main()
