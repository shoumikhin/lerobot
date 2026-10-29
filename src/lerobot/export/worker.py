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

"""The steps of `lerobot-export` that use PyTorch, each run in a process of its own.

`lerobot-export` never imports PyTorch. It runs `python -m lerobot.export.worker <request.json>` once to
trace the policy's parts, then once per part to compile it. Each step ends with its process, so the memory
it used is back with the system before the next step starts, which is what lets a large policy export on a
small board. A request is a JSON object naming its `step` and the `result` file to write.
"""

import json
import logging
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import draccus
import torch

from lerobot.bundle.manifest import (
    CONSTANTS,
    EMBEDDINGS,
    KNOWN_TASKS,
    TOKENIZER,
    Bundle,
    BundleManifest,
    ProgramFiles,
    ProgramRun,
    TaskTokens,
)
from lerobot.bundle.target import runtime_versions
from lerobot.configs import PreTrainedConfig, parser
from lerobot.policies.factory import get_export_recipe_class
from lerobot.utils import init_logging, register_third_party_plugins
from lerobot.utils.import_utils import PLUGIN_KINDS

from . import backends  # noqa: F401
from .bundle import write_bundle
from .configs import ExportBackendConfig, ExportConfig

logger = logging.getLogger(__name__)


def trace_step(request: dict[str, Any]) -> dict[str, Any]:
    """Parse the command line `request["args"]`, trace and save each part, and write the rest of the bundle.

    The bundle goes into a new folder beside the output folder. Returns that folder, its manifest, and what is
    left to compile.
    """
    # `--help` and parsing errors name the command the user ran.
    sys.argv = ["lerobot-export", *request["args"]]
    return cast(Callable[[], dict[str, Any]], _trace)()


@parser.wrap()
def _trace(cfg: ExportConfig) -> dict[str, Any]:
    output_dir = cast(Path, cfg.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    # The traced parts wait here, beside the bundle, until they are compiled.
    traced = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-traced-", dir=output_dir.parent))
    try:
        recipe = get_export_recipe_class(cast(PreTrainedConfig, cfg.policy)).load(cfg)
        programs: dict[str, ProgramFiles] = {}
        runs: list[ProgramRun] = []
        for part in recipe.parts():
            logger.info("Tracing %s", part.name)
            with torch.no_grad():
                program = torch.export.export(part.module, part.example_inputs)
            torch.export.save(program, traced / f"{part.name}.pt2")
            programs[part.name] = ProgramFiles(f"{part.name}{cfg.backend.file_suffix}")
            runs.append(ProgramRun(part.name, part.name, part.inputs, part.outputs))
            del part, program
        plan = recipe.plan()
        test_case = recipe.test_case()
        task_tokens = plan.task_tokens
        manifest = BundleManifest(
            backend=cfg.backend.type,
            runtime=cfg.backend.runtime,
            runtime_imports=list(cfg.backend.runtime_imports),
            programs=programs,
            runs=runs if plan.runs is None else plan.runs,
            constants=CONSTANTS if plan.constants else None,
            task_tokens=None
            if task_tokens is None
            else TaskTokens(
                task_tokens.prompt_ops, TOKENIZER, EMBEDDINGS, KNOWN_TASKS, task_tokens.max_tokens
            ),
            noise_shape=None if test_case.noise is None else list(test_case.noise.shape),
            task=recipe.task,
            action_names=getattr(recipe.config, "action_feature_names", None),
            action_scale=test_case.action_scale.tolist(),
            tolerance=recipe.tolerance,
            target=cfg.backend.target(),
            runtime_versions=runtime_versions(cfg.backend.runtime_modules),
        )
        write_bundle(Bundle(folder, manifest), test_case, recipe.config, plan.constants, task_tokens)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        shutil.rmtree(traced, ignore_errors=True)
        raise
    return {
        "folder": str(folder),
        "output_dir": str(output_dir),
        "manifest": manifest.to_dict(),
        "traced": str(traced),
        "backend": {"type": cfg.backend.type, **draccus.encode(cfg.backend)},
    }


def compile_step(request: dict[str, Any]) -> dict[str, Any]:
    """Compile the saved program `request["program"]` with `request["backend"]` into `request["path"]`.

    Returns the name of the data file the backend wrote beside it, if any.
    """
    backend = draccus.decode(ExportBackendConfig, request["backend"])
    path = Path(request["path"])
    logger.info("Compiling %s with %s", path.stem, backend.type)
    try:
        # Lowering needs no autograd, and some of ExecuTorch's passes fail when it records them (complex views).
        with torch.no_grad():
            # The trace step of this export wrote the program.
            data = backend.compile(torch.export.load(request["program"]), path)  # nosec B614
    except ImportError as error:
        if not backend.packages:
            raise
        packages = " ".join(f'"{package}"' for package in backend.packages)
        raise ImportError(f"The {backend.type} backend needs: pip install {packages}") from error
    return {"data": None if data is None else data.name}


STEPS = {"trace": trace_step, "compile": compile_step}


def main() -> None:
    """Run the step `sys.argv[1]` names and write its result."""
    request = json.loads(Path(sys.argv[1]).read_text())
    init_logging()
    register_third_party_plugins(kinds=(*PLUGIN_KINDS, "backend"))
    # The test case is the policy's float32 arithmetic, and the compiled programs must match it. TF32, which
    # PyTorch uses for cuDNN convolutions by default, moves a float32 policy's actions by about 1e-3 of their
    # spread, in the test case and in kernels a compiler builds from PyTorch's.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = STEPS[request["step"]](request)
    Path(request["result"]).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
