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
small board. A request is a JSON object naming its `step`, the `imports` to make first, and the `result`
file to write.
"""

import importlib
import json
import logging
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import draccus
import torch
from torch._higher_order_ops.wrap import wrap_with_autocast

from lerobot.configs import PreTrainedConfig, parser
from lerobot.inference.bundle import BundleManifest, ProgramSpec
from lerobot.inference.bundle.manifest import (
    CONSTANTS,
    EMBEDDINGS_FILE,
    KNOWN_TASKS,
    TOKENIZER,
    TaskTokensSpec,
)
from lerobot.utils import init_logging, register_third_party_plugins

from . import backends  # noqa: F401
from .bundle import write_bundle
from .configs import ExportBackendConfig, ExportConfig
from .recipe import Part, ProgramCall, recipe_class

logger = logging.getLogger(__name__)


def export_part(part: Part) -> torch.export.ExportedProgram:
    """The part traced by `torch.export`, ready to save for the compile step."""
    with torch.no_grad():
        program = torch.export.export(part.module, part.example_inputs)
    if any(node.target is wrap_with_autocast for node in program.graph.nodes):
        # PyTorch saves some programs with autocast regions that it then cannot load. With an empty table,
        # run_decompositions only inlines the regions; no operator is decomposed.
        program = program.run_decompositions({})
    return program


def trace(request: dict[str, Any]) -> dict[str, Any]:
    """Parse the command line `request["args"]`, trace and save each part, and write the rest of the bundle.

    The bundle goes into a new folder beside the output folder. Returns that folder and what is left to compile.
    """
    # `--help` and parsing errors name the command the user ran.
    sys.argv = ["lerobot-export", *request["args"]]
    return cast(Callable[[], dict[str, Any]], _trace)()


@parser.wrap()
def _trace(cfg: ExportConfig) -> dict[str, Any]:
    # The test case is the policy's float32 arithmetic. TF32, which PyTorch uses for cuDNN convolutions by
    # default, would move the expected actions of a float32 policy by about 1e-3 of their spread.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output_dir = cast(Path, cfg.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    # The traced parts wait here, beside the bundle, until they are compiled.
    traced = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-traced-", dir=output_dir.parent))
    try:
        recipe = recipe_class(cast(PreTrainedConfig, cfg.policy)).load(cfg)
        files: dict[str, str] = {}
        own_names: list[ProgramCall] = []
        for part in recipe.parts():
            logger.info("Tracing %s", part.name)
            program = export_part(part)
            torch.export.save(program, traced / f"{part.name}.pt2")
            files[part.name] = f"{part.name}{cfg.backend.file_suffix}"
            own_names.append(ProgramCall(part.name, part.input_names, part.output_names))
            del part, program
        plan = recipe.plan()
        calls = own_names if plan.calls is None else plan.calls
        repeats = Counter(call.part for call in calls)
        seen: Counter[str] = Counter()
        programs = []
        for call in calls:
            if call.part not in files:
                raise ValueError(f"{type(recipe).__name__} runs a part it did not yield: {call.part!r}.")
            name = call.part if repeats[call.part] == 1 else f"{call.part}_{seen[call.part]}"
            seen[call.part] += 1
            programs.append(ProgramSpec(name, files[call.part], call.inputs, call.outputs))
        test_case = recipe.test_case()
        prompt = plan.task_prompt
        task_tokens = (
            None
            if prompt is None
            else TaskTokensSpec(prompt.text, TOKENIZER, EMBEDDINGS_FILE, KNOWN_TASKS, prompt.max_tokens)
        )
        manifest = BundleManifest(
            backend=cfg.backend.type,
            runtime=cfg.backend.runtime,
            runtime_imports=list(cfg.backend.runtime_imports),
            programs=programs,
            constants=CONSTANTS if plan.constants else None,
            task_tokens=task_tokens,
            noise_shape=None if test_case.noise is None else list(test_case.noise.shape),
            history=plan.history,
            task=recipe.task,
            action_names=getattr(recipe.config, "action_feature_names", None),
            action_scale=test_case.action_scale.tolist(),
            tolerance=recipe.tolerance,
        )
        write_bundle(folder, manifest, test_case, recipe.config, plan.constants, prompt)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        shutil.rmtree(traced, ignore_errors=True)
        raise
    return {
        "folder": str(folder),
        "output_dir": str(output_dir),
        "traced": str(traced),
        "backend": {"type": cfg.backend.type, **draccus.encode(cfg.backend)},
        # A backend from a plugin is registered by importing the module that defines it.
        "backend_module": type(cfg.backend).__module__,
        "parts": [{"program": str(traced / f"{part}.pt2"), "file": file} for part, file in files.items()],
    }


def compile_part(request: dict[str, Any]) -> dict[str, Any]:
    """Compile the saved program `request["program"]` with `request["backend"]` into `request["path"]`.

    Returns the name of the data file the backend wrote beside it, if any.
    """
    backend = draccus.decode(ExportBackendConfig, request["backend"])
    path = Path(request["path"])
    logger.info("Compiling %s with %s", path.stem, backend.type)
    # The trace step of this export wrote the program.
    data = backend.compile(torch.export.load(request["program"]), path)  # nosec B614
    return {"data": None if data is None else data.name}


STEPS = {"trace": trace, "compile": compile_part}


def main() -> None:
    """Run the step `sys.argv[1]` names and write its result."""
    request = json.loads(Path(sys.argv[1]).read_text())
    init_logging()
    register_third_party_plugins()
    for module in request["imports"]:
        importlib.import_module(module)
    result = STEPS[request["step"]](request)
    Path(request["result"]).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
