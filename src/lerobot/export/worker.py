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
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import draccus
import torch

from lerobot.bundle.manifest import (
    CONSTANTS,
    EMBEDDINGS,
    KNOWN_TASKS,
    NOISE,
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
from lerobot.utils.import_utils import load_plugin
from lerobot.utils.robot_utils import MOTOR_CHANNEL_SUFFIXES

from . import backends  # noqa: F401
from .bundle import write_bundle
from .configs import ExportBackendConfig, ExportConfig

# By name: run with `python -m`, this module is `__main__`, which is not a LeRobot logger.
logger = logging.getLogger("lerobot.export.worker")


def _robot_action_names(names: list[str] | None) -> list[str] | None:
    """The checkpoint's action names if they are a robot's, which `lerobot-rollout` matches by name.

    A robot's actions are position and velocity channels, so names without `.pos` or `.vel` (Cosmos3 names
    its channels by type, `pos_0`) match no robot: the robot is then matched by the number of actions, as for
    a PyTorch policy. A checkpoint can also save an empty list, which names no actions.
    """
    return names if names and all(name.endswith(MOTOR_CHANNEL_SUFFIXES) for name in names) else None


class _DrawNoise(torch.nn.Module):
    """The starting noise of the runs, standard normal, drawn on the device the program runs on."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype, device: str):
        super().__init__()
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def forward(self) -> torch.Tensor:
        return torch.randn(self.shape, dtype=self.dtype, device=self.device)


def set_log_level(level: str) -> None:
    """Log LeRobot's lines from `level` up.

    At INFO other libraries log only their warnings. DEBUG adds every library's logs, such as those of ExecuTorch
    and of the backend's compiler, and the INFO logs of torch.export and Inductor.
    """
    init_logging(console_level=level)
    logging.getLogger("lerobot").setLevel(level)
    if level == "INFO":
        # A compiler's own INFO lines, such as one per node Torch-TensorRT converts, would bury the export's steps.
        logging.getLogger().setLevel(logging.WARNING)
    else:
        # PyTorch's loggers print through handlers of their own, only warnings unless asked. Its other logs, such
        # as every graph or DEBUG, run to thousands of lines for even a tiny policy: TORCH_LOGS asks for them.
        torch._logging.set_logs(export=logging.INFO, inductor=logging.INFO)


def trace_step(request: dict[str, Any]) -> dict[str, Any]:
    """Parse the command line `request["args"]`, trace and save each part, and write the rest of the bundle.

    The bundle goes into a new folder beside the output folder. The file `request["folders"]` names the folders
    the step makes there, and the parents of the output folder it makes. Returns the bundle folder, its manifest,
    what is left to compile, and the folders it made.
    """
    argv = sys.argv
    # `--help` and parsing errors name the command the user ran.
    sys.argv = ["lerobot-export", *request["args"]]
    try:
        return cast(Callable[[dict[str, Any]], dict[str, Any]], _trace)(request)
    finally:
        sys.argv = argv


@parser.wrap()
def _trace(cfg: ExportConfig, request: dict[str, Any]) -> dict[str, Any]:
    set_log_level(cfg.log_level)
    output_dir = cast(Path, cfg.output_dir)
    tag = uuid.uuid4().hex[:8]
    folder = output_dir.parent / f".{output_dir.name}-{tag}"
    # The traced parts wait here, beside the bundle, until they are compiled.
    traced = output_dir.parent / f".{output_dir.name}-traced-{tag}"
    # Deepest first, so each is empty by the time it is removed.
    parents = [path for path in (output_dir.parent, *output_dir.parent.parents) if not path.exists()]
    folders = {"made": [str(folder), str(traced)], "parents": [str(path) for path in parents]}
    # Named before any is made, so that lerobot-export removes them however this step ends.
    Path(request["folders"]).write_text(json.dumps(folders))
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        # Made the way the output folder it becomes would be, so the bundle gets the usual permissions.
        folder.mkdir()
        traced.mkdir(mode=0o700)
        recipe = get_export_recipe_class(cast(PreTrainedConfig, cfg.policy)).load(cfg)
        programs: dict[str, ProgramFiles] = {}
        runs: list[ProgramRun] = []
        for part in recipe.parts():
            logger.info("Tracing %s", part.name)
            # Non-strict, the default of PyTorch 2.15, on every PyTorch: PyTorch 2.7 traces strictly by
            # default, and its Dynamo cannot run some policies' Python, such as ACT's.
            with torch.no_grad():
                program = torch.export.export(part.module, part.example_inputs, strict=False)
            torch.export.save(program, traced / f"{part.name}.pt2")
            programs[part.name] = ProgramFiles(f"{part.name}{cfg.backend.file_suffix}")
            runs.append(ProgramRun(part.name, part.name, part.inputs, part.outputs))
            del part, program
        plan = recipe.plan()
        test_case = recipe.test_case()
        runs = runs if plan.runs is None else plan.runs
        if test_case.noise is not None and cfg.backend.device == "cuda":
            # Drawn where the programs read it, so a chunk uploads only the robot's frame.
            noise = torch.from_numpy(test_case.noise)
            draw = _DrawNoise(tuple(noise.shape), noise.dtype, cfg.backend.device)
            torch.export.save(torch.export.export(draw, ()), traced / f"{NOISE}.pt2")
            programs[NOISE] = ProgramFiles(f"{NOISE}{cfg.backend.file_suffix}")
            runs = [ProgramRun(NOISE, NOISE, [], [NOISE]), *runs]
        task_tokens = plan.task_tokens
        manifest = BundleManifest(
            backend=cfg.backend.type,
            runtime=cfg.backend.runtime,
            runtime_imports=list(cfg.backend.runtime_imports),
            runtime_options=cfg.backend.runtime_options,
            programs=programs,
            runs=runs,
            constants=CONSTANTS if plan.constants else None,
            task_tokens=None
            if task_tokens is None
            else TaskTokens(
                task_tokens.prompt_ops, TOKENIZER, EMBEDDINGS, KNOWN_TASKS, task_tokens.max_tokens
            ),
            noise_shape=None if test_case.noise is None else list(test_case.noise.shape),
            task=recipe.task,
            action_names=_robot_action_names(getattr(recipe.config, "action_feature_names", None)),
            action_scale=test_case.action_scale.tolist(),
            tolerance=recipe.tolerance,
            target=cfg.backend.target(),
            runtime_versions=runtime_versions(cfg.backend.runtime_modules),
            exact_runtime_modules=list(cfg.backend.exact_runtime_modules),
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
        "log_level": cfg.log_level,
    }


def compile_step(request: dict[str, Any]) -> dict[str, Any]:
    """Compile the saved program `request["program"]` with `request["backend"]` into `request["path"]`.

    Returns the name of the data file the backend wrote beside it, if any.
    """
    for plugin in request["plugins"]:
        load_plugin(plugin)
    backend = draccus.decode(ExportBackendConfig, request["backend"])
    path = Path(request["path"])
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
    # The trace step learns the level from the command line it parses.
    set_log_level(request.get("log_level", "INFO"))
    register_third_party_plugins(kinds=("policy", "backend"))
    # The test case is the policy's float32 arithmetic, and the compiled programs must match it. TF32, which
    # PyTorch uses for cuDNN convolutions by default, moves a float32 policy's actions by about 1e-3 of their
    # spread, in the test case and in kernels a compiler builds from PyTorch's.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = STEPS[request["step"]](request)
    Path(request["result"]).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
