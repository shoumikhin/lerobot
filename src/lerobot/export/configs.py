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

"""Configuration of `lerobot-export`, and the base of its compiler backends."""

from __future__ import annotations

import abc
import contextlib
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import draccus
from packaging.version import Version

from lerobot.bundle.manifest import Target
from lerobot.bundle.target import cuda_gpus
from lerobot.configs import PreTrainedConfig, parser
from lerobot.configs.default import DatasetConfig

if TYPE_CHECKING:
    from torch import Tensor, nn
    from torch.export import ExportedProgram


@dataclass
class ExportBackendConfig(draccus.ChoiceRegistry, abc.ABC):
    """A compiler that turns each part of a policy into one program file. Chosen with `--backend.type`.

    A backend knows its compiler, never a policy. Backends whose files the same runtime loads share that
    runtime and `lerobot-rollout` unchanged.

    Attributes:
        runtime (`str`):
            The runtime that loads the program files, such as `executorch`.
        file_suffix (`str`):
            The suffix of the program files, such as `.pte`.
        device (`str`):
            Where the programs run, such as `cuda` or `cpu`. The parts are traced and compiled there too.
        runtime_imports (`tuple[str, ...]`):
            Python modules the runtime imports before it loads the files, such as one that registers the
            compiler's kernels with the runtime.
        runtime_options (`dict[str, dict[str, bool | int | str]]`):
            Settings the runtime applies before it loads the files. For ExecuTorch, the options of its
            backends, by backend name.
        runtime_modules (`tuple[str, ...]`):
            Python modules the files need to load, such as `executorch`. The bundle records their versions as
            its `runtime_versions`.
        exact_runtime_modules (`tuple[str, ...]`):
            The modules of `runtime_modules` the files load only at the version that built them, such as
            `tensorrt`. A bundle is refused where another version of one is installed.
        packages (`tuple[str, ...]`):
            What to install beside LeRobot to export with this backend, such as `"executorch>=1.6.0,<1.7.0"`.
        min_torch (`str`, *optional*):
            The oldest PyTorch whose traced programs the compiler reads, such as `"2.15"`. `None` takes any.
        torch_pins (`tuple[str, ...]`):
            The PyTorch pins an export with this backend needs, such as
            `("torch>=2.15,<2.16", "torchvision>=0.29.2,<0.30")`. The refusal of an older PyTorch prints them
            with `packages` as one install command.
        quantizations (`tuple[str, ...]`):
            The `--quantize` modes the backend offers, such as `fp8`. Empty for none.
        quantization_imports (`dict[str, str]`):
            For a `--quantize` mode whose parts hold operators that a Python module registers on import, that
            module. The compile step imports it before it loads such a part.
    """

    runtime: ClassVar[str]
    file_suffix: ClassVar[str]
    device: ClassVar[str]
    runtime_imports: ClassVar[tuple[str, ...]] = ()
    runtime_options: ClassVar[dict[str, dict[str, bool | int | str]]] = {}
    runtime_modules: ClassVar[tuple[str, ...]] = ()
    exact_runtime_modules: ClassVar[tuple[str, ...]] = ()
    packages: ClassVar[tuple[str, ...]] = ()
    min_torch: ClassVar[str | None] = None
    torch_pins: ClassVar[tuple[str, ...]] = ()
    quantizations: ClassVar[tuple[str, ...]] = ()
    quantization_imports: ClassVar[dict[str, str]] = {}

    def target(self) -> Target | None:
        """The GPU the files are built on when `device` is `cuda`, since they then run only on that kind of GPU.

        `None` for files that run anywhere their runtime does.
        """
        if self.device != "cuda":
            return None
        # Read the way the engine reads the GPU it checks, so the two compare like with like.
        gpu, compute_capability = cuda_gpus()[0]
        return Target("cuda", gpu, compute_capability)

    @property
    def type(self) -> str:
        """The `--backend.type` name of this backend."""
        return self.get_choice_name(self.__class__)

    def check_torch(self) -> None:
        """Refuse a PyTorch older than `min_torch` before anything is traced, and say how to export."""
        if self.min_torch is None:
            return
        import torch

        installed = torch.__version__
        if Version(installed).release >= Version(self.min_torch).release:
            return
        message = (
            f"The {self.type} backend needs PyTorch {self.min_torch} or newer to export, but PyTorch {installed} "
            "is installed."
        )
        if self.torch_pins:
            pins = " ".join(f'"{pin}"' for pin in self.torch_pins + self.packages)
            message += f" Install it in this environment:\n\n    pip install {pins}\n\nThen export again."
        raise ValueError(f"{message} See https://huggingface.co/docs/lerobot/export.")

    def quantized(
        self, module: nn.Module, example_inputs: tuple[Tensor, ...], mode: str
    ) -> contextlib.AbstractContextManager:
        """Quantize the part `module` in place as `--quantize=mode` asks, and return the context to trace it in.

        `mode` is one of `quantizations`. `example_inputs` are the part's own inputs, for a mode that measures
        the range of its activations on them.
        """
        raise NotImplementedError(f"The {self.type} backend offers no quantization.")

    @abc.abstractmethod
    def compile(self, program: ExportedProgram, path: Path) -> Path | None:
        """Compile `program`, one part traced by `torch.export`, into the program file `path`.

        `program.example_inputs` holds the inputs it was traced with, on the backend's device. Returns a
        second file written beside `path` that the runtime loads with the program, such as its weights, or
        `None`.
        """

    def run(self, path: Path, data: Path | None, inputs: Sequence[Tensor]) -> tuple[Tensor, ...]:
        """Run the program file `path` once on PyTorch tensors, for a recipe that checks a part on its own.

        `data` is the file `compile` returned. A backend whose programs cannot run in the export's process does
        not override this, and cannot export such a recipe.
        """
        raise NotImplementedError(f"The {self.type} backend cannot run its programs at export.")


@dataclass
class ExportConfig:
    """Configuration of `lerobot-export`."""

    # The compiler, such as `executorch_tensorrt`.
    backend: ExportBackendConfig
    # The trained policy, loaded from `--policy.path`.
    policy: PreTrainedConfig | None = None
    # The bundle folder to write. It must not exist yet. Defaults to
    # `outputs/export/<policy type>_<backend type>`.
    output_dir: Path | None = None
    # The task, for a policy that reads one. The bundle's test case runs it, and so does the bundle by default;
    # a bundle without task tokens runs only this task.
    task: str = ""
    # The dataset the policy was trained on. The bundle then takes the robot's frame from the dataset's
    # observations, through the preprocessor's renames, and the dataset's first task when `task` is empty.
    dataset: DatasetConfig | None = None
    # The largest difference allowed between the compiled actions and PyTorch's on the test case, in the
    # policy's normalized action units (robot units divided by the spread of each action in the training
    # data). Defaults to the recipe's.
    tolerance: float | None = None
    # For a policy whose bundle takes any task, such as pi0.5, the most tokens a task may take in the bundle.
    # Defaults to the recipe's: a little more than the longest task known at export, as far as the policy's
    # prompt has room.
    max_task_tokens: int | None = None
    # Seeds the random test case.
    seed: int = 1000
    # Compute in fewer bits, with a mode the backend offers, such as `fp8` with `executorch_tensorrt`.
    # The bundle is still checked against the unquantized policy, so `tolerance` must then be given.
    quantize: str | None = None
    # INFO logs each step of the export and how long it took; DEBUG adds the logs of torch.export, ExecuTorch and
    # the backend's compiler.
    log_level: Literal["INFO", "DEBUG"] = "INFO"

    def __post_init__(self) -> None:
        """Check the PyTorch version, load the policy config from `--policy.path` and pick the bundle folder."""
        self.backend.check_torch()
        if self.quantize is not None and self.quantize not in self.backend.quantizations:
            offered = ", ".join(self.backend.quantizations) or "none"
            raise ValueError(
                f"The {self.backend.type} backend cannot --quantize={self.quantize}. It offers: {offered}."
            )
        if self.quantize is not None and self.tolerance is None:
            raise ValueError(
                "--quantize needs --tolerance: the bundle is checked against the unquantized policy, and "
                "quantization moves the actions by more than the default tolerance allows. Pass the largest "
                "difference you accept, in normalized action units."
            )
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            yaml_overrides = parser.get_yaml_overrides("policy")
            cli_overrides = parser.get_cli_overrides("policy") or []
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path,
                revision=parser.get_pretrained_revision("policy"),
                cli_overrides=yaml_overrides + cli_overrides,
            )
        if self.policy is None or self.policy.pretrained_path is None:
            raise ValueError("--policy.path is required: the trained policy to export.")
        if self.max_task_tokens is not None and self.max_task_tokens < 1:
            raise ValueError(f"--max_task_tokens must be at least 1, not {self.max_task_tokens}.")
        if self.tolerance is not None and not (math.isfinite(self.tolerance) and self.tolerance >= 0):
            raise ValueError(f"--tolerance must be a finite number of at least 0, not {self.tolerance}.")
        if self.output_dir is None:
            self.output_dir = Path("outputs/export") / f"{self.policy.type}_{self.backend.type}"
        # A link to nothing counts too: the bundle could not be renamed onto it.
        if os.path.lexists(self.output_dir):
            raise FileExistsError(f"{self.output_dir} already exists. Pick another --output_dir.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Let `--policy.path` load the policy config."""
        return ["policy"]
