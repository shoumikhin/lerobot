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
import importlib.metadata
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import draccus
from packaging.version import Version

from lerobot.bundle.manifest import Target
from lerobot.configs import PreTrainedConfig, parser
from lerobot.configs.default import DatasetConfig

if TYPE_CHECKING:
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
        runtime_modules (`tuple[str, ...]`):
            Python modules the files need to load, such as `executorch`. The bundle records their versions as
            its `runtime_versions`.
        packages (`tuple[str, ...]`):
            What to install beside LeRobot to export with this backend, such as `"executorch>=1.6.0,<1.7.0"`.
        min_torch (`str`, *optional*):
            The oldest PyTorch whose traced programs the compiler reads, such as `"2.15"`. `None` takes any.
        torch_overrides (`tuple[str, ...]`):
            The PyTorch pins that replace LeRobot's in an environment for exporting with this backend, such as
            `("torch>=2.15,<2.16", "torchvision>=0.29.2,<0.30")`.
    """

    runtime: ClassVar[str]
    file_suffix: ClassVar[str]
    device: ClassVar[str]
    runtime_imports: ClassVar[tuple[str, ...]] = ()
    runtime_modules: ClassVar[tuple[str, ...]] = ()
    packages: ClassVar[tuple[str, ...]] = ()
    min_torch: ClassVar[str | None] = None
    torch_overrides: ClassVar[tuple[str, ...]] = ()

    def target(self) -> Target | None:
        """The GPU the files are built on when `device` is `cuda`, since they then run only on that kind of GPU.

        `None` for files that run anywhere their runtime does.
        """
        if self.device != "cuda":
            return None
        import torch

        major, minor = torch.cuda.get_device_capability()
        return Target("cuda", torch.cuda.get_device_name(), f"{major}.{minor}")

    @property
    def type(self) -> str:
        """The `--backend.type` name of this backend."""
        return self.get_choice_name(self.__class__)

    def check_torch(self) -> None:
        """Refuse a PyTorch older than `min_torch` before anything is traced, and say how to export."""
        if self.min_torch is None:
            return
        installed = importlib.metadata.version("torch")
        if Version(installed).release >= Version(self.min_torch).release:
            return
        message = (
            f"The {self.type} backend needs PyTorch {self.min_torch} or newer to export, but PyTorch {installed} "
            "is installed."
        )
        if self.torch_overrides:
            overrides = "\\n".join(self.torch_overrides)
            packages = " ".join(f'"{package}"' for package in self.packages)
            message += (
                " LeRobot pins an older PyTorch, so export from a second environment that overrides the pin:\n\n"
                f"    printf '{overrides}\\n' > torch-overrides.txt\n"
                f"    uv pip install --override torch-overrides.txt lerobot {packages}\n\n"
                "The override replaces every PyTorch pin, so the packages after lerobot are pinned to that PyTorch "
                "by hand."
            )
        raise ValueError(f"{message} See https://huggingface.co/docs/lerobot/export.")

    @abc.abstractmethod
    def compile(self, program: ExportedProgram, path: Path) -> Path | None:
        """Compile `program`, one part traced by `torch.export`, into the program file `path`.

        `program.example_inputs` holds the inputs it was traced with, on the backend's device. Returns a
        second file written beside `path` that the runtime loads with the program, such as its weights, or
        `None`.
        """


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
    # The task, for a policy that reads one. Its bundle then runs only this task.
    task: str = ""
    # The dataset the policy was trained on. The bundle then takes the robot's frame from the dataset's
    # observations, through the renames the preprocessor saved, and the dataset's first task when `task` is
    # empty.
    dataset: DatasetConfig | None = None
    # The largest difference allowed between the compiled actions and PyTorch's on the test case, in the
    # policy's normalized action units (robot units divided by the spread of each action in the training
    # data). Defaults to the recipe's.
    tolerance: float | None = None
    # Seeds the random test case.
    seed: int = 1000

    def __post_init__(self) -> None:
        """Check the PyTorch version, load the policy config from `--policy.path` and pick the bundle folder."""
        self.backend.check_torch()
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            yaml_overrides = parser.get_yaml_overrides("policy")
            cli_overrides = parser.get_cli_overrides("policy") or []
            revision = parser.parse_arg("pretrained_revision", cli_overrides)
            if revision is None:
                revision = parser.parse_arg("pretrained_revision", yaml_overrides)
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, revision=revision, cli_overrides=yaml_overrides + cli_overrides
            )
        if self.policy is None or self.policy.pretrained_path is None:
            raise ValueError("--policy.path is required: the trained policy to export.")
        if self.tolerance is not None and not (math.isfinite(self.tolerance) and self.tolerance >= 0):
            raise ValueError(f"--tolerance must be a finite number of at least 0, not {self.tolerance}.")
        if self.output_dir is None:
            self.output_dir = Path("outputs/export") / f"{self.policy.type}_{self.backend.type}"
        if self.output_dir.exists():
            raise FileExistsError(f"{self.output_dir} already exists. Pick another --output_dir.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """Let `--policy.path` load the policy config."""
        return ["policy"]
