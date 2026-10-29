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
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import draccus

from lerobot.configs import PreTrainedConfig, parser
from lerobot.configs.default import DatasetConfig

if TYPE_CHECKING:
    from torch.export import ExportedProgram


@dataclass
class ExportBackendConfig(draccus.ChoiceRegistry, abc.ABC):
    """A compiler that turns each part of a policy into one program file. Chosen with `--backend.type`.

    A backend knows its compiler, never a policy. Backends whose files the same runtime loads share that
    runtime and `lerobot-rollout` unchanged.

    **Attributes**:
        - **runtime** (`str`) -- The runtime that loads the program files, such as `executorch`.
        - **file_suffix** (`str`) -- The suffix of the program files, such as `.pte`.
        - **device** (`str`) -- Where the policy runs while it is compiled, such as `cuda` or `cpu`.
        - **runtime_imports** (`tuple[str, ...]`) -- Python modules the runtime imports before it loads
          the files, such as one that registers the compiler's kernels with the runtime.
    """

    runtime: ClassVar[str]
    file_suffix: ClassVar[str]
    device: ClassVar[str]
    runtime_imports: ClassVar[tuple[str, ...]] = ()

    @property
    def type(self) -> str:
        """The `--backend.type` name of this backend."""
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def compile(self, program: ExportedProgram, path: Path) -> Path | None:
        """Compile `program`, one part traced by `torch.export`, into the program file `path`.

        `program.example_inputs` holds the inputs it was traced with, on the backend's device. Returns a
        second file written beside `path` that the runtime loads with the program, such as its weights, or
        `None`.
        """


@dataclass
class ExportConfig:
    """Configuration of `lerobot-export`.

    Args:
        backend (`ExportBackendConfig`):
            The compiler, chosen with `--backend.type`, for example `executorch_tensorrt`.
        policy (`PreTrainedConfig`, *optional*):
            The trained policy, loaded from `--policy.path`.
        output_dir (`Path`, *optional*):
            The bundle folder to write. It must not exist yet. Defaults to
            `outputs/export/<policy type>_<backend type>`.
        task (`str`, *optional*, defaults to `""`):
            The task, for a policy that reads one. Its bundle then runs only this task.
        dataset (`DatasetConfig`, *optional*):
            The dataset the policy was trained on, for a checkpoint that names placeholder cameras: the
            bundle then takes the dataset's camera names, and its first task when `task` is empty.
        tolerance (`float`, *optional*):
            The largest difference allowed between the compiled actions and PyTorch's on the test case, in
            the policy's normalized action units (robot units divided by the spread of each action in the
            training data). Defaults to the recipe's.
        max_task_tokens (`int`, *optional*):
            For a policy that reads its task, the most tokens a task may take in the bundle. Defaults to the
            recipe's, which fits the tasks known at export.
        seed (`int`, *optional*, defaults to `1000`):
            Seeds the random test case.
    """

    backend: ExportBackendConfig
    policy: PreTrainedConfig | None = None
    output_dir: Path | None = None
    task: str = ""
    dataset: DatasetConfig | None = None
    tolerance: float | None = None
    max_task_tokens: int | None = None
    seed: int = 1000

    def __post_init__(self) -> None:
        """Load the policy config from `--policy.path` and pick the bundle folder."""
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
        if self.max_task_tokens is not None and self.max_task_tokens < 1:
            raise ValueError(f"--max_task_tokens must be at least 1, not {self.max_task_tokens}.")
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
