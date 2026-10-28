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

"""A test runtime for bundles, so tests need no compiler: programs are saved `torch.export` programs, run in
PyTorch. A bundle names this module in `runtime_imports`, which registers the runtime, as a compiler's own
module registers its kernels with a real runtime.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub.constants import CONFIG_NAME
from safetensors.numpy import save_file
from torch import Tensor, nn

from lerobot.inference.export import MANIFEST, RUNTIMES, ExportBundle, ExportManifest, ProgramSpec
from lerobot.inference.export.bundle import TEST_CASE
from lerobot.utils.constants import ACTION


class TorchProgram:
    """Runs a saved `torch.export` program."""

    def __init__(self, path: Path):
        self._module = torch.export.load(path).module()

    def __call__(self, *inputs: Any) -> tuple[Tensor, ...]:
        outputs = self._module(*(torch.as_tensor(x) for x in inputs))
        return outputs if isinstance(outputs, tuple) else (outputs,)

    def to_numpy(self, value: Tensor) -> np.ndarray:
        return value.numpy()


RUNTIMES["torch_test"] = TorchProgram


def make_bundle(
    folder: Path,
    module: nn.Module,
    frame: dict[str, np.ndarray],
    task: str | None = None,
    expected_offset: float = 0.0,
) -> ExportBundle:
    """Write a one-program bundle of `module`, whose test case expects its actions plus `expected_offset`."""
    folder.mkdir(parents=True)
    inputs = tuple(torch.from_numpy(x) for x in frame.values())
    torch.export.save(torch.export.export(module, inputs), folder / "model.pt2")
    with torch.no_grad():
        actions = module(*inputs).numpy() + expected_offset
    save_file({**frame, ACTION: actions}, folder / TEST_CASE)
    (folder / CONFIG_NAME).write_text(json.dumps({"type": "act"}))
    manifest = ExportManifest(
        backend="torch_test",
        runtime="torch_test",
        runtime_imports=["tests.export.torch_backend"],
        programs=[ProgramSpec("model", "model.pt2", list(frame), [ACTION])],
        task=task,
        tolerance=1e-4,
    )
    (folder / MANIFEST).write_text(json.dumps(manifest.to_dict()))
    return ExportBundle.read(folder)
