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
from safetensors.numpy import save_file
from torch import Tensor, nn

from lerobot.inference.bundle import MANIFEST, RUNTIMES, Bundle, BundleManifest, ProgramSpec, Runtime
from lerobot.inference.bundle.manifest import NOISE, TEST_CASE
from lerobot.utils.constants import ACTION


class TorchProgram:
    """Runs a saved `torch.export` program. A data file, when given, must exist."""

    def __init__(self, path: Path, data: Path | None = None):
        if data is not None and not data.is_file():
            raise FileNotFoundError(data)
        self._module = torch.export.load(path).module()

    def __call__(self, *inputs: Any) -> tuple[Tensor, ...]:
        outputs = self._module(*(torch.as_tensor(x) for x in inputs))
        return outputs if isinstance(outputs, tuple) else (outputs,)

    def to_numpy(self, value: Tensor) -> np.ndarray:
        return value.numpy()


RUNTIMES["torch_test"] = Runtime(TorchProgram)


def make_bundle(
    folder: Path,
    module: nn.Module,
    frame: dict[str, np.ndarray],
    *,
    noise_shape: tuple[int, ...] | None = None,
    task: str | None = None,
    action_names: list[str] | None = None,
    data: str | None = None,
    expected_offset: float = 0.0,
) -> Bundle:
    """Write a one-program bundle of `module`, whose test case expects its actions plus `expected_offset`.

    With `noise_shape`, the program takes a starting noise after the frame. With `data`, the program names
    a data file of that name, written beside it.
    """
    folder.mkdir(parents=True)
    case = dict(frame)
    if noise_shape is not None:
        case[NOISE] = np.random.default_rng(0).standard_normal(noise_shape).astype(np.float32)
    inputs = tuple(torch.from_numpy(x) for x in case.values())
    torch.export.save(torch.export.export(module, inputs), folder / "model.pt2")
    if data is not None:
        (folder / data).write_bytes(b"")
    with torch.no_grad():
        actions = module(*inputs).numpy() + expected_offset
    save_file({**case, ACTION: actions}, folder / TEST_CASE)
    manifest = BundleManifest(
        backend="torch_test",
        runtime="torch_test",
        runtime_imports=["tests.export.torch_backend"],
        programs=[ProgramSpec("model", "model.pt2", list(case), [ACTION], data)],
        noise_shape=None if noise_shape is None else list(noise_shape),
        task=task,
        action_names=action_names,
        action_scale=[1.0] * actions.shape[-1],
        tolerance=1e-4,
    )
    (folder / MANIFEST).write_text(json.dumps(manifest.to_dict()))
    return Bundle.read(folder)
