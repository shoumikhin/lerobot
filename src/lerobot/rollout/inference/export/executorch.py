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

"""Run an exported policy's ExecuTorch program, with its TensorRT engine inside."""

from pathlib import Path

import torch


class ExecuTorchProgram:
    """Runs a `.pte` program whose inputs and outputs stay on the GPU."""

    def __init__(self, path: Path):
        import torch_tensorrt_executorch_runtime  # noqa: F401  # registers the TensorRT backend
        from executorch.runtime import Runtime

        self._method = Runtime.get().load_program(path).load_method("forward")

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self._method.execute(list(inputs))[0]
