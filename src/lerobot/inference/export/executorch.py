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

"""Run an exported policy's ExecuTorch programs on TensorRT or on ExecuTorch's CUDA backend, without PyTorch."""

import os
from pathlib import Path
from typing import Any

import numpy as np

# ExecuTorch returns its own results rather than torch tensors, even where torch is installed, so running
# a program never imports torch. The bindings read this once, when they first load, and the code below
# only reads their own results, so a shell setting of "torch" is overridden.
os.environ["EXECUTORCH_PYBINDINGS_TENSOR_OUTPUT"] = "executorch"

# DLPack's device type for CUDA memory.
DLPACK_CUDA = 2


class ExecuTorchProgram:
    """Run a `.pte` program whose inputs and outputs live on the GPU.

    `run_device` keeps its outputs on the GPU, so the next program in a chain reads them where they are.
    Calling the last program of a chain takes host arrays and returns its one output, the actions, on the host.
    """

    def __init__(self, path: Path, route: str):
        """Load the `forward` method of the program at `path`, and its data file if it has one."""
        cuda = route == "executorch_cuda"
        needs = (
            "a CUDA build of ExecuTorch 1.6.0.dev20261021 or newer"
            if cuda
            else 'ExecuTorch 1.6.0.dev20261021 or newer and its TensorRT runtime, which `pip install "torch-tensorrt[executorch]"` installs'
        )
        try:
            if not cuda:
                import torch_tensorrt_executorch_runtime  # noqa: F401  # registers the TensorRT backend

            # ExecuTorchResult arrived with the API this module uses, so an older ExecuTorch fails here.
            from executorch.runtime import ExecuTorchResult, Runtime  # noqa: F401
        except ImportError as error:
            raise ImportError(
                f"This export needs {needs}; see examples/export/README.md. ({error})"
            ) from error

        runtime = Runtime.get()
        if cuda:
            if not runtime.backend_registry.is_available("CudaBackend"):
                raise ImportError(f"This export needs {needs}; see examples/export/README.md.")
            # Replay each program's work as a CUDA graph, as the TensorRT routes do.
            runtime.backend_registry.set_option("CudaBackend", {"enable_cuda_graph_for_method": "forward"})
        else:
            # The programs of a chain run one after another, so they can share one activation scratch.
            runtime.backend_registry.set_option("TensorRTBackend", {"use_shared_activation_scratch": True})
        data = path.with_suffix(".ptd")
        program = runtime.load_program(path, data_path=data if data.is_file() else None)
        self._method = program.load_method("forward")

    def run_device(self, *inputs: np.ndarray | Any) -> tuple[Any, ...]:
        """Run the program; its outputs stay on the GPU for the next program to read."""
        return tuple(self._method.execute([on_device(value) for value in inputs]))

    def __call__(self, *inputs: np.ndarray | Any) -> np.ndarray:
        """Run the program and return its one output, the actions, as a host array."""
        (output,) = self.run_device(*inputs)
        return np.asarray(output.to_host())

    def upload(self, value: np.ndarray) -> Any:
        """`value` in GPU memory, for many runs to read without another copy."""
        return on_device(value)


def on_device(value: np.ndarray | Any) -> Any:
    """`value` in GPU memory: a result already there passes straight through, anything else is copied.

    The bindings hand a CUDA program whatever pointer an input has, so a host input would be read as
    GPU memory.
    """
    from executorch.runtime import ExecuTorchResult

    if not isinstance(value, ExecuTorchResult):
        return ExecuTorchResult.from_host(np.ascontiguousarray(value))
    return value if value.__dlpack_device__()[0] == DLPACK_CUDA else ExecuTorchResult.from_host(value)
