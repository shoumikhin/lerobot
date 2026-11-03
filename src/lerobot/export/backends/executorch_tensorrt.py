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

"""ExecuTorch programs that run on NVIDIA GPUs through TensorRT."""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..configs import ExportBackendConfig
from .executorch import EXPORT_EXTRA, ExecuTorchConfig, cuda_partitioner, gpu_memory_config

if TYPE_CHECKING:
    from torch import Tensor, nn
    from torch.export import ExportedProgram

# Model Optimizer's FP8 config already leaves the vision towers of vision-language models unquantized, by their usual
# names. GR00T's export names its encoder `vision`. Model Optimizer 0.48 also quantizes the inputs of LayerNorm
# layers, which its next release stops doing.
FP8_KEPT = (
    {"quantizer_name": "*vision*", "enable": False},
    {"parent_class": "nn.LayerNorm", "quantizer_name": "*", "enable": False},
)

logger = logging.getLogger(__name__)

MODELOPT_SKIP_CHECK = "MODELOPT_SKIP_WEIGHT_QUANT_CHECK"


@ExportBackendConfig.register_subclass("executorch_tensorrt")
@dataclass
class ExecuTorchTensorRTConfig(ExecuTorchConfig):
    """ExecuTorch programs whose work runs on the GPU through TensorRT, NVIDIA's compiler.

    An operator TensorRT does not take runs on the GPU too, compiled by ExecuTorch's CUDA backend, which needs
    the CUDA compiler (`nvcc`) at export. A TensorRT engine runs only on the GPU model that built it, so export
    on the kind of board that will run the policy.

    `--quantize=fp8` stores weights and activations in 8-bit floats, with scales NVIDIA Model Optimizer measures on
    each part's own inputs at export. It needs a GPU of compute capability 8.9 or newer (Ada, Hopper, Blackwell,
    Jetson Thor) and `pip install nvidia-modelopt`.
    """

    device = "cuda"
    # Registers the TensorRT backend with the ExecuTorch runtime.
    runtime_imports = ("torch_tensorrt_executorch_runtime",)
    # A bundle runs its TensorRT engines one after another, so they can share one scratch buffer for
    # activations instead of holding one each.
    runtime_options = {"TensorRTBackend": {"use_shared_activation_scratch": True}}
    runtime_modules = ("executorch", "tensorrt", "torch_tensorrt_executorch_runtime")
    # TensorRT loads an engine only with the TensorRT version that built it, unless built version compatible.
    exact_runtime_modules = ("tensorrt",)
    packages = (EXPORT_EXTRA, "torch_tensorrt[executorch]")
    quantizations = ("fp8",)
    # Registers `tensorrt::quantize_op`, which a part traced with --quantize=fp8 holds.
    quantization_imports = {"fp8": "modelopt.torch.quantization"}

    workspace_size: int | None = None
    """The most GPU memory, in bytes, TensorRT may use for scratch while it builds an engine, for example to fit
    a build into a small board. `None` lets TensorRT decide."""
    deduplicate_weights: bool = True
    """Keep one host copy of each weight, and of its transpose, instead of one per layer that reads it
    (pytorch/TensorRT#4965). `False` gives each layer its own copy, for a policy whose engines come out wrong
    with shared weights."""

    def quantized(
        self, module: nn.Module, example_inputs: tuple[Tensor, ...], mode: str
    ) -> contextlib.AbstractContextManager:
        """Quantize `module` with Model Optimizer, calibrated on `example_inputs`, and return its export context."""
        import torch

        if torch.cuda.get_device_capability() < (8, 9):
            raise ValueError(
                f"--quantize={mode} needs a GPU of compute capability 8.9 or newer, not "
                f"{torch.cuda.get_device_name()}."
            )
        try:
            import modelopt.torch.quantization as mtq
            from modelopt.torch.quantization.utils import export_torch_mode
        except ImportError as error:
            raise ImportError(
                f"--quantize={mode} could not import NVIDIA Model Optimizer ({error}). If it is not "
                "installed: pip install 'nvidia-modelopt>=0.48'"
            ) from error
        config = {**mtq.FP8_DEFAULT_CFG, "quant_cfg": [*mtq.FP8_DEFAULT_CFG["quant_cfg"], *FP8_KEPT]}
        # Calibration leaves every layer in its outermost module's mode, and a part's wrapper may be in training mode.
        module.eval()
        with _parts_may_hold_nothing_to_quantize():
            mtq.quantize(module, config, forward_loop=lambda part: part(*example_inputs))
        # A quantizer that saw only zeros has a scale of 0, and TensorRT divides by it. ACT's latent input is such.
        for name, quantizer in module.named_modules():
            if not name.endswith("_quantizer") or not quantizer.is_enabled or quantizer.amax is None:
                continue
            if (quantizer.amax == 0).any():
                quantizer.disable()
        quantized = [n for n, m in module.named_modules() if n.endswith("weight_quantizer") and m.is_enabled]
        logger.info("Quantized the weights of %d layers in %s", len(quantized), mode)
        return export_torch_mode()

    def compile(self, program: ExportedProgram, path: Path) -> Path | None:
        """Compile `program` with Torch-TensorRT and save it as an ExecuTorch program.

        Returns the weights file of the CUDA backend's part, if it has one.
        """
        import torch_tensorrt

        # Before TensorRT builds, so ExecuTorch's CUDA backend, which needs Triton, fails at once without it.
        partitioner = cuda_partitioner(path.stem)
        options = {
            "min_block_size": 1,
            # ExecuTorch programs run without libtorch, so a TensorRT plugin that needs it could never run.
            "fallback_pytorch_plugin_ops": True,
            # Flow-matching time embeddings are computed in float64, which TensorRT does not support.
            "truncate_double": True,
            # The part's PyTorch weights wait in host memory while TensorRT builds; on an 8 GB Jetson this
            # leaves about 600 MiB more free at the peak of the largest build.
            "offload_module_to_cpu": True,
            # Float32 math stays float32, as in PyTorch. TensorRT's default TF32 moved ACT's actions by up to
            # 5e-3 of their spread, a thousand times the float32 rounding.
            "disable_tf32": True,
            "deduplicate_weights": self.deduplicate_weights,
        }
        if self.workspace_size is not None:
            options["workspace_size"] = self.workspace_size
        args, _ = program.example_inputs
        engine = torch_tensorrt.dynamo.compile(program, arg_inputs=args, **options)
        torch_tensorrt.save(
            engine,
            str(path),
            output_format="executorch",
            # Tracing the compiled module again would drop the engines it holds.
            retrace=False,
            use_cuda_graphs=True,
            # Takes what TensorRT left, so no operator falls back to a CPU kernel between GPU parts.
            partitioners=[partitioner],
            backend_config=gpu_memory_config(),
        )
        data = path.with_suffix(".ptd")
        return data if data.exists() else None


@contextlib.contextmanager
def _parts_may_hold_nothing_to_quantize() -> Iterator[None]:
    """Let Model Optimizer quantize a part in which its config leaves no weight to quantize.

    Model Optimizer 0.48 refuses such a model, to catch a config that matches nothing. A part of a policy is often one,
    such as pi0.5's vision part, whose towers the config keeps, or a part with no linear layer. Its documented switch
    for a pipeline stage without the targeted layers is an environment variable, set here only around the call.
    """
    previous = os.environ.get(MODELOPT_SKIP_CHECK)
    os.environ[MODELOPT_SKIP_CHECK] = "1"
    try:
        yield
    finally:
        if previous is None:
            del os.environ[MODELOPT_SKIP_CHECK]
        else:
            os.environ[MODELOPT_SKIP_CHECK] = previous
