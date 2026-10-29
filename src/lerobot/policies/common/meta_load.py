#!/usr/bin/env python

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

"""Build a policy with its parameters on the meta device, and load chosen weights straight into them."""

import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import torch
from safetensors import safe_open
from torch import nn
from torch.nn.modules.module import register_module_parameter_registration_hook
from torch.utils.weak import WeakIdKeyDictionary

from lerobot.policies.pretrained import T

_meta_init = threading.local()
_hook_lock = threading.Lock()
_hook_registered = False


def _parameter_to_meta(module: nn.Module, name: str, param: nn.Parameter) -> nn.Parameter | None:
    replacements = getattr(_meta_init, "replacements", None)
    if replacements is None or param.is_meta:
        return None
    if param not in replacements:
        replacements[param] = nn.Parameter(param.to("meta"), requires_grad=param.requires_grad)
    return replacements[param]


def _register_hook_once() -> None:
    # Registered on first use rather than on import, since every policy import reaches this module, and
    # never removed, so that later blocks do not change PyTorch's process-wide hook registry while another
    # thread iterates it to register a parameter.
    global _hook_registered
    with _hook_lock:
        if not _hook_registered:
            register_module_parameter_registration_hook(_parameter_to_meta)
            _hook_registered = True


@contextmanager
def parameters_on_meta() -> Iterator[None]:
    """Create the parameters of modules built inside this block on the meta device.

    A meta tensor has a shape and a dtype but no memory. Modules like `nn.Linear` allocate a parameter and
    register it before they initialize it, so building them here frees that memory at once and skips
    randomly initializing weights that a checkpoint will replace; `nn.Parameter(torch.randn(n))` is still
    drawn before it moves. Buffers and plain tensors are still computed, so values built from the config
    (rotary tables, masks) stay correct, but one computed from a parameter ends up on meta. Only the thread
    that entered the block is affected, and a parameter registered in two modules stays shared.
    """
    _register_hook_once()
    previous = getattr(_meta_init, "replacements", None)
    _meta_init.replacements = WeakIdKeyDictionary() if previous is None else previous
    try:
        yield
    finally:
        _meta_init.replacements = previous


def load_modules(
    policy: T,
    model_file: str,
    modules: list[str],
    sources: Mapping[str, tuple[str, torch.dtype | None]],
    device: str | None,
) -> T:
    """Fill the `modules` of `policy`, built under `parameters_on_meta`, from `model_file`, and drop the rest.

    `sources` maps each name in the policy's state dict to its key in the file and the dtype to load it in,
    `None` keeping the file's. Every other submodule holding parameters is replaced by None, as the policy
    already does for modules it never runs, so code that reaches one fails instead of reading random weights,
    and an export of one part holds only its own weights. Raises if a listed submodule has a parameter the file
    does not hold. Returns the policy in evaluation mode, as from `from_pretrained`, since a policy's own
    `train` may reach a module that is now None, and on `device`.
    """
    with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
        # A CPU copy first, because a device copy straight from the memory-mapped file can keep its pages.
        tensors = {
            name: checkpoint.get_tensor(key).to(dtype=dtype, copy=True).to(device)
            for name, (key, dtype) in sources.items()
        }
    policy.load_state_dict(tensors, strict=False, assign=True)
    for name in modules:
        missing = [p for p, param in policy.get_submodule(name).named_parameters() if param.is_meta]
        if missing:
            raise KeyError(f"{model_file} does not hold {name}.{missing[0]} and {len(missing) - 1} more")
    policy.eval()
    _drop_unloaded(policy)
    # Buffers the constructor computed, like rotary tables, are not in the checkpoint and are still on the CPU.
    return policy.to(device)


def _drop_unloaded(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        parameters = list(child.parameters())
        if parameters and all(parameter.is_meta for parameter in parameters):
            setattr(module, name, None)
        else:
            _drop_unloaded(child)
