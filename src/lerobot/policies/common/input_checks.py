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

"""Input checks that a policy keeps when `torch.export` or `torch.compile` traces it."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.utils import torch_compilable_check
else:
    torch_compilable_check = None


def check_input(
    condition: Tensor | bool, message: str | Callable[[], str], error_type: type[Exception] = ValueError
) -> None:
    """Raise `error_type` with `message` unless `condition` holds (for every element of a tensor).

    In eager mode this is a plain check that always raises, as these input checks were before. While
    `torch.export` or `torch.compile` traces, it is transformers' `torch_compilable_check`, a runtime assert in
    the program (which `TRANSFORMERS_DISABLE_TORCH_CHECK=1` turns off; a failed assert loses its message).
    """
    if torch.compiler.is_compiling() or torch.compiler.is_exporting():
        torch_compilable_check(condition, message, error_type=error_type)
    elif not bool(torch.as_tensor(condition).all()):
        raise error_type(message() if callable(message) else message)
