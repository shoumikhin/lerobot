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
"""Checks shared by the fixed-grid Qwen-VL vision tests.

The fixed-grid forwards must call every vision block as a module, like the stock towers, so that FSDP2
hooks on a sharded block and gradient checkpointing still run.
"""

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest
import torch.distributed as dist
from torch import Tensor, nn

requires_gloo = pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()), reason="needs torch.distributed with gloo"
)


def count_block_calls_with_checkpointing(visual: nn.Module, forward: Callable[[], Tensor]) -> int:
    """Run `forward()` and its backward with gradient checkpointing on, and count the vision block calls."""
    calls = []
    handles = [block.register_forward_pre_hook(lambda *_: calls.append(None)) for block in visual.blocks]
    visual.train()
    visual.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    try:
        forward().square().sum().backward()
    finally:
        visual.gradient_checkpointing_disable()
        for handle in handles:
            handle.remove()
    return len(calls)


class _Root(nn.Module):
    def __init__(self, visual: nn.Module, forward: Callable[[nn.Module], Tensor]):
        super().__init__()
        self.visual = visual
        self.run = forward

    def forward(self) -> Tensor:
        return self.run(self.visual)


def run_with_sharded_blocks(
    tmp_path: Path, visual: nn.Module, forward: Callable[[nn.Module], Tensor]
) -> Tensor:
    """Run `forward(visual)` and its backward with FSDP2 on one CPU rank, every vision block its own unit.

    This is how LeRobot's FSDP training wraps these towers: the blocks are wrap units under a root module.
    """
    # Imported here because builds without torch.distributed cannot import them.
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    root = _Root(visual, forward)
    dist.init_process_group(
        "gloo",
        init_method=(tmp_path / "rendezvous").as_uri(),
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (1,))
        for block in visual.blocks:
            fully_shard(block, mesh=mesh)
        fully_shard(root, mesh=mesh)
        output = root()
        output.square().sum().backward()
        return output.detach()
    finally:
        dist.destroy_process_group()
