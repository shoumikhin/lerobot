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

In eager mode the fixed-grid helpers must call the stock tower module, so FSDP2 hooks on a sharded tower or
block and gradient checkpointing still run. Under export their programs must not depend on tensor values
for any shape.
"""

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

from lerobot.policies.common.qwen_vision import FixedImageGrid

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


def run_with_fsdp(
    tmp_path: Path, visual: nn.Module, forward: Callable[[nn.Module], Tensor], *, whole_tower: bool
) -> Tensor:
    """Run `forward(visual)` and its backward with FSDP2 on one CPU rank, and return the output.

    The tower sits under a root module, as in LeRobot's FSDP training. Every vision block is its own wrap unit
    (the policies' default), or with `whole_tower` the tower is one unit (an `fsdp.wrap_modules` override).
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
        for unit in [visual] if whole_tower else visual.blocks:
            fully_shard(unit, mesh=mesh)
        fully_shard(root, mesh=mesh)
        output = root()
        output.square().sum().backward()
        return output.detach()
    finally:
        dist.destroy_process_group()


def count_grid_builds(monkeypatch: pytest.MonkeyPatch) -> list[type]:
    """Record every `FixedImageGrid.build`, so a test can check that later calls reuse the cache."""
    builds: list[type] = []
    build = FixedImageGrid.build.__func__

    def counted(cls, visual, grid_thw):
        builds.append(cls)
        return build(cls, visual, grid_thw)

    monkeypatch.setattr(FixedImageGrid, "build", classmethod(counted))
    return builds


def data_dependent_nodes(program: torch.export.ExportedProgram) -> list[str]:
    """Nodes whose output shape depends on tensor values: `nonzero`, boolean-mask indexing, unbacked sizes.

    Runtime asserts (`torch_compilable_check`) read values too, but only to raise, so they are not listed.
    """
    found = []
    for node in program.graph.nodes:
        if node.op != "call_function":
            continue
        name = str(node.target)
        indices = node.args[1] if name.startswith(("aten.index.", "aten.index_put")) else []
        values = [getattr(index, "meta", {}).get("val") for index in indices or []]
        if any(isinstance(v, torch.Tensor) and v.dtype == torch.bool for v in values):
            found.append(f"{name} with a boolean mask")
        elif "nonzero" in name or "masked_select" in name:
            found.append(name)
        else:
            value = node.meta.get("val")
            for tensor in value if isinstance(value, (list, tuple)) else [value]:
                if isinstance(tensor, torch.Tensor) and free_unbacked_symbols(tensor.shape):
                    found.append(f"{name} with a data-dependent shape")
    return found
