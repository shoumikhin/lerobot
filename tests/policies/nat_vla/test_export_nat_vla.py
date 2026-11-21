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

"""NAT VLA's exported program: what the shared program tests in tests/export do not check."""

import pytest
import torch
from torch.utils._pytree import tree_leaves

pytest.importorskip("transformers")

from tests.policies.nat_vla.export_stand_in import make_recipe  # noqa: E402


def test_every_size_in_the_program_is_known_when_it_is_traced(tmp_path):
    """TensorRT needs an upper bound on every size, and a size the program reads from a tensor, such as each
    image's patch grid in `spatial_shapes`, has none."""
    [part] = make_recipe(tmp_path).parts()
    program = torch.export.export(part.module, part.example_inputs)
    tensors = [value for node in program.graph.nodes for value in tree_leaves(node.meta.get("val"))]
    shapes = [x.shape for x in tensors if isinstance(x, torch.Tensor)]
    assert not [size for shape in shapes for size in shape if not isinstance(size, int)]
