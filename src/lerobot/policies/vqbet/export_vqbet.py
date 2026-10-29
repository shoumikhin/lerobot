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

"""Export VQ-BeT as one program that reads its last `n_obs_steps` frames and samples its codes from the noise."""

from __future__ import annotations

from collections.abc import Iterator

from torch import Tensor, nn

from lerobot.export.recipe import Part, ReplayDraws
from lerobot.inference.bundle.manifest import NOISE
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION

from ..common.export_history import HistoryBatch, HistoryChunkRecipe
from .modeling_vqbet import VQBeTModel


class VQBeTExportRecipe(HistoryChunkRecipe):
    """VQ-BeT's rollout as one program: the last `n_obs_steps` frames and the noise in, the action chunk out.

    VQ-BeT samples a code for each action token. The program takes those draws from its noise input, so the
    bundle draws them at run time and the test case can replay them in PyTorch.
    """

    def _execution_steps(self) -> int:
        return self.policy.config.action_chunk_size

    def parts(self) -> Iterator[Part]:
        """The one part, after checking that it computes the policy's chunk on two sets of frames."""
        module = _VQBeTChunk(self._history_batch(), self.policy.vqbet, self.postprocessor, self._steps)
        self._check_chain(module)
        names = list(self._frame_shapes)
        inputs = self._history_inputs(self._case.frame, self._case.noise)
        yield Part("model", module, inputs, [*names, NOISE], [ACTION])


class _VQBeTChunk(nn.Module):
    def __init__(
        self, batch: HistoryBatch, model: VQBeTModel, postprocessor: PolicyProcessorPipeline, steps: int
    ):
        super().__init__()
        self.batch = batch
        self.model = model
        self.postprocessor = postprocessor
        self.steps = steps

    def forward(self, *inputs: Tensor) -> Tensor:
        *frames, noise = inputs
        with ReplayDraws(noise):
            actions = self.model(self.batch(*frames), rollout=True)[:, : self.steps]
        return self.postprocessor(actions)[0].float()
