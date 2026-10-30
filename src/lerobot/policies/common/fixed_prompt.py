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

"""A VLM tokenizer step a program can run: its prompt made once, its per-frame tensors made on each frame.

The transformers processors of the VLM policies count each image's tokens from its patch grid in Python and
tokenize text, so a program cannot run them. For one task and one camera size, the token ids, the masks and
the patch grid are the same on every frame; only the tensors read from the frame (the pixel values, and the
state for some policies) change. `FixedPromptStep` wraps the saved step: on eager frames it runs that step,
keeps what it adds as the prompt, and checks that the prompt does not change and that the policy's own
per-frame function gives the same tensors as the step. Under export it adds the kept prompt and only runs the
per-frame function, which the program can trace.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from lerobot.processor import EnvTransition, ProcessorStep, TransitionKey

from .qwen_vision import is_exporting


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b.to(a.device))
    return a is None and b is None


class FixedPromptStep(ProcessorStep):
    """`step` for one task and camera size: the prompt of the first eager frame, the tensors of each frame.

    `frame_inputs` computes, from a transition, the complementary-data entries that change from frame to
    frame. Every other tensor (or None) entry `step` adds to the complementary data is the prompt.
    """

    def __init__(self, step: ProcessorStep, frame_inputs: Callable[[EnvTransition], dict[str, Any]]) -> None:
        self.step = step
        self.frame_inputs = frame_inputs
        self.prompt: dict[str, Any] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        frame = self.frame_inputs(transition)
        if not is_exporting():
            self._check(transition, frame)
        if self.prompt is None:
            raise ValueError("The prompt is made from an eager call before the program is traced.")
        comp: dict[str, Any] = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        comp.update(self.prompt)
        comp.update(frame)
        return {**transition, TransitionKey.COMPLEMENTARY_DATA: comp}

    def _check(self, transition: EnvTransition, frame: dict[str, Any]) -> None:
        given = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        saved = self.step(transition)[TransitionKey.COMPLEMENTARY_DATA]
        prompt = {
            key: value
            for key, value in saved.items()
            if key not in frame and key not in given and (value is None or isinstance(value, torch.Tensor))
        }
        if self.prompt is None:
            self.prompt = prompt
        elif prompt.keys() != self.prompt.keys() or not all(
            _same(value, self.prompt[key]) for key, value in prompt.items()
        ):
            raise ValueError(
                f"{type(self.step).__name__}'s prompt changed between two frames of the same task and camera "
                "size, so the program cannot keep it."
            )
        for key, value in frame.items():
            if not _same(value, saved.get(key)):
                raise ValueError(
                    f"The per-frame {key!r} differs from what {type(self.step).__name__} computes."
                )

    def transform_features(self, features):
        return features
