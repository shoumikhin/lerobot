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

"""pi0-FAST's export recipe on a tiny pi0-FAST: its state prompt, its decoding, and what it refuses."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("scipy")

from lerobot.bundle.action_tokens import ActionDecoder  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.common.state_prompt import StatePromptTokens  # noqa: E402
from lerobot.policies.pi0_fast import PI0FastPolicy  # noqa: E402
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig  # noqa: E402
from lerobot.policies.pi0_fast.export_pi0_fast import PI0FastExportRecipe, PI0FastPrefill  # noqa: E402
from lerobot.policies.pi0_fast.processor_pi0_fast import (  # noqa: E402
    Pi0FastPrepareStateAndLanguageTokenizerProcessorStep,
)
from lerobot.processor import TransitionKey  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402
from tests.export.tiny_pi0_fast import VOCABULARY, save_checkpoint, use_tiny_pi0_fast  # noqa: E402

TASK = "pick up the cube"


def test_state_prompt_step_leaves_the_transition_it_is_given_unchanged():
    step = Pi0FastPrepareStateAndLanguageTokenizerProcessorStep()
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.tensor([[0.0, 1.0]])},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick up the cube"]},
    }
    prompts = [step(transition)[TransitionKey.COMPLEMENTARY_DATA]["task"] for _ in range(2)]
    assert prompts == [["Task: pick up the cube, State: 128 255;\n"]] * 2
    assert transition[TransitionKey.COMPLEMENTARY_DATA] == {"task": ["pick up the cube"]}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A tiny pi0-FAST checkpoint and its config."""
    folder = tmp_path_factory.mktemp("pi0_fast")
    with pytest.MonkeyPatch.context() as patch:
        return folder / "checkpoint", save_checkpoint(folder / "checkpoint", patch)


def load_policy(checkpoint, config, monkeypatch):
    """The tiny pi0-FAST and its processors, as lerobot-rollout loads them."""
    use_tiny_pi0_fast(monkeypatch)
    policy = PI0FastPolicy.from_pretrained(checkpoint, config=config).eval()
    return policy, *make_pre_post_processors(config, pretrained_path=str(checkpoint))


def make_recipe(checkpoint, config, monkeypatch) -> PI0FastExportRecipe:
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task=TASK, tolerance=None
    )
    return PI0FastExportRecipe(*load_policy(checkpoint, config, monkeypatch), cfg)


def answers(spec, fast_vocabulary, count):
    """Answers of random FAST tokens. Some hold `Action: ` again, the end marker, an id of no FAST token, or an
    unused id from the end of the vocabulary, whose FAST id is negative."""
    generator = torch.Generator().manual_seed(0)
    for i in range(count):
        tokens = spec.id_offset - torch.randint(fast_vocabulary, (12,), generator=generator)
        extra = [spec.prefix, [spec.end], [spec.id_offset - fast_vocabulary - 3], [spec.id_offset + 2], []][
            i % 5
        ]
        at = int(torch.randint(len(tokens), (1,), generator=generator))
        yield torch.cat(
            (torch.tensor(spec.prefix), tokens[:at], torch.tensor(extra, dtype=torch.long), tokens[at:])
        )


def test_the_decoding_of_answers_matches_the_policy(checkpoint, monkeypatch):
    recipe = make_recipe(*checkpoint, monkeypatch)
    next(recipe.parts())
    # What the bundle holds: the FAST tokenizer and how the policy reads its answer.
    decoding = recipe.plan().action_tokens
    spec = decoding.spec
    fast = recipe.policy.action_tokenizer.bpe_tokenizer
    decoder = ActionDecoder(json.loads(decoding.tokenizer), spec)
    for answer in answers(spec, len(fast), 100):
        expected = recipe.policy.detokenize_actions(answer, spec.horizon, spec.action_dim).numpy()
        np.testing.assert_allclose(decoder(answer.numpy()), expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "change, error",
    [
        ({"temperature": 0.7}, "set --policy.temperature=0"),
        ({"use_kv_cache": False}, "use_kv_cache=true"),
        ({"dtype": torch.bfloat16}, "set --policy.dtype=float32"),
    ],
)
def test_recipe_refuses_decoding_it_does_not_export(change, error):
    with pytest.raises(ValueError, match=error):
        PI0FastExportRecipe(SimpleNamespace(config=PI0FastConfig(**change)), None, None, None)


def test_recipe_refuses_a_chain_that_generates_other_tokens(checkpoint, monkeypatch):
    recipe = make_recipe(*checkpoint, monkeypatch)
    prefill = PI0FastPrefill.forward

    def first_token_off_by_one(self, *frame):
        tokens, *rest = prefill(self, *frame)
        return (tokens + 1) % VOCABULARY, *rest

    monkeypatch.setattr(PI0FastPrefill, "forward", first_token_off_by_one)
    with pytest.raises(
        ValueError, match="chain generates token .* where the policy generates .*, at position 0"
    ):
        next(recipe.parts())


def test_recipe_refuses_a_decoding_that_differs_from_the_policy(checkpoint, monkeypatch):
    recipe = make_recipe(*checkpoint, monkeypatch)
    decode = ActionDecoder.__call__
    monkeypatch.setattr(ActionDecoder, "__call__", lambda self, tokens: decode(self, tokens) + 1)
    with pytest.raises(ValueError, match="decoding of pi0-FAST's tokens differs from the policy's"):
        next(recipe.parts())


def test_recipe_refuses_a_prompt_with_other_tokens_than_the_processor(checkpoint, monkeypatch):
    recipe = make_recipe(*checkpoint, monkeypatch)
    build = StatePromptTokens.__call__

    def one_token_off(self, bins):
        ids, mask = build(self, bins)
        return (ids + 1) % VOCABULARY, mask

    monkeypatch.setattr(StatePromptTokens, "__call__", one_token_off)
    with pytest.raises(ValueError, match="writes another prompt than pi0-FAST's state prompt"):
        next(recipe.parts())
