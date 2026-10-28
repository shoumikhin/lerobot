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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")

from lerobot.policies.pi0_fast import modeling_pi0_fast  # noqa: E402
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig  # noqa: E402
from lerobot.policies.pi_gemma import PaliGemmaForConditionalGenerationWithPiGemma  # noqa: E402

END_OF_ACTION = 3


class TinyPaliGemma(modeling_pi0_fast.PI0FastPaliGemma):
    """The PI0Fast wrapper around a tiny PaliGemma, so decoding runs without the full-size model."""

    def __init__(self, vlm_config, use_adarms=None, precision=torch.float32):
        nn.Module.__init__(self)
        config = transformers.PaliGemmaConfig(
            text_config=transformers.GemmaConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                vocab_size=64,
                hidden_activation="gelu_pytorch_tanh",
            ).to_dict(),
            vision_config=transformers.SiglipVisionConfig(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                image_size=8,
                patch_size=4,
            ).to_dict(),
            image_token_index=63,
            projection_dim=16,
            vocab_size=64,
        )
        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config)
        # Every step predicts the end-of-action marker, so greedy decoding stops after one token.
        self.paligemma.lm_head = nn.Linear(16, 64)
        with torch.no_grad():
            self.paligemma.lm_head.weight.zero_()
            self.paligemma.lm_head.bias.zero_()
            self.paligemma.lm_head.bias[END_OF_ACTION] = 100


class Tokenizer:
    bos_token_id = 1
    vocab_size = 64

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return END_OF_ACTION if tokens == "|" else int(tokens)
        return [self.convert_tokens_to_ids(token) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return ["|" if i == END_OF_ACTION else str(i) for i in ids]

    def encode(self, text, add_special_tokens=True):  # the "Action: " prefix, absent from the rows below
        return [60, 61]


def test_greedy_decoding_exports_and_keeps_its_early_stop_under_torch_compile(monkeypatch):
    monkeypatch.setattr(modeling_pi0_fast, "PI0FastPaliGemma", TinyPaliGemma)
    core = modeling_pi0_fast.PI0FastPytorch(
        PI0FastConfig(dtype=torch.float32), paligemma_tokenizer=Tokenizer()
    )

    class Decode(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = core

        def forward(self, image, image_mask, tokens, masks):
            return self.core.sample_actions_fast_kv_cache(
                [image], [image_mask], tokens, masks, max_decoding_steps=4, temperature=0
            )

    model = Decode().eval()
    tokens = torch.tensor([[4, 5], [6, 7]])
    image, image_mask = torch.randn(2, 3, 8, 8), torch.ones(2, dtype=torch.bool)
    inputs = (image, image_mask, tokens, torch.ones_like(tokens, dtype=torch.bool))
    with torch.no_grad():
        eager = model(*inputs)
        assert eager.tolist() == [[END_OF_ACTION, 0, 0, 0]] * 2
        # torch.compile on PyTorch 2.11 also reports is_exporting(); the early stop must survive it.
        assert torch.equal(torch.compile(model, backend="eager")(*inputs), eager)
        # The exported loop runs every step, then zero-fills the steps after the stop, as eager leaves them.
        exported = torch.export.export(model, inputs, strict=False).module()(*inputs)
    assert torch.equal(exported, eager)


class ScriptedHead(nn.Module):
    """An lm_head that emits the next step of a token script on each decoder pass, and counts the passes."""

    def __init__(self, script):
        super().__init__()
        self.register_buffer("logits", 100 * nn.functional.one_hot(torch.tensor(script), 64).float())
        self.register_buffer("passes", torch.zeros(1, dtype=torch.long))

    def forward(self, hidden):
        logits = self.logits.index_select(0, self.passes).transpose(0, 1)
        self.passes += 1
        return logits


def test_greedy_decoding_keeps_each_row_until_the_batch_wide_stop(monkeypatch):
    monkeypatch.setattr(modeling_pi0_fast, "PI0FastPaliGemma", TinyPaliGemma)
    core = modeling_pi0_fast.PI0FastPytorch(
        PI0FastConfig(dtype=torch.float32), paligemma_tokenizer=Tokenizer()
    )
    # One step per decoder pass: the first row ends at once, the second after two more tokens.
    head = ScriptedHead([[END_OF_ACTION, 10], [20, 11], [21, END_OF_ACTION], [22, 12]])
    core.paligemma_with_expert.paligemma.lm_head = head

    class Decode(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = core

        def forward(self, image, image_mask, tokens, masks):
            self.core.paligemma_with_expert.paligemma.lm_head.passes.zero_()
            return self.core.sample_actions_fast_kv_cache(
                [image], [image_mask], tokens, masks, max_decoding_steps=4, temperature=0
            )

    model = Decode().eval()
    tokens = torch.tensor([[4, 5], [6, 7]])
    image, image_mask = torch.randn(2, 3, 8, 8), torch.ones(2, dtype=torch.bool)
    inputs = (image, image_mask, tokens, torch.ones_like(tokens, dtype=torch.bool))
    expected = [[END_OF_ACTION, 20, 21, 0], [10, 11, END_OF_ACTION, 0]]
    with torch.no_grad():
        assert model(*inputs).tolist() == expected
        # The prefill and two decoding steps; the fourth pass would mean the early stop was lost.
        assert head.passes.item() == 3
        assert torch.compile(model, backend="eager")(*inputs).tolist() == expected
        assert head.passes.item() == 3
        exported = torch.export.export(model, inputs, strict=False).module()(*inputs)
    assert exported.tolist() == expected


def test_decoding_drops_the_tokens_after_the_end_of_action_marker():
    pytest.importorskip("scipy")
    policy = modeling_pi0_fast.PI0FastPolicy.__new__(modeling_pi0_fast.PI0FastPolicy)
    nn.Module.__init__(policy)
    policy.config = PI0FastConfig(validate_action_token_prefix=False)
    policy._paligemma_tokenizer = Tokenizer()
    fast = SimpleNamespace(decode=lambda tokens: "".join(chr(int(t) % 128) for t in tokens))
    policy.action_tokenizer = SimpleNamespace(min_token=0, scale=1.0, bpe_tokenizer=fast)
    # A row that ends before the others keeps the tokens decoded after its marker until the batch-wide stop.
    ended_last = torch.tensor([[5, 6, END_OF_ACTION, 0]])
    ended_first = torch.tensor([[5, 6, END_OF_ACTION, 9]])
    actions = [
        policy.detokenize_actions(tokens, action_horizon=2, action_dim=2)
        for tokens in (ended_last, ended_first)
    ]
    assert actions[0].abs().sum() > 0
    assert torch.equal(*actions)
