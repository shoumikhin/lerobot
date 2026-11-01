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

"""pi0-FAST's export recipe end to end on a tiny pi0-FAST."""

import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file, save_file

transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("scipy")

from lerobot.bundle.action_tokens import ActionDecoder  # noqa: E402
from lerobot.bundle.engine import BundleEngine  # noqa: E402
from lerobot.bundle.manifest import DECODED_ACTIONS, GENERATED_TOKENS, Bundle  # noqa: E402
from lerobot.policies import make_pre_post_processors, prepare_observation_for_inference  # noqa: E402
from lerobot.policies.pi0_fast import PI0FastPolicy  # noqa: E402
from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig  # noqa: E402
from lerobot.policies.pi0_fast.export_pi0_fast import PI0FastExportRecipe, PI0FastPrefill  # noqa: E402
from lerobot.policies.pi0_fast.processor_pi0_fast import (  # noqa: E402
    Pi0FastPrepareStateAndLanguageTokenizerProcessorStep,
)
from lerobot.processor import TransitionKey  # noqa: E402
from lerobot.scripts.lerobot_export import export  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402
from tests.export.plugins import use_plugins  # noqa: E402
from tests.export.tiny_pi0_fast import (  # noqa: E402
    ACTION_DIM,
    IMAGE,
    SCALE,
    SKIP_TOKENS,
    VOCABULARY,
    save_checkpoint,
    use_tiny_pi0_fast,
)

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
def exported(tmp_path_factory):
    """A tiny pi0-FAST checkpoint, its config, and its bundle exported with the test backend."""
    folder = tmp_path_factory.mktemp("pi0_fast")
    with pytest.MonkeyPatch.context() as patch:
        config = save_checkpoint(folder / "checkpoint", patch)
        # The export's steps run in processes of their own, which find the test backend and make pi0-FAST tiny.
        use_plugins(patch, "torch_backend", "tiny_pi0_fast")
        args = [
            f"--policy.path={folder / 'checkpoint'}",
            f"--output_dir={folder / 'bundle'}",
            f"--task={TASK}",
        ]
        export(["--backend.type=torch_test", *args])
    return folder / "checkpoint", config, Bundle.read(folder / "bundle")


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


def test_manifest_generates_the_tokens_and_carries_the_fast_tokenizer(exported):
    checkpoint, config, bundle = exported
    manifest = bundle.manifest
    steps = [f"step_{t}" for t in range(1, config.max_decoding_steps)]
    assert [run.name for run in manifest.runs] == ["prefill", *steps, "actions"]
    assert {run.program for run in manifest.runs if run.name in steps} == {"step"}
    assert sorted(manifest.programs) == ["actions", "prefill", "step"]
    assert GENERATED_TOKENS in manifest.runs[-2].outputs and manifest.runs[-1].inputs == [DECODED_ACTIONS]
    constants = load_file(bundle.path / manifest.constants)
    assert [constants[step].tolist() for step in steps] == [[t] for t in range(1, config.max_decoding_steps)]
    spec = manifest.action_tokens
    fast = checkpoint.parent / "fast"
    assert json.loads((bundle.path / spec.tokenizer).read_text()) == json.loads(
        (fast / "tokenizer.json").read_text()
    )
    min_token = json.loads((fast / "processor_config.json").read_text())["min_token"]
    assert (spec.min_token, spec.scale, spec.clean_up_spaces) == (min_token, SCALE, True)
    assert (spec.horizon, spec.action_dim, spec.id_offset) == (
        config.n_action_steps,
        ACTION_DIM,
        VOCABULARY - 1 - SKIP_TOKENS,
    )


def test_bundle_plays_the_actions_of_the_pytorch_policy(exported, monkeypatch):
    checkpoint, config, bundle = exported
    policy, preprocessor, postprocessor = load_policy(checkpoint, config, monkeypatch)
    engine = BundleEngine(bundle)
    case = load_file(bundle.path / bundle.manifest.test_case)
    rng = np.random.default_rng(1)
    played = []
    for i in range(4):
        image = case[IMAGE] if i == 0 else rng.integers(0, 256, case[IMAGE].shape, dtype=np.uint8)
        state = case[OBS_STATE] if i == 0 else rng.uniform(-1.5, 1.5, 3).astype(np.float32)
        frame = {IMAGE: image, OBS_STATE: state}
        actions = np.stack([engine.get_action(dict(frame)) for _ in range(config.n_action_steps)])
        with torch.no_grad():
            batch = preprocessor(prepare_observation_for_inference(dict(frame), torch.device("cpu"), TASK))
            expected = postprocessor(policy.predict_action_chunk(batch))[0].numpy()
        np.testing.assert_allclose(actions, expected, rtol=0, atol=1e-5)
        played.append(expected)
    # Some frames' answers hold FAST tokens, so the comparison covers the decoding too.
    assert len({chunk.tobytes() for chunk in played if np.abs(chunk).max() > 0.1}) >= 2


def test_bundle_decodes_answers_as_the_policy_does(exported, monkeypatch):
    checkpoint, config, bundle = exported
    policy, _, _ = load_policy(checkpoint, config, monkeypatch)
    spec = bundle.manifest.action_tokens
    tokenizer = json.loads((bundle.path / spec.tokenizer).read_text())
    fast = policy.action_tokenizer.bpe_tokenizer
    for clean_up in (True, False):
        monkeypatch.setattr(fast, "clean_up_tokenization_spaces", clean_up)
        decoder = ActionDecoder(tokenizer, replace(spec, clean_up_spaces=clean_up))
        for answer in answers(spec, len(fast), 100):
            expected = policy.detokenize_actions(answer, spec.horizon, spec.action_dim).numpy()
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


def test_recipe_refuses_a_chain_that_generates_other_tokens(exported, monkeypatch):
    recipe = make_recipe(*exported[:2], monkeypatch)
    prefill = PI0FastPrefill.forward

    def first_token_off_by_one(self, *frame):
        tokens, *rest = prefill(self, *frame)
        return (tokens + 1) % VOCABULARY, *rest

    monkeypatch.setattr(PI0FastPrefill, "forward", first_token_off_by_one)
    with pytest.raises(
        ValueError, match="chain generates token .* where the policy generates .*, at position 0"
    ):
        next(recipe.parts())


def test_recipe_refuses_a_decoding_that_differs_from_the_policy(exported, monkeypatch):
    recipe = make_recipe(*exported[:2], monkeypatch)
    decode = ActionDecoder.__call__
    monkeypatch.setattr(ActionDecoder, "__call__", lambda self, tokens: decode(self, tokens) + 1)
    with pytest.raises(ValueError, match="decoding of pi0-FAST's tokens differs from the policy's"):
        next(recipe.parts())


def test_export_refuses_a_checkpoint_that_leaves_a_weight_unfilled(exported, tmp_path, monkeypatch, capfd):
    partial = tmp_path / "checkpoint"
    shutil.copytree(exported[0], partial)
    weights = load_file(partial / "model.safetensors")
    del weights["paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"]
    save_file(weights, partial / "model.safetensors", metadata={"format": "pt"})
    use_plugins(monkeypatch, "torch_backend", "tiny_pi0_fast")
    args = [f"--policy.path={partial}", f"--output_dir={tmp_path / 'bundle'}", f"--task={TASK}"]
    with pytest.raises(RuntimeError, match="trace step failed"):
        export(["--backend.type=torch_test", *args])
    assert "multi_modal_projector.linear.bias" in capfd.readouterr().err
