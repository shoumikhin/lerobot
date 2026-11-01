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

"""A bundle's action tokens: the actions the bundle decodes from the tokens its programs generate."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("scipy")

from lerobot.bundle.action_tokens import ActionDecoder  # noqa: E402
from lerobot.bundle.engine import BundleEngine  # noqa: E402
from lerobot.bundle.manifest import (  # noqa: E402
    ACTION_TOKENIZER,
    DECODED_ACTIONS,
    GENERATED_TOKENS,
    MANIFEST,
    TEST_CASE,
    ActionTokens,
    Bundle,
)
from lerobot.policies.pi0_fast import PI0FastPolicy  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402
from tests.export.numpy_bundle import make_token_bundle  # noqa: E402
from tests.export.tiny_pi0_fast import (  # noqa: E402
    ACTION_DIM,
    HORIZON,
    TinyFastProcessor,
    save_fast_tokenizer,
)
from tests.torch_free import CHECK_BUNDLE, assert_runs_without_torch  # noqa: E402

REPO = Path(__file__).parents[2]
PREFIX, END = [2, 3], 1


@pytest.fixture
def token_bundle(tmp_path):
    """A bundle that generates `Action: `, FAST tokens, the end marker and a token after it, whose test case
    holds the actions pi0-FAST decodes from them."""
    save_fast_tokenizer(tmp_path / "fast")
    processor = TinyFastProcessor.from_pretrained(tmp_path / "fast")
    fast_ids = np.random.default_rng(0).integers(len(processor.bpe_tokenizer), size=6)
    spec = ActionTokens(
        tokenizer=ACTION_TOKENIZER,
        clean_up_spaces=True,
        id_offset=len(processor.bpe_tokenizer) + 9,
        end=END,
        prefix=PREFIX,
        start=PREFIX,
        min_token=processor.min_token,
        scale=processor.scale,
        horizon=HORIZON,
        action_dim=ACTION_DIM,
    )
    tokens = np.array([*PREFIX, *(spec.id_offset - fast_ids), END, 50])
    # The policy's own decoding, which reads only its FAST processor.
    actions = PI0FastPolicy.decode_actions_with_fast(
        SimpleNamespace(action_tokenizer=processor), [fast_ids.tolist()], HORIZON, ACTION_DIM
    )[0]
    tokenizer = processor.bpe_tokenizer.backend_tokenizer.to_str()
    return make_token_bundle(tmp_path / "bundle", tokenizer, spec, tokens, actions.astype(np.float32))


def _edit(path: Path, **changes) -> None:
    data = json.loads((path / MANIFEST).read_text())
    for key, value in changes.items():
        data[key] = {**data[key], **value} if isinstance(value, dict) else value
    (path / MANIFEST).write_text(json.dumps(data))


def test_the_bundle_check_decodes_the_tokens_without_torch(token_bundle, monkeypatch):
    path = token_bundle
    # The check does not look in the current folder, so the test runtime's module comes from PYTHONPATH.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))
    assert_runs_without_torch(CHECK_BUNDLE, str(path))


# A locale whose encoding is ASCII, which Python neither turns into UTF-8 nor overrides with its UTF-8 mode.
ASCII_LOCALE = {"LC_ALL": "C", "PYTHONCOERCECLOCALE": "0", "PYTHONUTF8": "0"}

WRITE_TOKENIZER = """
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lerobot.export.bundle import write_bundle

folder = Path(sys.argv[1])
manifest = SimpleNamespace(
    test_case="test.safetensors",
    constants=None,
    task_tokens=None,
    action_tokens=SimpleNamespace(tokenizer="tokenizer.json"),
)
config = SimpleNamespace(save_pretrained=lambda path: (path / "config.json").write_text("{}"))
case = SimpleNamespace(frame={}, actions=np.zeros((1, 1), np.float32), noise=None)
tokens = SimpleNamespace(tokenizer='{"\\u0120a": 1}')
write_bundle(SimpleNamespace(path=folder, manifest=manifest), case, config, action_tokens=tokens)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="sets an ASCII locale through the environment")
def test_the_fast_tokenizer_is_written_and_read_as_utf8_in_an_ascii_locale(token_bundle, tmp_path):
    """Byte-level BPE vocabularies, such as FAST's, hold characters like U+0120 that ASCII cannot encode."""
    env = {**os.environ, **ASCII_LOCALE}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))
    folder = tmp_path / "written"
    folder.mkdir()
    subprocess.run([sys.executable, "-c", WRITE_TOKENIZER, str(folder)], env=env, check=True)
    assert (folder / "tokenizer.json").read_bytes() == '{"\u0120a": 1}'.encode()
    path = token_bundle
    assert not (path / ACTION_TOKENIZER).read_bytes().isascii()
    subprocess.run([sys.executable, "-m", "lerobot.bundle", str(path)], env=env, check=True)


def test_bundle_refuses_action_tokens_it_does_not_know(token_bundle):
    path = token_bundle
    _edit(path, action_tokens={"vocabulary": "fast"})
    with pytest.raises(ValueError, match=r"\['vocabulary'\].*Upgrade LeRobot"):
        Bundle.read(path)


def test_engine_refuses_tokens_that_do_not_start_as_the_policy_checks(token_bundle):
    path = token_bundle
    _edit(path, action_tokens={"start": PREFIX[::-1]})
    with pytest.raises(RuntimeError, match='do not start with "Action:"'):
        BundleEngine(Bundle.read(path))


@pytest.mark.parametrize(
    "runs, error",
    [
        (
            [
                ["tokens", [], [GENERATED_TOKENS, DECODED_ACTIONS], "tokens"],
                ["actions", [DECODED_ACTIONS], [ACTION], "actions"],
            ],
            "writes 'decoded_actions', which the bundle decodes itself",
        ),
        (
            [
                ["tokens", [], [GENERATED_TOKENS], "tokens"],
                ["first", [DECODED_ACTIONS], ["first"], "actions"],
                ["retokens", [], [GENERATED_TOKENS], "tokens"],
                ["actions", [DECODED_ACTIONS], [ACTION], "actions"],
            ],
            "writes 'generated_tokens' after a run read their decoded actions",
        ),
    ],
)
def test_engine_refuses_runs_that_do_not_leave_it_the_decoding(token_bundle, runs, error):
    path = token_bundle
    _edit(
        path,
        runs=[dict(zip(("name", "inputs", "outputs", "program"), run, strict=True)) for run in runs],
    )
    with pytest.raises(ValueError, match=error):
        BundleEngine(Bundle.read(path))


def test_engine_refuses_decoded_actions_of_tokens_that_no_run_writes(token_bundle):
    """Tokens in the frame are not decoded: the bundle decodes the tokens a run of the chunk writes."""
    path = token_bundle
    case = load_file(path / TEST_CASE)
    save_file({**case, GENERATED_TOKENS: np.load(path / "tokens.npy")}, path / TEST_CASE)
    run = {"name": "actions", "inputs": [DECODED_ACTIONS], "outputs": [ACTION], "program": "actions"}
    _edit(path, runs=[run])
    with pytest.raises(ValueError, match="no earlier run of its chunk writes 'generated_tokens'"):
        BundleEngine(Bundle.read(path))


@pytest.mark.parametrize(
    "tokenizer_change, spec_change, error",
    [
        ({"model": {"type": "WordPiece"}}, {}, "must be a byte-level BPE without added tokens"),
        ({"decoder": {"type": "Metaspace"}}, {}, "must be a byte-level BPE without added tokens"),
        ({"added_tokens": [{"id": 0}]}, {}, "must be a byte-level BPE without added tokens"),
        ({}, {"prefix": []}, "name no answer prefix"),
    ],
)
def test_decoder_refuses_what_it_cannot_decode_as_the_policy_does(
    token_bundle, tokenizer_change, spec_change, error
):
    path = token_bundle
    spec = Bundle.read(path).manifest.action_tokens
    tokenizer = json.loads((path / spec.tokenizer).read_text(encoding="utf-8")) | tokenizer_change
    with pytest.raises(ValueError, match=error):
        ActionDecoder(tokenizer, replace(spec, **spec_change))


@pytest.mark.parametrize("cleans_up", [False, True])
def test_the_bundle_decodes_a_space_before_punctuation_as_the_policy_does(cleans_up):
    """Transformers 5 ignores `clean_up_tokenization_spaces` for a BPE tokenizer unless forced to apply it,
    so the export records whether the policy's tokenizer really drops the space before a period."""
    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.trainers import BpeTrainer
    from transformers import PreTrainedTokenizerFast

    from lerobot.policies.pi0_fast.export_pi0_fast import cleans_up_spaces

    # An alphabet up to "<", so the coefficients can decode to a space before a period, a comma or a "!".
    alphabet = [chr(i) for i in range(ord("<") + 1)]
    bpe = ByteLevelBPETokenizer()
    trainer = BpeTrainer(vocab_size=100, initial_alphabet=alphabet, show_progress=False)
    bpe._tokenizer.train_from_iterator(["".join(alphabet)], trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=bpe,
        clean_up_tokenization_spaces=True,
        clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=cleans_up,
    )
    assert cleans_up_spaces(tokenizer) == cleans_up
    fast_ids = tokenizer.encode("a . b , c ! d", add_special_tokens=False)
    processor = TinyFastProcessor(tokenizer, min_token=-20, scale=10.0)
    actions = PI0FastPolicy.decode_actions_with_fast(
        SimpleNamespace(action_tokenizer=processor), [fast_ids], HORIZON, ACTION_DIM
    )[0]
    spec = ActionTokens(
        tokenizer=ACTION_TOKENIZER,
        clean_up_spaces=cleans_up_spaces(tokenizer),
        id_offset=len(tokenizer) + 9,
        end=END,
        prefix=PREFIX,
        start=PREFIX,
        min_token=processor.min_token,
        scale=processor.scale,
        horizon=HORIZON,
        action_dim=ACTION_DIM,
    )
    decoder = ActionDecoder(json.loads(tokenizer.backend_tokenizer.to_str()), spec)
    tokens = np.array([*PREFIX, *(spec.id_offset - np.array(fast_ids)), END])
    np.testing.assert_allclose(decoder(tokens), actions, rtol=0, atol=1e-6)
