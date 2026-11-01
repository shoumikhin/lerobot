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
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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
    ActionTokens,
    Bundle,
    BundleManifest,
)
from lerobot.policies.pi0_fast import PI0FastPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from tests.export.numpy_bundle import make_token_bundle  # noqa: E402
from tests.export.tiny_pi0_fast import (  # noqa: E402
    ACTION_DIM,
    HORIZON,
    TinyFastProcessor,
    save_fast_tokenizer,
)
from tests.torch_free import assert_runs_without_torch  # noqa: E402

REPO = Path(__file__).parents[2]
PREFIX, END = [2, 3], 1


@pytest.fixture
def token_bundle(tmp_path):
    """A bundle that generates `Action: `, FAST tokens, the end marker and a token after it, and the actions
    pi0-FAST decodes from them."""
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
    policy = SimpleNamespace(action_tokenizer=processor)
    actions = PI0FastPolicy.decode_actions_with_fast(policy, [fast_ids.tolist()], HORIZON, ACTION_DIM)[0]
    tokenizer = processor.bpe_tokenizer.backend_tokenizer.to_str()
    return make_token_bundle(
        tmp_path / "bundle", tokenizer, spec, tokens, actions.astype(np.float32)
    ), actions


def _edit(path: Path, **changes) -> None:
    data = json.loads((path / MANIFEST).read_text())
    for key, value in changes.items():
        data[key] = {**data[key], **value} if isinstance(value, dict) else value
    (path / MANIFEST).write_text(json.dumps(data))


def test_engine_plays_the_actions_it_decodes_from_the_tokens(token_bundle):
    path, actions = token_bundle
    engine = BundleEngine(Bundle.read(path))
    played = np.stack([engine.get_action({OBS_STATE: np.zeros(1, np.float32)}) for _ in range(HORIZON)])
    np.testing.assert_allclose(played, actions, rtol=0, atol=1e-6)
    assert np.abs(actions).max() > 0.1


def test_the_bundle_check_decodes_the_tokens_without_torch(token_bundle, monkeypatch):
    path, _ = token_bundle
    # The check does not look in the current folder, so the test runtime's module comes from PYTHONPATH.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))
    assert_runs_without_torch(
        """
        import runpy
        import sys

        sys.argv = ["lerobot.bundle", sys.argv[1]]
        runpy.run_module("lerobot.bundle", run_name="__main__")
        """,
        str(path),
    )


def test_engine_refuses_tokens_that_do_not_start_as_the_policy_checks(token_bundle):
    path, _ = token_bundle
    _edit(path, action_tokens={"start": PREFIX[::-1]})
    with pytest.raises(RuntimeError, match='do not start with "Action:"'):
        BundleEngine(Bundle.read(path))


@pytest.mark.parametrize(
    "runs, error",
    [
        (
            [["tokens", [], [GENERATED_TOKENS, DECODED_ACTIONS]], ["actions", [DECODED_ACTIONS], [ACTION]]],
            "writes 'decoded_actions', which the bundle decodes itself",
        ),
        (
            [["actions", [DECODED_ACTIONS], [ACTION]], ["tokens", [], [GENERATED_TOKENS]]],
            r"'actions' reads \['decoded_actions'\]",
        ),
    ],
)
def test_engine_refuses_runs_that_do_not_leave_it_the_decoding(token_bundle, runs, error):
    path, _ = token_bundle
    _edit(
        path,
        runs=[dict(zip(("name", "inputs", "outputs"), run, strict=True), program=run[0]) for run in runs],
    )
    with pytest.raises(ValueError, match=error):
        BundleEngine(Bundle.read(path))


@pytest.mark.parametrize(
    "tokenizer_change, spec_change, error",
    [
        ({"model": {"type": "WordPiece"}}, {}, "must be a byte-level BPE without added tokens"),
        ({"decoder": {"type": "Metaspace"}}, {}, "must be a byte-level BPE without added tokens"),
        (
            {"added_tokens": [{"id": 0, "content": "<s>"}]},
            {},
            "must be a byte-level BPE without added tokens",
        ),
        ({}, {"prefix": []}, "name no answer prefix"),
    ],
)
def test_decoder_refuses_what_it_cannot_decode_as_the_policy_does(
    token_bundle, tokenizer_change, spec_change, error
):
    path, _ = token_bundle
    spec = Bundle.read(path).manifest.action_tokens
    tokenizer = json.loads((path / spec.tokenizer).read_text()) | tokenizer_change
    with pytest.raises(ValueError, match=error):
        ActionDecoder(tokenizer, replace(spec, **spec_change))


def test_manifest_refuses_action_tokens_it_does_not_know(token_bundle):
    path, _ = token_bundle
    data = json.loads((path / MANIFEST).read_text())
    data["action_tokens"] = asdict(Bundle.read(path).manifest.action_tokens) | {"vocabulary": 2048}
    with pytest.raises(ValueError, match=r"\['vocabulary'\].*Upgrade LeRobot"):
        BundleManifest.from_dict(data)
