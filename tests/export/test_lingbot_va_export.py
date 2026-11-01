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

"""LingBot-VA exported as programs that keep its caches between chunks, end to end on a tiny LingBot-VA."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from packaging.version import Version

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from lerobot.bundle.engine import BundleEngine  # noqa: E402
from lerobot.bundle.manifest import Bundle  # noqa: E402
from lerobot.policies import make_pre_post_processors, prepare_observation_for_inference  # noqa: E402
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig  # noqa: E402
from lerobot.policies.lingbot_va.export_lingbot_va import (  # noqa: E402
    CARRY,
    LingBotVAExportRecipe,
    _Allocate,
    _ClearPredicted,
    _Write,
)
from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy  # noqa: E402
from lerobot.policies.lingbot_va.utils import WanAttention  # noqa: E402
from lerobot.processor import policy_output_to_transition, transition_to_policy_action  # noqa: E402
from lerobot.scripts.lerobot_export import export  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402
from lerobot.utils.import_utils import is_package_available  # noqa: E402
from tests.export.plugins import use_plugins  # noqa: E402
from tests.export.tiny_lingbot_va import CAMERAS, TASK, save_checkpoint  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def torch_backend_plugin():
    """Add the test backend to the export's steps, which run in processes of their own."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        use_plugins(monkeypatch, "torch_backend")
        yield


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    folder = tmp_path_factory.mktemp("lingbot_va")
    config = save_checkpoint(folder / "checkpoint")
    export(
        [
            "--backend.type=torch_test",
            f"--policy.path={folder / 'checkpoint'}",
            f"--output_dir={folder / 'bundle'}",
            f"--task={TASK}",
        ]
    )
    return folder / "checkpoint", config, Bundle.read(folder / "bundle")


def test_bundle_runs_two_first_chunks_then_the_same_runs_with_one_copy_of_the_blocks(exported):
    _, config, bundle = exported
    manifest = bundle.manifest
    first, second = manifest.first_runs
    assert [run.program for run in first[:2]] == ["encode_first", "empty_cache"]
    assert [run.program for run in second[:2]] == ["clear_predicted", "encode_start"]
    assert [run.program for run in manifest.runs[:2]] == ["clear_predicted", "encode_next"]
    assert first[-1].program == "actions_first" and manifest.runs[-1].program == "actions"
    passes = (config.num_inference_steps + 1) + (config.action_num_inference_steps + 1)
    for runs, feedback in ((first, 0), (second, 2), (manifest.runs, 2)):
        assert sum(run.program == "blocks" for run in runs) == passes + feedback
    assert manifest.carry == dict.fromkeys(CARRY)
    assert manifest.history == dict.fromkeys(CAMERAS, config.frame_chunk_size * config.action_per_frame)
    assert manifest.task == TASK


def test_each_program_holds_only_the_weights_it_runs(exported):
    checkpoint, config, bundle = exported
    policy = LingBotVAPolicy.from_pretrained(checkpoint)
    policy._ensure_frozen_modules()
    weight = policy.transformer.blocks[0].attn1.to_q.weight
    sizes, holders = {}, []
    for name, files in bundle.manifest.programs.items():
        weights = list(torch.export.load(bundle.path / files.file).state_dict.values())
        sizes[name] = sum(t.numel() for t in weights)
        holders += [name for t in weights if t.shape == weight.shape and torch.equal(t, weight)]
    assert holders == ["blocks"]
    # The blocks and the projected text of the task and of the negative prompt.
    text = 2 * config.max_sequence_length * config.num_attention_heads * config.attention_head_dim
    assert sizes["blocks"] == sum(p.numel() for p in policy.transformer.blocks.parameters()) + text
    # The VAE's encoder, without its decoder.
    encoder = sum(p.numel() for m in (policy._vae.encoder, policy._vae.quant_conv) for p in m.parameters())
    assert sizes["encode_first"] == sizes["encode_start"] == sizes["encode_next"] == encoder


def play_policy(checkpoint, frames, noises, reset_at):
    """The actions the PyTorch policy plays at each frame, as lerobot-rollout plays them.

    Each chunk draws the next of `noises`, and the policy is reset before the frame `reset_at`.
    """
    policy = LingBotVAPolicy.from_pretrained(checkpoint).eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    latent_h, latent_w = policy._latent_hw
    config = policy.config
    video_shape = (1, config.in_channels, config.frame_chunk_size, latent_h, latent_w)
    action_shape = (1, config.action_dim, config.frame_chunk_size, config.action_per_frame, 1)
    noises = iter(noises)
    played = []
    for tick, frame in enumerate(frames):
        if tick == reset_at:
            policy.reset()
        if policy.count_queued_actions() == 0:
            flat = torch.from_numpy(next(noises))
            video, action = flat.split([int(np.prod(video_shape)), int(np.prod(action_shape))])
        with torch.no_grad():
            batch = preprocessor(prepare_observation_for_inference(dict(frame), torch.device("cpu"), TASK))
            output = policy.select_action(
                batch, noise=action.view(action_shape), video_noise=video.view(video_shape)
            )
            played.append(
                postprocessor(transition_to_policy_action(policy_output_to_transition(output)))[0].numpy()
            )
    return np.stack(played)


def test_bundle_plays_the_actions_of_the_pytorch_policy_chunk_after_chunk(exported):
    checkpoint, config, bundle = exported
    engine = BundleEngine(bundle)
    rng = np.random.default_rng(1)
    first_chunk = (config.frame_chunk_size - 1) * config.action_per_frame
    chunk = config.frame_chunk_size * config.action_per_frame
    # Past the KV cache's first eviction, then a reset and two more chunks.
    reset_at = first_chunk + 4 * chunk
    frames = [
        {
            **{name: rng.integers(0, 256, (48, 48, 3), dtype=np.uint8) for name in CAMERAS},
            OBS_STATE: rng.random(3, dtype=np.float32),
        }
        for _ in range(reset_at + first_chunk + chunk)
    ]
    np.random.seed(0)
    actions = []
    for tick, frame in enumerate(frames):
        if tick == reset_at:
            engine.reset()
        actions.append(engine.get_action(dict(frame)))
    draws = np.random.RandomState(0)
    noises = [draws.standard_normal(bundle.manifest.noise_shape).astype(np.float32) for _ in range(8)]

    expected = play_policy(checkpoint, frames, noises, reset_at)
    scale = np.asarray(bundle.manifest.action_scale)
    assert np.abs((np.stack(actions) - expected) / scale).max() < 1e-4


def test_static_cache_takes_and_frees_slots_as_the_policy_does():
    torch.manual_seed(0)
    heads, head_dim, batch, latent, action, window = 2, 4, 2, 6, 4, 4
    slots = (window // 2) * (latent + action)
    attention = WanAttention(dim=heads * head_dim, heads=heads, dim_head=head_dim)
    attention.init_kv_cache("pos", slots, heads, head_dim, "cpu", torch.float32, batch)
    cache = attention.attn_caches["pos"]
    keys = torch.zeros(1, batch, slots, heads, head_dim)
    values = torch.zeros_like(keys)
    valid = torch.zeros(slots, dtype=torch.bool)
    ids = torch.full((slots,), -1)
    predicted = torch.zeros(slots, dtype=torch.bool)
    allocate = {latent: _Allocate(latent), action: _Allocate(action)}
    write = {latent: _Write(latent), action: _Write(action)}
    tokens = max(latent, action)

    def check():
        assert torch.equal(cache["mask"], valid)
        assert torch.equal(cache["id"][valid], ids[valid])
        assert torch.equal(cache["is_pred"][valid], predicted[valid])
        assert torch.equal(cache["k"][:, valid], keys[0][:, valid])
        assert torch.equal(cache["v"][:, valid], values[0][:, valid])

    for chunk in range(30):
        if chunk:
            attention.clear_pred_cache("pos")
            valid = _ClearPredicted()(valid, predicted)
        for count, passes in ((latent, 0), (action, 0), (latent, 3), (action, 3)):
            # The observed latents and actions are written once; the predicted ones after passes that write
            # and then free their slots, as the denoising steps do.
            is_pred = passes > 0
            new_keys = torch.randn(1, batch, tokens, heads, head_dim)
            new_values = torch.randn_like(new_keys)
            for _ in range(passes):
                taken = attention.update_cache(
                    "pos", torch.randn(batch, count, heads, head_dim), new_values[0, :, :count], False
                )
                attention.restore_cache("pos", taken)
            attention.update_cache("pos", new_keys[0, :, :count], new_values[0, :, :count], is_pred)
            valid, taken, cache_id = allocate[count](valid, ids)
            keys, values, valid, ids, predicted = write[count](
                keys,
                values,
                valid,
                ids,
                predicted,
                taken,
                cache_id,
                new_keys,
                new_values,
                torch.tensor([is_pred]),
            )
            check()
    # The first chunk's observed latents and actions were freed for newer ones.
    assert int(ids[valid].min()) > 1


# ExecuTorch 1.6 reads programs traced by PyTorch 2.15 or newer.
EXECUTORCH = is_package_available("executorch") and Version(torch.__version__).release >= (2, 15)


@pytest.mark.skipif(not EXECUTORCH, reason="needs ExecuTorch and PyTorch 2.15")
def test_cache_allocation_runs_in_executorch_on_the_cpu(tmp_path):
    from lerobot.bundle.executorch import ExecuTorchProgram
    from lerobot.export.backends.executorch_xnnpack import ExecuTorchXNNPACKConfig

    allocate = _Allocate(3)
    path = tmp_path / "allocate.pte"
    empty = (torch.zeros(8, dtype=torch.bool), torch.full((8,), -1))
    ExecuTorchXNNPACKConfig().compile(torch.export.export(allocate, empty), path)
    program = ExecuTorchProgram(path)
    states = [
        ([0, 0, 0, 0, 0, 0, 0, 0], [-1, -1, -1, -1, -1, -1, -1, -1]),
        ([1, 0, 1, 0, 0, 1, 1, 0], [0, -1, 1, -1, -1, 1, 2, -1]),
        # One slot is free, so the two slots of the oldest id are freed for the other two.
        ([1, 1, 1, 0, 1, 1, 1, 1], [3, 1, 1, -1, 2, 2, 3, 2]),
    ]
    for mask, slot_ids in states:
        valid, ids = torch.tensor(mask, dtype=torch.bool), torch.tensor(slot_ids)
        actual = program(valid.numpy(), ids.numpy())
        for value, expected in zip(actual, allocate(valid, ids), strict=True):
            np.testing.assert_array_equal(program.to_numpy(value), expected.numpy())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"camera_layout": "robotwin_tshape"}, "two streaming VAEs"),
        ({"frame_chunk_size": 1}, "frame_chunk_size is 1"),
        ({"action_per_frame": 6}, "multiple of 4"),
    ],
)
def test_export_refuses_what_it_does_not_run(change, message):
    config = LingBotVAConfig(**change)
    with pytest.raises(ValueError, match=message):
        LingBotVAExportRecipe(SimpleNamespace(config=config), None, None, None)
