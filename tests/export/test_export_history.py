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

"""The recipes of the policies that read past frames, end to end on tiny checkpoints.

Each bundle is played next to the PyTorch policy in `lerobot-rollout`'s engines, on the same frames and the
same random draws, for more ticks than one chunk, so both the history and the noise are exercised.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from lerobot.bundle.engine import BundleEngine
from lerobot.bundle.manifest import NOISE, Bundle, ProgramRun
from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.export import worker
from lerobot.export.recipe import ReplayDraws, frame_shapes
from lerobot.inference.sync import SyncInferenceEngine
from lerobot.policies import make_pre_post_processors
from lerobot.policies.common.export_history import HistoryChunkRecipe
from lerobot.policies.factory import get_policy_class
from lerobot.processor import ProcessorStep, TransitionKey
from lerobot.scripts.lerobot_export import export
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.export import torch_backend  # noqa: F401  # registers torch_test for the in-process trace step
from tests.export.backends import BACKENDS as EXECUTORCH_BACKENDS, GPU_BACKENDS
from tests.export.plugins import use_plugins

IMAGE = "observation.images.front"
# Big enough for a 3x3 last ResNet feature map: on a 1x1 map SpatialSoftmax has one cell, so the camera
# would not change the actions.
FRAME = (96, 96)
CROP = (84, 84)
TICKS = 20
BACKENDS = {"torch_test": [], **{name: [mark] for name, mark in EXECUTORCH_BACKENDS.items()}}


def _min_max(size: int) -> dict[str, torch.Tensor]:
    return {"min": torch.full((size,), -2.0), "max": torch.full((size,), 3.0)}


def vqbet_checkpoint(path, actions: int = 2) -> None:
    from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig

    config = VQBeTConfig(
        input_features={
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *FRAME)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(actions,))},
        n_obs_steps=3,
        n_action_pred_token=2,
        action_chunk_size=4,
        crop_shape=CROP,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=8,
        vqvae_n_embed=8,
        vqvae_embedding_dim=16,
        vqvae_enc_hidden_dim=16,
        gpt_input_dim=32,
        gpt_output_dim=32,
        gpt_hidden_dim=32,
        gpt_n_layer=1,
        gpt_n_head=2,
        # Spread the code probabilities, so that the draws, not the logits, pick the codes.
        bet_softmax_temperature=10.0,
        device="cpu",
    )
    stats = {OBS_STATE: _min_max(2), ACTION: _min_max(actions)}
    _save(config, stats, path)


def _save(config, stats, path) -> None:
    torch.manual_seed(0)
    get_policy_class(config.type)(config).save_pretrained(path)
    for processor in make_pre_post_processors(config, dataset_stats=stats):
        processor.save_pretrained(path)


CHECKPOINTS = {"vqbet": vqbet_checkpoint}


def play(path: Path, bundle: Bundle, monkeypatch) -> list[float]:
    """The largest difference on each tick between the bundle and the PyTorch policy, on the same draws, in
    normalized action units."""
    policy = get_policy_class(PreTrainedConfig.from_pretrained(path).type).from_pretrained(path)
    reference = SyncInferenceEngine(
        policy, *make_pre_post_processors(policy.config, path), {}, [], "", "cpu", ""
    )
    rng = np.random.default_rng(0)
    features = policy.config.input_features
    frames = [
        {
            name: rng.integers(0, 256, shape, dtype=np.uint8)
            if features[name].type is FeatureType.VISUAL
            else rng.random(shape, dtype=np.float32)
            for name, shape in frame_shapes(features).items()
        }
        for _ in range(TICKS)
    ]
    engine = BundleEngine(bundle)
    draws: list[np.ndarray] = []
    standard_normal = np.random.standard_normal
    monkeypatch.setattr(
        np.random,
        "standard_normal",
        lambda *args, **kwargs: draws.append(standard_normal(*args, **kwargs)) or draws[-1],
    )
    scale = np.asarray(bundle.manifest.action_scale, dtype=np.float32)
    differences = []
    for frame in frames:
        drawn = len(draws)
        actual = engine.get_action(dict(frame))
        if len(draws) == drawn:
            expected = reference.get_action(dict(frame))
        else:
            noise = torch.from_numpy(draws[-1].astype(np.float32))
            with ReplayDraws(noise) as replay:
                expected = reference.get_action(dict(frame))
            assert len(replay.shapes) == len(noise)
        differences.append(float((np.abs(actual - expected.numpy()) / scale).max()))
    return differences


@pytest.mark.parametrize(
    ("name", "backend"),
    [pytest.param(name, backend, marks=BACKENDS[backend]) for name in CHECKPOINTS for backend in BACKENDS],
)
def test_bundle_plays_like_the_pytorch_policy_over_many_ticks(name, backend, tmp_path, monkeypatch):
    path = tmp_path / "checkpoint"
    CHECKPOINTS[name](path)
    use_plugins(monkeypatch, "torch_backend")
    export([f"--backend.type={backend}", f"--policy.path={path}", f"--output_dir={tmp_path / 'b'}"])
    bundle = Bundle.read(tmp_path / "b")
    if backend in GPU_BACKENDS:
        # The GPU draws the noise, which the PyTorch policy cannot replay, so the export's own check of the
        # test case, made with the recorded noise, is the comparison.
        assert bundle.manifest.runs[0] == ProgramRun(NOISE, NOISE, [], [NOISE])
        return
    # A compiled backend sums in another order than PyTorch, so hold it to the tolerance the bundle was checked with.
    assert max(play(path, bundle, monkeypatch)) <= bundle.manifest.tolerance


# In normalized action units: how far a chain of parts drifted from its policy on a Jetson Thor GPU, and ten
# percent more than float32's tolerance.
@pytest.mark.parametrize(
    ("offset", "passes"), [(1.8e-6, True), (2.2e-5, False), pytest.param(float("nan"), False, id="nan")]
)
def test_the_parts_are_held_to_the_export_tolerance(tmp_path, monkeypatch, offset, passes):
    vqbet_checkpoint(tmp_path / "ckpt")
    check = HistoryChunkRecipe._check_chain

    def drifted(self, run):
        scale = torch.as_tensor(self._case.action_scale)
        return check(self, lambda *inputs: run(*inputs) + offset * scale)

    monkeypatch.setattr(HistoryChunkRecipe, "_check_chain", drifted)
    args = [
        "--backend.type=torch_test",
        f"--policy.path={tmp_path / 'ckpt'}",
        f"--output_dir={tmp_path / 'b'}",
    ]
    request = {"step": "trace", "args": args, "folders": str(tmp_path / "folders.json")}
    if passes:
        worker.trace_step(request)
    else:
        with pytest.raises(ValueError, match="compute another chunk"):
            worker.trace_step(request)


class FirstTwoActions(ProcessorStep):
    """Keeps the first two actions, as a postprocessor that crops a padded model width once unnormalized."""

    def __call__(self, transition):
        return {**transition, TransitionKey.ACTION: transition[TransitionKey.ACTION][..., :2]}

    def transform_features(self, features):
        return features


def test_the_action_scale_is_measured_at_the_width_the_policy_predicts(tmp_path):
    from types import SimpleNamespace

    from lerobot.policies.vqbet.export_vqbet import VQBeTExportRecipe
    from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

    vqbet_checkpoint(tmp_path, actions=3)
    policy = VQBeTPolicy.from_pretrained(tmp_path)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, tmp_path)
    # Right after the unnormalizer, which needs all three actions.
    postprocessor.steps.insert(1, FirstTwoActions())
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task="", tolerance=None
    )
    recipe = VQBeTExportRecipe(policy, preprocessor, postprocessor, cfg)
    assert recipe.test_case().action_scale.shape == (2,)


def test_the_parts_are_checked_on_a_window_of_frames_that_all_differ(tmp_path, monkeypatch):
    # The test case's first frame stands in for the ones not seen yet, so only other frames tell its two oldest apart.
    vqbet_checkpoint(tmp_path / "ckpt")
    check = HistoryChunkRecipe._check_chain

    def oldest_two_swapped(self, run):
        def swapped(*inputs):
            *frames, noise = inputs
            return run(*(x[[1, 0, *range(2, len(x))]] for x in frames), noise)

        return check(self, swapped)

    monkeypatch.setattr(HistoryChunkRecipe, "_check_chain", oldest_two_swapped)
    args = [
        "--backend.type=torch_test",
        f"--policy.path={tmp_path / 'ckpt'}",
        f"--output_dir={tmp_path / 'b'}",
    ]
    request = {"step": "trace", "args": args, "folders": str(tmp_path / "folders.json")}
    with pytest.raises(ValueError, match="compute another chunk"):
        worker.trace_step(request)


def test_a_checkpoint_whose_residual_vq_was_never_trained_is_exported_with_a_warning(tmp_path, monkeypatch):
    from lerobot.policies.vqbet import export_vqbet

    # Untrained, as the tiny checkpoint is.
    vqbet_checkpoint(tmp_path / "ckpt")
    warned = []
    # Not `caplog`: the export sets up its own logging, which drops pytest's handler.
    monkeypatch.setattr(export_vqbet.logger, "warning", lambda message, *args: warned.append(message % args))
    args = [
        "--backend.type=torch_test",
        f"--policy.path={tmp_path / 'ckpt'}",
        f"--output_dir={tmp_path / 'b'}",
    ]
    request = {"step": "trace", "args": args, "folders": str(tmp_path / "folders.json")}
    worker.trace_step(request)
    assert any("residual VQ was never trained" in message for message in warned)


def test_a_policy_whose_actions_are_not_finite_is_refused(tmp_path, monkeypatch):
    vqbet_checkpoint(tmp_path / "ckpt")
    reference = HistoryChunkRecipe._history_reference

    def not_finite(self, ticks, noise):
        actions, shapes = reference(self, ticks, noise)
        return actions * np.nan, shapes

    monkeypatch.setattr(HistoryChunkRecipe, "_history_reference", not_finite)
    args = [
        "--backend.type=torch_test",
        f"--policy.path={tmp_path / 'ckpt'}",
        f"--output_dir={tmp_path / 'b'}",
    ]
    request = {"step": "trace", "args": args, "folders": str(tmp_path / "folders.json")}
    with pytest.raises(ValueError, match="computes actions that are not finite"):
        worker.trace_step(request)


def test_the_program_holds_uint8_only_in_its_inputs(tmp_path, monkeypatch):
    """TensorRT holds uint8 tensors only at an engine's inputs. A frame indexed before its cast would leave
    that indexing to ExecuTorch's CUDA backend, outside the TensorRT engine."""
    vqbet_checkpoint(tmp_path / "ckpt")
    use_plugins(monkeypatch, "torch_backend")
    export(
        [
            "--backend.type=torch_test",
            f"--policy.path={tmp_path / 'ckpt'}",
            f"--output_dir={tmp_path / 'b'}",
        ]
    )
    bundle = Bundle.read(tmp_path / "b")
    graph = torch.export.load(bundle.path / bundle.manifest.programs["model"].file).graph
    uint8 = [node.op for node in graph.nodes if getattr(node.meta.get("val"), "dtype", None) == torch.uint8]
    # The camera's frames the program takes in, and no tensor it computes.
    assert uint8 == ["placeholder"]
