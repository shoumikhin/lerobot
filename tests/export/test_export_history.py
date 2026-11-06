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

"""Bundles played next to their PyTorch policies, tick by tick, end to end on tiny checkpoints.

Each bundle and its policy run in `lerobot-rollout`'s engines on the same frames and the same random draws,
for many chunks, so what the bundle carries from frame to frame (past frames, a warm-started plan or
caches), the programs it runs for each chunk and its noise are all exercised.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

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
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_STATE
from lerobot.utils.import_utils import is_package_available
from tests.export import (
    tiny_lingbot_va,
    tiny_longwam,
    tiny_multi_task_dit,
    tiny_pi0_fast,
    torch_backend,  # noqa: F401  # registers torch_test for the in-process trace step
)
from tests.export.backends import BACKENDS as EXECUTORCH_BACKENDS, GPU_BACKENDS
from tests.export.plugins import use_plugins

IMAGE = "observation.images.front"
# Big enough for a 3x3 last ResNet feature map: on a 1x1 map SpatialSoftmax has one cell, so the camera
# would not change the actions.
FRAME = (96, 96)
CROP = (84, 84)
TASK = "pick up the cube"
BACKENDS = {"torch_test": [], **{name: [mark] for name, mark in EXECUTORCH_BACKENDS.items()}}


def _min_max(size: int) -> dict[str, torch.Tensor]:
    return {"min": torch.full((size,), -2.0), "max": torch.full((size,), 3.0)}


def vqbet_checkpoint(path, actions: int = 2, **changes) -> None:
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
        **changes,
    )
    stats = {OBS_STATE: _min_max(2), ACTION: _min_max(actions)}
    _save(config, stats, path)


def diffusion_checkpoint(path, **sampler) -> None:
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    config = DiffusionConfig(
        input_features={
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *FRAME)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        n_obs_steps=3,
        horizon=8,
        n_action_steps=4,
        drop_n_last_frames=2,
        crop_shape=CROP,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=8,
        down_dims=(16, 32),
        kernel_size=3,
        n_groups=4,
        diffusion_step_embed_dim=16,
        num_train_timesteps=10,
        device="cpu",
        **sampler,
    )
    stats = {
        IMAGE: {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.2)},
        OBS_STATE: _min_max(2),
        ACTION: _min_max(2),
    }
    _save(config, stats, path)


def multi_task_dit_checkpoint(path, **sampler) -> None:
    from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig

    config = MultiTaskDiTConfig(
        input_features={
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *FRAME)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        # Three, so the test case of two ticks also has a frame the first one stands in for.
        n_obs_steps=3,
        horizon=8,
        n_action_steps=4,
        num_train_timesteps=10,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        timestep_embed_dim=16,
        # The tiny CLIP takes square images of its own size; the policy resizes the camera's.
        image_resize_shape=(tiny_multi_task_dit.IMAGE_SIZE, tiny_multi_task_dit.IMAGE_SIZE),
        image_crop_shape=None,
        tokenizer_max_length=8,
        device="cpu",
        **sampler,
    )
    stats = {
        IMAGE: {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.2)},
        OBS_STATE: _min_max(2),
        ACTION: _min_max(2),
    }
    with pytest.MonkeyPatch.context() as patch:
        tiny_multi_task_dit.use_tiny_clip(patch)
        _save(config, stats, path)


def tdmpc_checkpoint(path, features: dict[str, PolicyFeature], **changes) -> None:
    from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig

    config = TDMPCConfig(
        input_features=features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        image_encoder_hidden_dim=8,
        state_encoder_hidden_dim=16,
        latent_dim=8,
        mlp_dim=16,
        horizon=3,
        cem_iterations=2,
        n_gaussian_samples=16,
        n_pi_samples=4,
        n_elites=4,
        # More than two, so the planner also draws which of them value the end of each trajectory.
        q_ensemble_size=3,
        device="cpu",
        **changes,
    )
    stats = {ACTION: _min_max(2)}
    _save(config, stats, path)


def _save(config, stats, path) -> None:
    torch.manual_seed(0)
    policy = get_policy_class(config.type)(config)
    if config.type == "multi_task_dit":
        # Untrained, its adaLN gates are zero, so the denoiser would ignore the frames, the task and the timestep.
        for name, parameter in policy.named_parameters():
            if "adaLN_modulation" in name:
                torch.nn.init.normal_(parameter, std=0.1)
    if config.type == "tdmpc":
        # Untrained, the reward and value heads are zero, so every trajectory would score the same.
        for head in [policy.model._reward, *policy.model._Qs]:
            torch.nn.init.normal_(head[-1].weight)
    policy.save_pretrained(path)
    for processor in make_pre_post_processors(config, dataset_stats=stats):
        processor.save_pretrained(path)


def pi0_fast_checkpoint(path) -> None:
    with pytest.MonkeyPatch.context() as patch:
        tiny_pi0_fast.save_checkpoint(path, patch)


def some_answers_hold_fast_tokens(policy: torch.nn.Module, bundle: Bundle, played: np.ndarray) -> None:
    from lerobot.policies.pi0_fast.export_pi0_fast import cleans_up_spaces

    chunks = played.reshape(-1, policy.config.n_action_steps, played.shape[-1])
    # Some chunks' answers hold FAST tokens, so the comparison covers the bundle's decoding too.
    assert len({chunk.tobytes() for chunk in chunks if np.abs(chunk).max() > 0.1}) >= 2
    # What the policy's tokenizer does when it decodes, which its clean-up setting alone does not say.
    assert bundle.manifest.action_tokens.clean_up_spaces == cleans_up_spaces(
        policy.action_tokenizer.bpe_tokenizer
    )
    assert policy.action_tokenizer.bpe_tokenizer.clean_up_tokenization_spaces


def lingbot_va_programs_hold_one_copy_of_their_weights(
    policy: torch.nn.Module, bundle: Bundle, played: np.ndarray
) -> None:
    if bundle.manifest.backend != "torch_test":
        # The weights are read from the test backend's PyTorch programs.
        return
    config = policy.config
    policy._ensure_frozen_modules()
    weight = policy.transformer.blocks[0].attn1.to_q.weight
    parameters = [*policy.parameters(), *policy._vae.parameters()]
    sizes, holders = {}, []
    for name, files in bundle.manifest.programs.items():
        program = torch.export.load(bundle.path / files.file)
        weights = list(program.state_dict.values())
        sizes[name] = sum(t.numel() for t in weights)
        holders += [name for t in weights if t.shape == weight.shape and torch.equal(t, weight)]
        # A tensor a program holds as a plain attribute is a constant, outside its state dict: none is a weight.
        for t in program.constants.values():
            assert not any(
                isinstance(t, torch.Tensor) and t.shape == p.shape and torch.equal(t, p) for p in parameters
            ), name
    assert holders == ["blocks"]
    # The blocks and the projected text of the task and of the negative prompt.
    projected_text = 2 * config.max_sequence_length * config.num_attention_heads * config.attention_head_dim
    assert sizes["blocks"] == sum(p.numel() for p in policy.transformer.blocks.parameters()) + projected_text
    # The VAE's encoder, without its decoder.
    encoder = sum(p.numel() for m in (policy._vae.encoder, policy._vae.quant_conv) for p in m.parameters())
    assert sizes["encode_first"] == sizes["encode_start"] == sizes["encode_next"] == encoder


def longwam_program_holds_no_weight_it_does_not_run(
    policy: torch.nn.Module, bundle: Bundle, played: np.ndarray
) -> None:
    if bundle.manifest.backend != "torch_test":
        # The weights are read from the test backend's PyTorch programs.
        return
    (files,) = bundle.manifest.programs.values()
    program = torch.export.load(bundle.path / files.file)
    # A tensor a program holds as a plain attribute is a constant, outside its state dict.
    held = [
        *program.state_dict.values(),
        *(t for t in program.constants.values() if isinstance(t, torch.Tensor)),
    ]
    run = ("dit", "vae.encoder", "vae.quant_conv", "proprio_encoder")
    ran = {id(p) for name in run for p in policy.get_submodule(name).parameters()}
    # A weight that holds one value, such as a norm's, matches any other of its shape.
    varied = [p for p in policy.parameters() if p.unique().numel() > 1]
    unused = [p for p in varied if id(p) not in ran]
    # The text encoder, the VAE's decoder and its post_quant_conv.
    assert unused
    # A weight it runs is found where the others are looked for.
    assert any(torch.equal(t, p) for p in varied if id(p) in ran for t in held if t.shape == p.shape)
    for p in unused:
        assert not any(t.shape == p.shape and torch.equal(t, p) for t in held)


@dataclass
class Checkpoint:
    """A tiny checkpoint that `save` writes into a folder, and how its bundle plays.

    The bundle is exported for `task` with each of `backends`, and plays `ticks` frames, reset with the
    policy at the tick `reset_at`. `plugins` make the policy tiny in the export's own processes, as `tiny` does
    in this one. `programs` are the bundle's program names, if checked, and `check` checks what else it holds
    or plays.
    """

    save: Callable[[Path], object]
    backends: tuple[str, ...] = ("torch_test", "executorch_xnnpack", "executorch_tensorrt")
    task: str = TASK
    ticks: int = 24
    reset_at: int | None = None
    plugins: tuple[str, ...] = ()
    tiny: Callable[[pytest.MonkeyPatch], None] | None = None
    programs: list[str] | None = None
    check: Callable[[torch.nn.Module, Bundle, np.ndarray], None] | None = None
    marks: tuple[pytest.MarkDecorator, ...] = ()


# Diffusion Policy and Multi-Task DiT sample with the schedulers of diffusers, an optional package.
DIFFUSERS = pytest.mark.skipif(not is_package_available("diffusers"), reason="diffusers not installed")
# Multi-Task DiT also encodes the task and the images with CLIP from transformers.
TRANSFORMERS = pytest.mark.skipif(
    not is_package_available("transformers"), reason="transformers not installed"
)
# pi0-FAST decodes its actions with a discrete cosine transform from scipy.
SCIPY = pytest.mark.skipif(not is_package_available("scipy"), reason="scipy not installed")
# TD-MPC's image encoder needs a square image.
TDMPC_CAMERA = {
    OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
    OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
}
TDMPC_KEYPOINTS = {
    OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
    OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
}
# Mid-chunk, so the reset drops the queued actions, and with MPC the bundle and the policy both plan again
# from nothing.
TDMPC_RESET = 13
CHECKPOINTS = {
    "vqbet": Checkpoint(vqbet_checkpoint),
    # Two code draws from one noise input, the second picked from the first.
    "vqbet_sequential": Checkpoint(lambda path: vqbet_checkpoint(path, sequentially_select=True)),
    "diffusion": Checkpoint(
        diffusion_checkpoint, programs=["actions", "encoder", "step"], marks=(DIFFUSERS,)
    ),
    "diffusion_ddim": Checkpoint(
        lambda path: diffusion_checkpoint(
            path, noise_scheduler_type="DDIM", prediction_type="sample", num_inference_steps=5
        ),
        programs=["actions", "encoder", "step"],
        marks=(DIFFUSERS,),
    ),
    "multi_task_dit_flow": Checkpoint(
        lambda path: multi_task_dit_checkpoint(path, objective="flow_matching", num_integration_steps=5),
        plugins=("tiny_multi_task_dit",),
        tiny=tiny_multi_task_dit.use_tiny_clip,
        marks=(DIFFUSERS, TRANSFORMERS),
    ),
    # The defaults: the first action of each plan, repeated.
    "tdmpc_repeats": Checkpoint(
        lambda path: tdmpc_checkpoint(path, TDMPC_CAMERA),
        ("torch_test", "executorch_xnnpack", *GPU_BACKENDS),
        reset_at=TDMPC_RESET,
    ),
    "tdmpc_steps": Checkpoint(
        lambda path: tdmpc_checkpoint(path, TDMPC_KEYPOINTS, n_action_repeats=1, n_action_steps=2),
        ("torch_test", "executorch_xnnpack"),
        reset_at=TDMPC_RESET,
    ),
    "tdmpc_policy_only": Checkpoint(
        lambda path: tdmpc_checkpoint(path, TDMPC_CAMERA, use_mpc=False),
        ("torch_test", "executorch_xnnpack"),
        reset_at=TDMPC_RESET,
    ),
    "pi0_fast": Checkpoint(
        pi0_fast_checkpoint,
        ("torch_test",),
        plugins=("tiny_pi0_fast",),
        tiny=tiny_pi0_fast.use_tiny_pi0_fast,
        programs=["actions", "prefill", "step"],
        check=some_answers_hold_fast_tokens,
        marks=(TRANSFORMERS, SCIPY),
    ),
    # Past the KV cache's first eviction, then a reset and two more chunks.
    "lingbot_va": Checkpoint(
        tiny_lingbot_va.save_checkpoint,
        ("torch_test", "executorch_xnnpack"),
        task=tiny_lingbot_va.TASK,
        ticks=48,
        reset_at=36,
        check=lingbot_va_programs_hold_one_copy_of_their_weights,
        marks=(DIFFUSERS, TRANSFORMERS),
    ),
    # Past the 17 frames it keeps, then a reset mid-chunk.
    "longwam": Checkpoint(
        tiny_longwam.save_checkpoint,
        ("torch_test", "executorch_xnnpack"),
        task=tiny_longwam.TASK,
        ticks=40,
        reset_at=25,
        check=longwam_program_holds_no_weight_it_does_not_run,
        programs=["model"],
        marks=(DIFFUSERS, TRANSFORMERS),
    ),
    # Long-WAM's other denoising, which denoises the future video beside the actions.
    "longwam_cod": Checkpoint(
        lambda path: tiny_longwam.save_checkpoint(path, denoising="cod", imagine_sigma=0.0),
        ("torch_test", "executorch_xnnpack"),
        task=tiny_longwam.TASK,
        ticks=40,
        reset_at=25,
        programs=["model"],
        marks=(DIFFUSERS, TRANSFORMERS),
    ),
}


def play(
    checkpoint: Checkpoint, path: Path, bundle: Bundle, monkeypatch
) -> tuple[torch.nn.Module, np.ndarray, list[float]]:
    """The PyTorch policy, the bundle's action on each tick, and the largest difference on each tick between
    the bundle and the policy, on the same draws, in normalized action units."""
    if checkpoint.tiny is not None:
        checkpoint.tiny(monkeypatch)
    policy = get_policy_class(PreTrainedConfig.from_pretrained(path).type).from_pretrained(path)
    reference = SyncInferenceEngine(
        policy, *make_pre_post_processors(policy.config, path), {}, [], bundle.manifest.task or "", "cpu", ""
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
        for _ in range(checkpoint.ticks)
    ]
    with ReplayDraws() as recorded:
        reference.get_action(dict(frames[0]))
    reference.reset()
    shapes = recorded.shapes
    engine = BundleEngine(bundle)
    draws: list[np.ndarray] = []
    # Seeded, so every run replays the same draws: the TD-MPC planner picks among candidates, and on another
    # draw two of them could score close enough for rounding to pick another one.
    standard_normal = np.random.default_rng(1).standard_normal
    monkeypatch.setattr(
        np.random,
        "standard_normal",
        lambda *args, **kwargs: draws.append(standard_normal(*args, **kwargs)) or draws[-1],
    )
    scale = np.asarray(bundle.manifest.action_scale, dtype=np.float32)
    played, differences = [], []
    for tick, frame in enumerate(frames):
        if tick == checkpoint.reset_at:
            engine.reset()
            reference.reset()
        drawn = len(draws)
        actual = engine.get_action(dict(frame))
        if len(draws) == drawn:
            expected = reference.get_action(dict(frame))
        else:
            noise = torch.from_numpy(draws[-1].astype(np.float32)).flatten()
            pieces = noise.split([math.prod(shape) for shape in shapes])
            with ReplayDraws(
                [piece.view(shape) for piece, shape in zip(pieces, shapes, strict=True)]
            ) as replay:
                expected = reference.get_action(dict(frame))
            assert len(replay.shapes) == len(shapes)
        played.append(actual)
        differences.append(float((np.abs(actual - expected.numpy()) / scale).max()))
    return policy, np.stack(played), differences


@pytest.mark.parametrize(
    ("name", "backend"),
    [
        pytest.param(name, backend, marks=[*checkpoint.marks, *BACKENDS[backend]])
        for name, checkpoint in CHECKPOINTS.items()
        for backend in checkpoint.backends
    ],
)
def test_bundle_plays_like_the_pytorch_policy_over_many_ticks(name, backend, tmp_path, monkeypatch):
    checkpoint, path = CHECKPOINTS[name], tmp_path / "checkpoint"
    checkpoint.save(path)
    use_plugins(monkeypatch, "torch_backend", *checkpoint.plugins)
    export(
        [
            f"--backend.type={backend}",
            f"--policy.path={path}",
            f"--output_dir={tmp_path / 'b'}",
            f"--task={checkpoint.task}",
        ]
    )
    bundle = Bundle.read(tmp_path / "b")
    if backend in GPU_BACKENDS:
        # The GPU draws the noise, which the PyTorch policy cannot replay. Without that program the bundle reads
        # the noise the host draws, which `play` replays, and its other programs run on the GPU as compiled.
        manifest = bundle.manifest
        assert manifest.runs[0] == ProgramRun(NOISE, NOISE, [], [NOISE])
        programs = {name: files for name, files in manifest.programs.items() if name != NOISE}
        bundle = Bundle(bundle.path, replace(manifest, programs=programs, runs=manifest.runs[1:]))
    if checkpoint.programs is not None:
        assert sorted(bundle.manifest.programs) == checkpoint.programs
    policy, played, differences = play(checkpoint, path, bundle, monkeypatch)
    # A compiled backend sums in another order than PyTorch, so hold it to the tolerance the bundle was checked with.
    # Not `max`, which skips a NaN that does not come first.
    assert np.all(np.array(differences) <= bundle.manifest.tolerance), differences
    if checkpoint.check is not None:
        checkpoint.check(policy, bundle, played)


@DIFFUSERS
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("kind", ["DDPMScheduler", "DDIMScheduler"])
def test_a_schedule_reads_a_scheduler_built_on_the_gpu(kind):
    import diffusers

    from lerobot.policies.common.export_denoising import scheduler_schedule

    schedules = []
    # An export builds the policy, and so its scheduler, on its device.
    for device in ("cpu", "cuda"):
        with torch.device(device):
            scheduler = getattr(diffusers, kind)(num_train_timesteps=10)
        scheduler.set_timesteps(10)
        schedules.append(scheduler_schedule(scheduler))
    # `add_noise` moves only `alphas_cumprod` to the samples' device, so the scheduler's other tensors stay on the GPU.
    sample = torch.zeros(1)
    scheduler.add_noise(sample, sample, torch.zeros(1, dtype=torch.long))
    schedules.append(scheduler_schedule(scheduler))
    cpu, gpu, mixed = schedules
    np.testing.assert_allclose(np.stack(gpu.coefficients), np.stack(cpu.coefficients), rtol=1e-6)
    np.testing.assert_allclose(np.stack(mixed.coefficients), np.stack(cpu.coefficients), rtol=1e-6)


@DIFFUSERS
@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"variance_type": "fixed_large"}, "fixed_small"),
        ({"thresholding": True}, "thresholding True"),
        ({"prediction_type": "v_prediction"}, "'v_prediction'"),
    ],
)
def test_a_schedule_the_step_cannot_compute_is_refused(change, error):
    from diffusers import DDPMScheduler

    from lerobot.policies.common.export_denoising import scheduler_schedule

    scheduler = DDPMScheduler(num_train_timesteps=10, **change)
    scheduler.set_timesteps(10)
    with pytest.raises(ValueError, match=error):
        scheduler_schedule(scheduler)


def test_a_schedule_without_diffusers_asks_for_its_extra(monkeypatch):
    from lerobot.policies.common.export_denoising import scheduler_schedule
    from lerobot.utils import import_utils

    monkeypatch.setitem(import_utils._require_package_cache, "diffusers", False)
    with pytest.raises(ImportError, match=r"lerobot\[diffusion\]"):
        scheduler_schedule(None)


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


def test_a_long_wam_that_draws_new_noise_for_each_chunk_is_refused_before_its_weights_load():
    from lerobot.policies.longwam.configuration_longwam import LongWAMConfig
    from lerobot.policies.longwam.export_longwam import LongWAMExportRecipe

    with pytest.raises(ValueError, match="inference_seed is null"):
        LongWAMExportRecipe.load(SimpleNamespace(policy=LongWAMConfig(inference_seed=None)))
