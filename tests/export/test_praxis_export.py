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

"""Praxis-1's export recipe: what it refuses to export, its bundle, and the check of its parts."""

import dataclasses
import json
import logging
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.bundle.manifest import NOISE, Bundle
from lerobot.export import recipe as export_recipe, worker
from lerobot.export.backends.executorch_cuda import ExecuTorchCUDAConfig
from lerobot.export.backends.executorch_tensorrt import ExecuTorchTensorRTConfig
from lerobot.policies.praxis import export_praxis
from lerobot.policies.praxis.export_praxis import PraxisExportRecipe
from lerobot.policies.praxis.praxis_core import layers as praxis_layers
from lerobot.policies.praxis.praxis_core.vae import Autoencoder
from lerobot.scripts import lerobot_export
from lerobot.scripts.lerobot_export import export
from lerobot.utils.constants import ACTION
from tests.export import (
    tiny_praxis,
    torch_backend,  # noqa: F401  # registers torch_test for the in-process trace step
)
from tests.export.plugins import use_plugins
from tests.export.torch_backend import TorchTestBackend
from tests.policies.praxis.tiny_model import tiny_lerobot_config


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (tiny_lerobot_config(max_context_chunks=2), "max_context_chunks=1"),
        (tiny_lerobot_config(n_action_steps=12), "n_action_steps=24"),
    ],
)
def test_export_refuses_what_it_does_not_run(monkeypatch, config, message):
    with pytest.raises(ValueError, match=message):
        PraxisExportRecipe(SimpleNamespace(config=config), None, None, None)

    def load_policy(cfg):
        raise AssertionError("loaded the policy")

    monkeypatch.setattr(export_recipe, "load_policy", load_policy)
    with pytest.raises(ValueError, match=message):
        PraxisExportRecipe.load(SimpleNamespace(policy=config))


def use_fake_torch_tensorrt(monkeypatch, *settings, kernels=("triton_op",)):
    """Replace Torch-TensorRT by a release whose compiler has only the settings `settings`, and whose
    `torch_tensorrt.kernels` has only the functions `kernels`."""
    module = types.ModuleType("torch_tensorrt")
    module.__version__ = "2.15.0"
    module.kernels = SimpleNamespace(**{name: lambda *args, **kwargs: None for name in kernels})
    dynamo = types.ModuleType("torch_tensorrt.dynamo")
    dynamo.CompilationSettings = dataclasses.make_dataclass(
        "CompilationSettings", [(s, bool) for s in settings]
    )
    module.dynamo = dynamo
    monkeypatch.setitem(sys.modules, "torch_tensorrt", module)
    monkeypatch.setitem(sys.modules, "torch_tensorrt.dynamo", dynamo)


class LoadedPolicyError(Exception):
    pass


@pytest.mark.parametrize(
    ("backend", "settings", "refusal"),
    [
        (
            ExecuTorchTensorRTConfig(),
            ["workspace_size"],
            (
                ValueError,
                r"Torch-TensorRT 2\.15\.0 has no grouped_mm_kernel .*TensorRT#5021, in Torch-TensorRT 2\.15\.1",
            ),
        ),
        (ExecuTorchTensorRTConfig(), None, (ImportError, r"pip install .*torch_tensorrt\[executorch\]")),
        (ExecuTorchTensorRTConfig(), ["workspace_size", "grouped_mm_kernel"], None),
        (ExecuTorchCUDAConfig(), ["workspace_size"], None),
    ],
    ids=["tensorrt_without_the_setting", "no_tensorrt", "tensorrt_with_the_setting", "cuda"],
)
def test_export_refuses_a_torch_tensorrt_that_leaves_the_experts_to_pytorch(
    monkeypatch, backend, settings, refusal
):
    """Torch-TensorRT ignores a setting it does not have, so the recipe asks before any weight loads."""
    if settings is None:
        monkeypatch.setitem(sys.modules, "torch_tensorrt", None)
    else:
        use_fake_torch_tensorrt(monkeypatch, *settings)

    def load_policy(cfg):
        raise LoadedPolicyError

    monkeypatch.setattr(export_recipe, "load_policy", load_policy)
    cfg = SimpleNamespace(policy=tiny_lerobot_config(), backend=backend)
    error, match = refusal or (LoadedPolicyError, None)
    with pytest.raises(error, match=match):
        PraxisExportRecipe.load(cfg)


@pytest.mark.parametrize("kernels", [("triton_op",), ()], ids=["with_triton_op", "without_triton_op"])
def test_export_says_when_its_torch_tensorrt_cannot_make_the_top_k_sum_plugin(monkeypatch, caplog, kernels):
    """Without it, the export keeps the PyTorch top-k sum, which TensorRT runs slower."""
    use_fake_torch_tensorrt(monkeypatch, "workspace_size", "grouped_mm_kernel", kernels=kernels)

    def load_policy(cfg):
        raise LoadedPolicyError

    monkeypatch.setattr(export_recipe, "load_policy", load_policy)
    cfg = SimpleNamespace(policy=tiny_lerobot_config(), backend=ExecuTorchTensorRTConfig())
    with caplog.at_level(logging.WARNING, logger=export_praxis.__name__), pytest.raises(LoadedPolicyError):
        PraxisExportRecipe.load(cfg)
    warned = "2.15.0 has no torch_tensorrt.kernels.triton_op (pytorch/TensorRT#4435" in caplog.text
    assert warned == (not kernels)


def test_the_compile_options_of_sample_reach_only_its_program(tmp_path, monkeypatch):
    """Through the trace and compile steps of an export, with the test backend standing in for TensorRT."""
    tiny_praxis.save_checkpoint(tmp_path / "checkpoint")
    tiny_praxis.use_tiny_praxis(monkeypatch)
    use_plugins(monkeypatch, "torch_backend")
    compile, seen = TorchTestBackend.compile, {}

    def compile_and_record(self, program, path, **options):
        seen[path.stem] = options
        return compile(self, program, path)

    def run_step_in_this_process(work, name, request):
        # Through JSON, as between processes.
        return json.loads(json.dumps(worker.STEPS[request["step"]](json.loads(json.dumps(request)))))

    monkeypatch.setattr(TorchTestBackend, "compile", compile_and_record)
    monkeypatch.setattr(export_praxis, "ExecuTorchTensorRTConfig", TorchTestBackend)
    monkeypatch.setattr(export_praxis, "_check_torch_tensorrt", lambda backend: None)
    monkeypatch.setattr(lerobot_export, "_run_step", run_step_in_this_process)
    monkeypatch.setattr(worker, "set_log_level", lambda level: None)
    export(
        [
            "--backend.type=torch_test",
            f"--policy.path={tmp_path / 'checkpoint'}",
            f"--output_dir={tmp_path / 'bundle'}",
            f"--task={tiny_praxis.TASK}",
        ]
    )
    moe = {"compile_options": {"grouped_mm_kernel": True}}
    assert seen == {
        name: moe if name == "sample" else {}
        for name in ("encode", "start", "sample", "actions", "actions_first")
    }


@pytest.mark.parametrize("bimanual", [False, True], ids=["single_arm", "bimanual"])
def test_bundle_plays_the_policys_first_chunks(tmp_path, monkeypatch, bimanual):
    """The export replays the bundle's test case, three chunks from a reset."""
    tiny_praxis.save_checkpoint(tmp_path / "checkpoint", bimanual=bimanual)
    use_plugins(monkeypatch, "torch_backend", "tiny_praxis")
    export(
        [
            "--backend.type=torch_test",
            f"--policy.path={tmp_path / 'checkpoint'}",
            f"--output_dir={tmp_path / 'bundle'}",
            f"--task={tiny_praxis.TASK}",
        ]
    )
    manifest = Bundle.read(tmp_path / "bundle").manifest
    assert sorted(manifest.programs) == ["actions", "actions_first", "encode", "sample", "start"]
    assert [run.program for run in manifest.first_runs[0]] == ["encode", "start", "sample", "actions_first"]
    assert [run.program for run in manifest.runs] == ["encode", "start", "sample", "actions"]
    assert set(manifest.history.values()) == {22}
    assert manifest.task == tiny_praxis.TASK


def make_recipe(path, monkeypatch, **overrides) -> PraxisExportRecipe:
    """The recipe for the tiny checkpoint saved into `path`, with `overrides` of its config."""
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.praxis.modeling_praxis import PraxisPolicy

    tiny_praxis.save_checkpoint(path, **overrides)
    tiny_praxis.use_tiny_praxis(monkeypatch)
    config = PreTrainedConfig.from_pretrained(path)
    policy = PraxisPolicy.from_pretrained(path, config=config).eval()
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(path))
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task=tiny_praxis.TASK, tolerance=None
    )
    return PraxisExportRecipe(policy, preprocessor, postprocessor, cfg)


def test_the_test_case_holds_three_chunks_of_actions_and_noise(tmp_path, monkeypatch):
    """The noise's values are checked where the bundle plays the policy's actions with it."""
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    case = recipe.test_case()
    # 21 actions from the first chunk, 24 from the second, and the first of the third.
    assert case.actions.shape == (46, 7)
    assert case.noise is not None and len(case.noise) == export_praxis.CHUNKS
    assert np.isfinite(case.actions).all()


@pytest.mark.parametrize(
    "settings",
    [
        {"cfg_scale": 2.0},
        {"cfg_scale": 2.0, "cfg_video_scale": 1.0},
        {"cfg_video_scale": 1.0},
        {"num_steps": 1},
        {"num_steps": 3},
    ],
    ids=["actions_guided", "only_actions_guided", "no_guidance", "one_step", "three_steps"],
)
def test_parts_play_the_policys_actions_with_other_sampler_settings(tmp_path, monkeypatch, settings):
    """The parts must play the policy's actions over the test case, which `parts` checks before it yields them."""
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch, **settings)
    parts = {part.name: part for part in recipe.parts()}
    assert list(parts) == ["encode", "start", "sample", "actions_first", "actions"]
    assert hasattr(parts["sample"].module, "decoder") == (settings.get("num_steps", 2) > 1)


@pytest.mark.parametrize(
    ("settings", "passes"), [({}, 3), ({"cfg_scale": 2.0}, 4)], ids=["actions_unguided", "actions_guided"]
)
def test_the_last_step_runs_only_the_passes_whose_action_velocities_it_combines(
    tmp_path, monkeypatch, settings, passes
):
    """Two steps with a guided video: the first runs the encoder tower for both texts, and the last, whose video
    moves no action, runs it for the blank caption only when the actions are guided too."""
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch, **settings)
    sample = next(part for part in recipe.parts() if part.name == "sample")
    encoder, calls = sample.module.encoder, []
    forward = encoder.forward_chunk_causal_pair
    monkeypatch.setattr(
        encoder,
        "forward_chunk_causal_pair",
        lambda texts, *args, **kwargs: calls.extend(texts) or forward(texts, *args, **kwargs),
    )
    with torch.no_grad():
        sample.module(*sample.example_inputs)
    assert len(calls) == passes


def test_sample_returns_what_its_decoder_tower_moves(tmp_path, monkeypatch):
    """The actions reach the decoder tower only through the last step's encoder pass, which can hide a broken
    decoder from the check of `sample` on its own, so the program also returns the latents the last step reads."""
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    sample = next(part for part in recipe.parts() if part.name == "sample")
    with torch.no_grad():
        outputs = sample.module(*sample.example_inputs)
        decoder = sample.module.decoder
        forward = decoder.forward_chunk_causal
        monkeypatch.setattr(decoder, "forward_chunk_causal", lambda *args: torch.zeros_like(forward(*args)))
        broken = sample.module(*sample.example_inputs)
    assert sample.outputs[0] == export_praxis.VIDEO
    assert not torch.equal(broken[0], outputs[0])


def test_the_rehearsal_draws_other_noise_than_the_test_case(tmp_path, monkeypatch):
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    drawn = []
    noise = recipe._noise
    monkeypatch.setattr(recipe, "_noise", lambda seed: drawn.append(noise(seed)) or drawn[-1])
    list(recipe.parts())
    case = {chunk.tobytes() for chunk in recipe.test_case().noise}
    assert not any(chunk.tobytes() in case for chunk in drawn[0])


@pytest.mark.parametrize("enabled", [True, False])
def test_the_export_leaves_the_custom_kernels_as_it_found_them(tmp_path, monkeypatch, enabled):
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    monkeypatch.setattr(praxis_layers, "_CUSTOM_KERNELS", enabled)
    list(recipe.parts())
    assert praxis_layers.custom_kernels_enabled() == enabled


def test_encode_holds_only_the_autoencoders_encoder_half(tmp_path, monkeypatch):
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    with torch.device("meta"):
        vae = Autoencoder()
    # As the policy keeps its frozen modules: out of its submodules, so not with `policy.vae = vae`.
    recipe.policy._frozen.vae = vae
    held = {id(parameter) for parameter in export_praxis._Encode(recipe.policy, 21).parameters()}
    assert held == {id(parameter) for module in (vae.encoder, vae.conv1) for parameter in module.parameters()}


def test_encode_tiles_the_frames_as_the_policy_does(tmp_path, monkeypatch):
    """A config without the video's tiling keys tiles at the default size, as the policy's sessions do."""
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    policy = recipe.policy
    policy.config.model_config["video"] = {}
    encode = export_praxis._Encode(policy, 21)
    assert (encode.tile_height, encode.tile_width) == tuple(policy.variant.tile_size)
    assert (encode.aspect_tol, encode.antialias) == (policy.variant.aspect_tol, policy.variant.antialias)


def test_encode_takes_the_anchor_latent_from_the_clean_latents(tmp_path, monkeypatch):
    """The autoencoder encodes the first frame alone, so one encode of the frames also gives the anchor."""
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    case = recipe.test_case()
    vae, clips = recipe.policy.vae, []
    encode_eager = vae.encode_eager
    monkeypatch.setattr(
        vae, "encode_eager", lambda clip, **kwargs: clips.append(clip) or encode_eager(clip, **kwargs)
    )
    kept = recipe._executed + 1
    inputs = [torch.from_numpy(case.frame[name][:kept]) for name in [*recipe._cameras, recipe._state]]
    with torch.no_grad():
        anchor, clean, _ = export_praxis._Encode(recipe.policy, recipe._executed)(*inputs)
        expected = encode_eager(clips[0][:, :, :1])
    assert len(clips) == 1
    assert clean.shape[2] > 1
    assert torch.equal(anchor, expected)


def test_the_bundle_config_leaves_out_a_local_frozen_modules_folder(tmp_path, monkeypatch):
    # The tiny checkpoint has no text encoder to name: a folder there would load a tokenizer step.
    frozen = str(tmp_path / "frozen")
    (tmp_path / "frozen").mkdir()
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch, vae_id=frozen)
    assert recipe.config.vae_id == ""
    assert recipe.policy.config.vae_id == frozen


def test_export_refuses_parts_that_play_other_actions_than_the_policy(tmp_path, monkeypatch):
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    forward = export_praxis._Actions.forward

    def drifted(self, *inputs):
        return forward(self, *inputs) + 1e-3

    monkeypatch.setattr(export_praxis._Actions, "forward", drifted)
    with pytest.raises(ValueError, match="play other actions than the policy itself"):
        list(recipe.parts())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sample_is_checked_on_its_own_in_bfloat16(tmp_path, monkeypatch, dtype):
    """In bfloat16 the test case's chunks amplify rounding so much that it cannot tell a correct program from a
    broken one, so `sample` is checked on its own. No rerun measures the other parts' rounding."""
    parts = list(make_recipe(tmp_path / "checkpoint", monkeypatch, dtype=dtype).parts())
    checked = {part.name for part in parts if part.check is not None}
    assert checked == ({"sample"} if dtype == torch.bfloat16 else set())
    for part in parts:
        if part.check is not None:
            assert part.check.failures(part.name, part.check.outputs) == []
            assert part.check.failures(part.name, [torch.zeros_like(x) for x in part.check.outputs])


@pytest.mark.parametrize(
    "backend", [ExecuTorchTensorRTConfig(), ExecuTorchCUDAConfig()], ids=["tensorrt", "cuda"]
)
def test_the_parts_with_mixture_of_experts_layers_ask_for_tensorrts_grouped_matrix_multiply(
    tmp_path, monkeypatch, backend
):
    """Only with TensorRT, since only `executorch_tensorrt` reads compile options."""
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    recipe.cfg.backend = backend
    parts = list(recipe.parts())
    moe = {
        part.name
        for part in parts
        if any(isinstance(module, praxis_layers.MoEFeedForward) for module in part.module.modules())
    }
    assert moe == {"sample"}
    tensorrt = isinstance(backend, ExecuTorchTensorRTConfig)
    assert {part.name: part.compile_options for part in parts} == {
        part.name: {"grouped_mm_kernel": True} if tensorrt and part.name in moe else {} for part in parts
    }


def test_the_encode_part_encodes_the_executed_chunk_in_one_pass(tmp_path, monkeypatch):
    """Chunk by chunk, TensorRT spends about as long carrying each chunk's causal context as on the convolutions."""
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    encode = next(part for part in recipe.parts() if part.name == "encode")
    encode_eager, calls = recipe.policy.vae.encode_eager, []
    monkeypatch.setattr(
        recipe.policy.vae,
        "encode_eager",
        lambda *args, **kwargs: calls.append(kwargs) or encode_eager(*args, **kwargs),
    )
    with torch.no_grad():
        encode.module(*encode.example_inputs)
    assert calls == [{"streamed": False}]


def test_every_chunk_reads_the_noise_and_the_kept_frames(tmp_path, monkeypatch):
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    first, later = recipe.plan().first_runs[0], recipe.plan().runs
    for runs in (first, later):
        assert runs[1].inputs == [NOISE]
        assert runs[-1].outputs == [ACTION]
        assert runs[0].inputs == [*recipe._cameras, recipe._state]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_with_tensorrt_in_bfloat16_the_parts_trace_the_top_k_sum_as_one_plugin(tmp_path, monkeypatch, dtype):
    """Written in PyTorch, TensorRT fuses the top-k sum with the next block's norm into one slow kernel."""
    sum_rows_of_tokens = praxis_layers._sum_rows_of_tokens
    monkeypatch.setattr(export_praxis, "_top_k_sum_plugin", lambda: sum_rows_of_tokens)
    monkeypatch.setattr(export_praxis, "_take_rows_plugin", lambda: None)
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch, dtype=dtype)
    recipe.cfg.backend = ExecuTorchTensorRTConfig()
    seen = []
    routed = praxis_layers._routed_chunk_group

    def routed_chunk(*args):
        seen.append(praxis_layers._TOP_K_SUM)
        return routed(*args)

    monkeypatch.setattr(praxis_layers, "_routed_chunk_group", routed_chunk)
    parts = list(recipe.parts())
    bfloat16 = dtype == torch.bfloat16
    # The policy's own runs, outside the parts, keep the PyTorch sum.
    assert set(seen) <= {None, sum_rows_of_tokens}
    assert (sum_rows_of_tokens in seen) == bfloat16
    assert praxis_layers._TOP_K_SUM is None
    # The compile step registers the plugin's operator before it loads the program.
    assert {part.name: part.operator_modules for part in parts} == {
        part.name: (export_praxis.TOP_K_SUM_MODULE,) if bfloat16 and part.name == "sample" else ()
        for part in parts
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_with_tensorrt_in_bfloat16_the_parts_trace_the_row_gather_as_one_plugin(tmp_path, monkeypatch, dtype):
    """Written in PyTorch, TensorRT runs the gather of the rows in expert order slower than the plugin."""

    def take_rows(x, index):
        return x[index]

    monkeypatch.setattr(export_praxis, "_take_rows_plugin", lambda: take_rows)
    monkeypatch.setattr(export_praxis, "_top_k_sum_plugin", lambda: None)
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch, dtype=dtype)
    recipe.cfg.backend = ExecuTorchTensorRTConfig()
    seen = []
    routed = praxis_layers._routed_chunk_group

    def routed_chunk(*args):
        seen.append(praxis_layers._TAKE_ROWS)
        return routed(*args)

    monkeypatch.setattr(praxis_layers, "_routed_chunk_group", routed_chunk)
    parts = list(recipe.parts())
    bfloat16 = dtype == torch.bfloat16
    # The policy's own runs, outside the parts, keep the PyTorch gather.
    assert set(seen) <= {None, take_rows}
    assert (take_rows in seen) == bfloat16
    assert praxis_layers._TAKE_ROWS is None
    # The compile step registers the plugin's operator before it loads the program.
    assert {part.name: part.operator_modules for part in parts} == {
        part.name: (export_praxis.TAKE_ROWS_MODULE,) if bfloat16 and part.name == "sample" else ()
        for part in parts
    }


def _need_triton_op():
    """Skip without a Torch-TensorRT that makes TensorRT plugins of Triton kernels; return the module otherwise."""
    torch_tensorrt = pytest.importorskip("torch_tensorrt")
    if getattr(getattr(torch_tensorrt, "kernels", None), "triton_op", None) is None:
        pytest.skip(
            f"Torch-TensorRT {torch_tensorrt.__version__} makes no TensorRT plugin of a Triton kernel"
        )
    return torch_tensorrt


def _assert_runs_in_tensorrt(compiled):
    """The compiled module is TensorRT engines only, and at least one."""
    children = [name for name, _ in compiled.named_children()]
    assert children and all(name.startswith("_run_on_acc") for name in children), children


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the plugin runs on CUDA")
@pytest.mark.parametrize("top_k", [8, 6])
def test_the_top_k_sum_plugin_sums_as_pytorch_does_inside_tensorrt(top_k):
    torch_tensorrt = _need_triton_op()
    from lerobot.policies.praxis.export_top_k_sum import TOP_K_SUM

    assert TOP_K_SUM is not None
    generator = torch.Generator(device="cuda").manual_seed(0)
    # A width that is no multiple of the plugin's block.
    tokens, width = 37, 2100
    rows = torch.randn(tokens * top_k, width, device="cuda", generator=generator).to(torch.bfloat16)
    rows_of_token = torch.randperm(tokens * top_k, device="cuda", generator=generator).view(tokens, top_k)
    scale = torch.rand(tokens, top_k, device="cuda", generator=generator)

    class TopKSum(torch.nn.Module):
        def forward(self, rows, rows_of_token, scale):
            return TOP_K_SUM(rows, rows_of_token, scale)

    inputs = [rows, rows_of_token, scale]
    program = torch.export.export(TopKSum(), tuple(inputs))
    compiled = torch_tensorrt.dynamo.compile(program, inputs=inputs, min_block_size=1, disable_tf32=True)
    _assert_runs_in_tensorrt(compiled)
    expected = praxis_layers._sum_rows_of_tokens(rows, rows_of_token, scale)
    torch.testing.assert_close(compiled(*inputs), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the plugin runs on CUDA")
def test_the_row_gather_plugin_gathers_as_pytorch_does_inside_tensorrt():
    torch_tensorrt = _need_triton_op()
    from lerobot.policies.praxis.export_take_rows import TAKE_ROWS

    assert TAKE_ROWS is not None
    generator = torch.Generator(device="cuda").manual_seed(0)
    # A width that is no multiple of the plugin's block, and each token's row taken top_k times.
    tokens, top_k, width = 37, 8, 2100
    x = torch.randn(tokens, width, device="cuda", generator=generator).to(torch.bfloat16)
    index = torch.arange(tokens, device="cuda").repeat_interleave(top_k)
    index = index[torch.randperm(index.numel(), device="cuda", generator=generator)]

    class TakeRows(torch.nn.Module):
        def forward(self, x, index):
            return TAKE_ROWS(x, index)

    inputs = [x, index]
    program = torch.export.export(TakeRows(), tuple(inputs))
    compiled = torch_tensorrt.dynamo.compile(program, inputs=inputs, min_block_size=1)
    _assert_runs_in_tensorrt(compiled)
    assert torch.equal(compiled(*inputs), x[index])
