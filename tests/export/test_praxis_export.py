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
from lerobot.policies.praxis import export_autoencoder_kernels, export_praxis
from lerobot.policies.praxis.export_praxis import PraxisExportRecipe
from lerobot.policies.praxis.praxis_core import layers as praxis_layers, vae as praxis_vae
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
    cfg = SimpleNamespace(policy=tiny_lerobot_config(), backend=backend, quantize=None)
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
    cfg = SimpleNamespace(policy=tiny_lerobot_config(), backend=ExecuTorchTensorRTConfig(), quantize=None)
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
        backend=SimpleNamespace(device="cpu"),
        seed=0,
        dataset=None,
        task=tiny_praxis.TASK,
        tolerance=None,
        quantize=None,
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
        monkeypatch.setattr(
            decoder,
            "forward_chunk_causal",
            lambda *args, **kwargs: torch.zeros_like(forward(*args, **kwargs)),
        )
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
    monkeypatch.setattr(export_praxis, "_autoencoder_kernel_plugins", lambda: None)
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
    monkeypatch.setattr(export_praxis, "_autoencoder_kernel_plugins", lambda: None)
    monkeypatch.setattr(export_praxis, "_expert_gate_up_plugin", lambda: None)
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_with_tensorrt_in_bfloat16_the_parts_trace_the_expert_gate_up_and_swiglu_as_one_plugin(
    tmp_path, monkeypatch, dtype
):
    """As a gather, a grouped GEMM and a SwiGLU, TensorRT writes the gathered rows and both products to memory."""

    def expert_gate_up(x, index, bank, offsets):
        gate, up = torch._grouped_mm(x[index], bank, offsets).chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

    def take_rows(x, index):
        return x[index]

    monkeypatch.setattr(export_praxis, "_expert_gate_up_plugin", lambda: expert_gate_up)
    monkeypatch.setattr(export_praxis, "_take_rows_plugin", lambda: take_rows)
    monkeypatch.setattr(export_praxis, "_top_k_sum_plugin", lambda: None)
    monkeypatch.setattr(export_praxis, "_autoencoder_kernel_plugins", lambda: None)
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch, dtype=dtype)
    recipe.cfg.backend = ExecuTorchTensorRTConfig()
    seen = []
    routed = praxis_layers._routed_chunk_group

    def routed_chunk(*args):
        seen.append(praxis_layers._EXPERT_GATE_UP)
        return routed(*args)

    monkeypatch.setattr(praxis_layers, "_routed_chunk_group", routed_chunk)
    parts = list(recipe.parts())
    bfloat16 = dtype == torch.bfloat16
    assert set(seen) <= {None, expert_gate_up}
    assert (expert_gate_up in seen) == bfloat16
    assert praxis_layers._EXPERT_GATE_UP is None
    modules = (export_praxis.TAKE_ROWS_MODULE, export_praxis.EXPERT_GATE_UP_MODULE)
    assert {part.name: part.operator_modules for part in parts} == {
        part.name: modules if bfloat16 and part.name in export_praxis.MOE_PARTS else () for part in parts
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the plugin runs on CUDA")
def test_the_expert_gate_up_plugin_computes_as_pytorch_does_inside_tensorrt():
    torch_tensorrt = _need_triton_op()
    from lerobot.policies.praxis.export_expert_gate_up import EXPERT_GATE_UP, _expert_gate_up_eager

    if EXPERT_GATE_UP is None:
        pytest.skip("the plugin needs tcgen05 tensor cores (sm_100 or sm_110)")
    generator = torch.Generator(device="cuda").manual_seed(0)
    # Experts of several row tiles and columns of two tiles, none of them full, an expert without rows, and each token
    # taken top_k times.
    tokens, top_k, experts, dim, hidden = 400, 2, 6, 96, 208
    x = torch.randn(tokens, dim, device="cuda", generator=generator).to(torch.bfloat16)
    bank = torch.randn(experts, dim, 2 * hidden, device="cuda", generator=generator) / dim**0.5
    bank = bank.to(torch.bfloat16)
    chosen = torch.randint(0, experts - 1, (tokens * top_k,), device="cuda", generator=generator)
    chosen[chosen == 3] = experts - 1
    _, order = chosen.sort(stable=True)
    index = torch.arange(tokens, device="cuda").repeat_interleave(top_k)[order]
    offsets = torch.bincount(chosen, minlength=experts).cumsum(0).to(torch.int32)

    class ExpertGateUp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bank = torch.nn.Parameter(bank, requires_grad=False)

        def forward(self, x, index, offsets):
            return EXPERT_GATE_UP(x, index, self.bank, offsets)

    inputs = [x, index, offsets]
    program = torch.export.export(ExpertGateUp(), tuple(inputs))
    compiled = torch_tensorrt.dynamo.compile(program, inputs=inputs, min_block_size=1)
    _assert_runs_in_tensorrt(compiled)
    expected = _expert_gate_up_eager(x, index, bank.unflatten(2, (2, hidden)), offsets)
    # The order of the products' float32 sums can move a bfloat16 rounding, and near zero that has no relative bound.
    torch.testing.assert_close(compiled(*inputs), expected, atol=1e-2, rtol=2**-6)


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


def _refuse_loading(monkeypatch):
    """Make the recipe's policy load fail, so a test sees whether a refusal comes before it."""

    def load_policy(cfg):
        raise LoadedPolicyError

    monkeypatch.setattr(export_recipe, "load_policy", load_policy)


def test_float8_experts_off_tensorrt_need_the_backend_on_a_hopper_gpu(monkeypatch):
    """A Hopper GPU in the machine does not help a backend that runs on the CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (9, 0))
    _refuse_loading(monkeypatch)
    cfg = SimpleNamespace(policy=tiny_lerobot_config(), backend=SimpleNamespace(device="cpu"), quantize="fp8")
    with pytest.raises(ValueError, match="Hopper GPU"):
        PraxisExportRecipe.load(cfg)


@pytest.mark.parametrize("quantize", ["fp8"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_quantized_experts_in_tensorrt_need_a_bfloat16_policy(monkeypatch, quantize, dtype):
    """The plugins take bfloat16 rows, and the experts' PyTorch fallback does not build in TensorRT."""
    _refuse_loading(monkeypatch)
    cfg = SimpleNamespace(
        policy=tiny_lerobot_config(dtype=dtype), backend=ExecuTorchTensorRTConfig(), quantize=quantize
    )
    with pytest.raises(ValueError, match="needs a bfloat16 policy"):
        PraxisExportRecipe.load(cfg)


def test_quantized_experts_in_tensorrt_take_the_checkpoint_dtype_when_none_is_set(monkeypatch):
    """dtype None keeps the checkpoint's own, so the recipe goes on to load it."""
    _refuse_loading(monkeypatch)
    monkeypatch.setattr(export_praxis, "_check_torch_tensorrt", lambda backend: None)
    cfg = SimpleNamespace(
        policy=tiny_lerobot_config(dtype=None), backend=ExecuTorchTensorRTConfig(), quantize="fp8"
    )
    with pytest.raises(LoadedPolicyError):
        PraxisExportRecipe.load(cfg)


@pytest.mark.parametrize("quantize", ["fp8"])
def test_quantized_experts_in_tensorrt_refuse_a_loaded_policy_that_is_not_bfloat16(
    tmp_path, monkeypatch, quantize
):
    """A policy that keeps its checkpoint's dtype is checked once it is loaded, before the plugins are looked up."""
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch, dtype=torch.float32)
    recipe.cfg.backend = ExecuTorchTensorRTConfig()
    recipe.cfg.quantize = quantize
    with pytest.raises(ValueError, match="needs a bfloat16 policy, not torch.float32"):
        next(recipe.parts())


def float8_recipe(path, monkeypatch, grouped_mm) -> PraxisExportRecipe:
    """The bfloat16 recipe of the tiny checkpoint for TensorRT with `--quantize=fp8`, with `grouped_mm` as the
    plugin of the float8 experts' grouped GEMMs and their rows rounded in PyTorch."""
    # Whatever the host's GPU and Torch-TensorRT build, only the float8 plugins of the test are set.
    monkeypatch.setattr(export_praxis, "_top_k_sum_plugin", lambda: None)
    monkeypatch.setattr(export_praxis, "_expert_gate_up_plugin", lambda: None)
    monkeypatch.setattr(export_praxis, "_take_rows_plugin", lambda: None)
    monkeypatch.setattr(export_praxis, "_autoencoder_kernel_plugins", lambda: None)
    monkeypatch.setattr(export_praxis, "_float8_rows_plugin", lambda: praxis_layers._float8_rows)
    monkeypatch.setattr(export_praxis, "_float8_grouped_mm_plugin", lambda: grouped_mm)
    monkeypatch.setattr(export_praxis, "_float8_gate_up_plugin", lambda: None)
    recipe = make_recipe(path, monkeypatch, dtype=torch.bfloat16)
    recipe.cfg.backend = ExecuTorchTensorRTConfig()
    recipe.cfg.quantize = "fp8"
    return recipe


def test_with_tensorrt_the_float8_experts_trace_their_rounding_and_grouped_gemms_as_plugins(
    tmp_path, monkeypatch
):
    """PyTorch's float8 grouped GEMM runs only on sm_90 and sm_100, and TensorRT has no layer for it elsewhere, so
    the parts give each one to TensorRT as a plugin, with the float8 bank as it is stored."""
    banks = []

    def grouped_mm(rows, rows_scale, data, scale, offsets):
        banks.append(data)
        return praxis_layers._float8_grouped_mm_pytorch(
            rows, rows_scale, data, scale, offsets, torch.bfloat16
        )

    recipe = float8_recipe(tmp_path / "checkpoint", monkeypatch, grouped_mm)
    parts = list(recipe.parts())
    assert banks and all(data.dtype == torch.float8_e4m3fn and data.is_contiguous() for data in banks)
    assert praxis_layers._FLOAT8_GROUPED_MM is None and praxis_layers._FLOAT8_ROWS is None
    # The compile step registers the plugins' operators before it loads the program.
    modules = (export_praxis.FLOAT8_ROWS_MODULE, export_praxis.FLOAT8_GROUPED_MM_MODULE)
    assert {part.name: part.operator_modules for part in parts} == {
        part.name: modules if part.name in export_praxis.MOE_PARTS else () for part in parts
    }


def test_with_tensorrt_the_float8_experts_trace_their_gather_gate_up_and_swiglu_as_one_plugin(
    tmp_path, monkeypatch
):
    """As a gather, a grouped GEMM and a SwiGLU, TensorRT writes the float8 rows and both products to memory."""
    seen = []

    def gate_up(tokens, tokens_scale, index, data, scale, offsets):
        seen.append(
            (tokens.dtype, data.dtype, index.shape[0] % tokens.shape[0], index.shape[0] // tokens.shape[0])
        )
        rows, rows_scale = tokens[index], tokens_scale[index]
        out = praxis_layers._float8_grouped_mm_pytorch(rows, rows_scale, data, scale, offsets, torch.bfloat16)
        gate, up = out.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

    def grouped_mm(rows, rows_scale, data, scale, offsets):
        return praxis_layers._float8_grouped_mm_pytorch(
            rows, rows_scale, data, scale, offsets, torch.bfloat16
        )

    recipe = float8_recipe(tmp_path / "checkpoint", monkeypatch, grouped_mm)
    monkeypatch.setattr(export_praxis, "_float8_gate_up_plugin", lambda: gate_up)
    parts = list(recipe.parts())
    # Each token's float8 row is rounded once and gathered once for each of its experts.
    assert seen and all(
        entry[:3] == (torch.float8_e4m3fn, torch.float8_e4m3fn, 0) and entry[3] > 1 for entry in seen
    )
    assert praxis_layers._FLOAT8_GATE_UP is None
    modules = (
        export_praxis.FLOAT8_ROWS_MODULE,
        export_praxis.FLOAT8_GROUPED_MM_MODULE,
        export_praxis.FLOAT8_GATE_UP_MODULE,
    )
    assert {part.name: part.operator_modules for part in parts} == {
        part.name: modules if part.name in export_praxis.MOE_PARTS else () for part in parts
    }


def test_with_tensorrt_the_float8_experts_need_the_grouped_gemm_plugin(tmp_path, monkeypatch):
    """Without it, each grouped GEMM would stay in PyTorch and split its program."""
    recipe = float8_recipe(tmp_path / "checkpoint", monkeypatch, None)
    with pytest.raises(ValueError, match="needs the float8 grouped GEMM plugin"):
        next(recipe.parts())


def test_float8_experts_outside_tensorrt_need_a_hopper_gpu(monkeypatch):
    """Other backends trace PyTorch's float8 grouped GEMM, which runs on sm_90, so the recipe refuses them before
    any weight loads."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def load_policy(cfg):
        raise LoadedPolicyError

    monkeypatch.setattr(export_recipe, "load_policy", load_policy)
    cfg = SimpleNamespace(policy=tiny_lerobot_config(), backend=ExecuTorchCUDAConfig(), quantize="fp8")
    with pytest.raises(ValueError, match="Hopper GPU"):
        PraxisExportRecipe.load(cfg)


def test_the_float8_grouped_gemm_plugin_takes_the_deepest_pipeline_that_fits(monkeypatch):
    """With tcgen05 (sm_100, sm_110) 3 stages need more shared memory than TensorRT gives a plugin kernel, and the
    kernel takes the tiles measured fastest there."""
    pytest.importorskip("triton")
    from lerobot.policies.praxis import export_float8_grouped_mm

    stages, tiles = [], []

    def triton_op(name, kernel, signature, constexprs, *, num_stages, **kwargs):
        stages.append(num_stages)
        tiles.append((constexprs["BLOCK_M"], constexprs["BLOCK_N"], constexprs["BLOCK_K"]))
        if num_stages > 2:
            raise ValueError("triton_op needs 49168 bytes of dynamic shared memory")

    def unsupported(*args, **kwargs):
        raise RuntimeError("Triton kernel requires unsupported launch feature(s): tmem_size=128")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (11, 0))
    monkeypatch.setattr(export_float8_grouped_mm, "_triton_op", lambda: triton_op)
    assert export_float8_grouped_mm._register() is not None
    assert stages == [3, 2] and set(tiles) == {export_float8_grouped_mm.TCGEN05_TILES}
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (12, 1))
    tiles.clear()
    assert export_float8_grouped_mm._register() is not None
    assert set(tiles) == {export_float8_grouped_mm.TILES}
    monkeypatch.setattr(export_float8_grouped_mm, "_triton_op", lambda: unsupported)
    assert export_float8_grouped_mm._register() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the plugin runs on CUDA")
def test_the_float8_grouped_gemm_plugin_multiplies_as_pytorch_does_inside_tensorrt():
    torch_tensorrt = _need_triton_op()
    from lerobot.policies.praxis.export_float8_grouped_mm import FLOAT8_GROUPED_MM

    if FLOAT8_GROUPED_MM is None:
        pytest.skip(
            f"Torch-TensorRT {torch_tensorrt.__version__} builds no float8 grouped GEMM plugin on this GPU"
        )
    generator = torch.Generator(device="cuda").manual_seed(0)
    # Sizes that are no multiples of the kernel's tiles.
    experts, k, n, count = 8, 272, 400, 1000
    data, scale = praxis_layers._float8_bank(
        torch.randn(experts, k, n, device="cuda", generator=generator) * 0.02
    )
    x = torch.randn(count, k, device="cuda", generator=generator).to(torch.bfloat16)
    rows, rows_scale = praxis_layers._float8_rows(x)
    # Uneven experts, an empty one, and every row in an expert, as the experts' dispatch sorts them.
    offsets = torch.tensor([100, 100, 350, 351, 600, 640, 900, count], dtype=torch.int32, device="cuda")

    class Experts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("data", data)
            self.register_buffer("scale", scale)

        def forward(self, rows, rows_scale, offsets):
            return FLOAT8_GROUPED_MM(rows, rows_scale, self.data, self.scale, offsets)

    inputs = [rows, rows_scale, offsets]
    program = torch.export.export(Experts(), tuple(inputs))
    compiled = torch_tensorrt.dynamo.compile(program, inputs=inputs, min_block_size=1)
    _assert_runs_in_tensorrt(compiled)
    expected = praxis_layers._float8_grouped_mm_pytorch(
        rows, rows_scale, data, scale, offsets, torch.bfloat16
    )
    # Only the order of the float32 sums differs.
    torch.testing.assert_close(compiled(*inputs), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the plugin runs on CUDA")
def test_the_float8_gate_up_plugin_computes_as_pytorch_does_inside_tensorrt():
    torch_tensorrt = _need_triton_op()
    from lerobot.policies.praxis.export_float8_gate_up import FLOAT8_GATE_UP, _float8_gate_up_eager

    if FLOAT8_GATE_UP is None:
        pytest.skip(
            f"Torch-TensorRT {torch_tensorrt.__version__} builds no float8 gate and up plugin on this GPU"
        )
    generator = torch.Generator(device="cuda").manual_seed(0)
    # Experts of several row tiles and columns of two tiles, none of them full, an expert without rows, and each token
    # taken top_k times.
    tokens, top_k, experts, k, hidden = 200, 2, 6, 272, 208
    data, scale = praxis_layers._float8_bank(
        torch.randn(experts, k, 2 * hidden, device="cuda", generator=generator) * 0.02
    )
    x = torch.randn(tokens, k, device="cuda", generator=generator).to(torch.bfloat16)
    rows, rows_scale = praxis_layers._float8_rows(x)
    chosen = torch.randint(0, experts - 1, (tokens * top_k,), device="cuda", generator=generator)
    chosen[chosen == 3] = experts - 1
    _, order = chosen.sort(stable=True)
    index = torch.arange(tokens, device="cuda").repeat_interleave(top_k)[order]
    offsets = torch.bincount(chosen, minlength=experts).cumsum(0).to(torch.int32)

    class GateUp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("data", data)
            self.register_buffer("scale", scale)

        def forward(self, rows, rows_scale, index, offsets):
            return FLOAT8_GATE_UP(rows, rows_scale, index, self.data, self.scale, offsets)

    inputs = [rows, rows_scale, index, offsets]
    program = torch.export.export(GateUp(), tuple(inputs))
    compiled = torch_tensorrt.dynamo.compile(program, inputs=inputs, min_block_size=1)
    _assert_runs_in_tensorrt(compiled)
    expected = _float8_gate_up_eager(rows, rows_scale, index, data, scale.unflatten(1, (2, hidden)), offsets)
    # The order of the products' float32 sums can move a bfloat16 rounding, and near zero that has no relative bound.
    torch.testing.assert_close(compiled(*inputs), expected, atol=1e-2, rtol=2**-6)


@pytest.mark.parametrize("tensorrt", [True, False], ids=["tensorrt", "test_backend"])
def test_with_tensorrt_encode_traces_the_autoencoders_norm_steps_and_averages_as_plugins(
    tmp_path, monkeypatch, tensorrt
):
    """Written in PyTorch, TensorRT runs each norm step of the autoencoder as about five kernels."""
    kernels = praxis_vae.AutoencoderKernels(lambda x, gamma: x, lambda x, factor_t: x)
    monkeypatch.setattr(export_praxis, "_autoencoder_kernel_plugins", lambda: kernels)
    monkeypatch.setattr(export_praxis, "_top_k_sum_plugin", lambda: None)
    monkeypatch.setattr(export_praxis, "_expert_gate_up_plugin", lambda: None)
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    if tensorrt:
        recipe.cfg.backend = ExecuTorchTensorRTConfig()
    seen = []
    encode_eager = recipe.policy.vae.encode_eager

    def encode(*args, **kwargs):
        seen.append(praxis_vae._KERNELS)
        return encode_eager(*args, **kwargs)

    monkeypatch.setattr(recipe.policy.vae, "encode_eager", encode)
    parts = list(recipe.parts())
    assert seen
    assert set(seen) == ({kernels} if tensorrt else {None})
    assert praxis_vae._KERNELS is None
    # The compile step registers the plugins' operators before it loads the program.
    assert {part.name: part.operator_modules for part in parts} == {
        part.name: (export_praxis.AUTOENCODER_KERNELS_MODULE,) if tensorrt and part.name == "encode" else ()
        for part in parts
    }


@pytest.mark.parametrize("factor_t", [1, 2])
def test_the_autoencoder_plugins_pytorch_sides_compute_what_the_encoders_modules_do(factor_t):
    """So the export's checks run the values the encoder computes."""
    generator = torch.Generator().manual_seed(0)
    channels = 24
    x = torch.randn(1, channels, 2 * 3, 8, 12, generator=generator).to(torch.float16)
    norm = praxis_vae.RMSNorm(channels, images=False)
    torch.nn.init.normal_(norm.gamma, generator=generator)
    norm = norm.to(torch.bfloat16)
    expected = torch.nn.functional.silu(norm(x.float())).to(torch.float16)
    gamma = norm.gamma.reshape(-1).float()
    assert torch.equal(export_autoencoder_kernels.norm_silu_reference(x, gamma), expected)
    patches = x.reshape(1, channels, 6 // factor_t, factor_t, 4, 2, 6, 2)
    average = praxis_vae.AvgDown3D(channels, channels * factor_t, factor_t, 2)
    assert torch.equal(export_autoencoder_kernels.avg_down_reference(patches), average(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the plugins run on CUDA")
def test_the_autoencoder_plugins_compute_as_pytorch_does_inside_tensorrt():
    torch_tensorrt = _need_triton_op()
    kernels = export_autoencoder_kernels.AUTOENCODER_KERNELS
    assert kernels is not None
    generator = torch.Generator(device="cuda").manual_seed(0)
    # Channels and positions that are no multiple of the norm kernel's tile.
    x = torch.randn(2, 40, 3, 6, 10, device="cuda", generator=generator).to(torch.float16) * 4
    gamma = torch.rand(40, device="cuda", generator=generator) + 0.5

    class Steps(torch.nn.Module):
        def forward(self, x, gamma):
            return (
                kernels.norm_silu(x, gamma),
                kernels.avg_down(x[:, :, 1:], 1),
                kernels.avg_down(x[:, :, 1:], 2),
            )

    inputs = [x, gamma]
    program = torch.export.export(Steps(), tuple(inputs))
    compiled = torch_tensorrt.dynamo.compile(program, inputs=inputs, min_block_size=1, disable_tf32=True)
    _assert_runs_in_tensorrt(compiled)
    norm_silu, average, average_2 = compiled(*inputs)
    torch.testing.assert_close(
        norm_silu, export_autoencoder_kernels.norm_silu_reference(x, gamma), rtol=2e-3, atol=2e-3
    )
    for result, factor_t in ((average, 1), (average_2, 2)):
        expected = praxis_vae.AvgDown3D(40, 40 * factor_t, factor_t, 2)(x[:, :, 1:])
        torch.testing.assert_close(result, expected, rtol=2e-3, atol=2e-3)


def test_the_sample_decoder_decodes_a_chunk_alone_as_in_the_whole_window(tmp_path, monkeypatch):
    """`sample` decodes one chunk alone, with the rotary table of its positions in the window: it gets the whole
    window's velocity of that chunk, and the other chunk's positions give another velocity."""
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    parts = {part.name: part for part in recipe.parts()}
    sample = parts["sample"].module
    anchor, clean, video, _, proprio, _, _, time, *_ = parts["sample"].example_inputs
    chunks = export_praxis.WINDOW_CHUNKS
    constants = recipe.plan().constants
    ropes = [torch.from_numpy(constants[f"decoder_rope_{chunk}"]) for chunk in range(chunks)]
    dtype = sample.encoder.registers.dtype
    noised, timesteps = video.to(dtype), time.to(dtype).expand(1, chunks)
    with torch.no_grad(), recipe._plain_pytorch():
        ((cond, _),) = sample.encoder.forward_chunk_causal_pair(
            [sample.text], anchor.to(dtype), clean.to(dtype), noised, timesteps
        )
        window = sample.decoder.forward_chunk_causal(noised, cond, timesteps)
        for chunk, expected in enumerate(window.chunk(chunks, dim=2)):
            for positions, rope in enumerate(ropes):
                alone = sample.decoder.forward_chunk_causal(
                    noised.chunk(chunks, dim=2)[chunk],
                    cond.chunk(chunks, dim=1)[chunk],
                    timesteps[:, :1],
                    rope=rope,
                )
                if positions == chunk:
                    torch.testing.assert_close(alone, expected, rtol=0, atol=1e-5)
                else:
                    assert (alone - expected).abs().max() > 1e-2


def test_each_chunk_decodes_and_steps_only_the_chunk_it_plays(tmp_path, monkeypatch):
    """The first chunk after a reset picks the first chunk of the window, every later one the second, and
    `sample` moves only the picked chunk's latents."""
    recipe = make_recipe(tmp_path / "checkpoint", monkeypatch)
    plan = recipe.plan()
    for runs, played in ((plan.first_runs[0], 0), (plan.runs, 1)):
        (run,) = (run for run in runs if run.program == "sample")
        assert run.inputs[5:7] == [f"played_{played}", f"decoder_rope_{played}"]
    part = next(part for part in recipe.parts() if part.name == "sample")
    video, played = part.example_inputs[2], part.example_inputs[5]
    with torch.no_grad(), recipe._plain_pytorch():
        moved = part.module(*part.example_inputs)[0]
    chunks = export_praxis.WINDOW_CHUNKS
    for chunk, (before, after) in enumerate(
        zip(video.chunk(chunks, dim=2), moved.chunk(chunks, dim=2), strict=True)
    ):
        assert torch.equal(before, after) != bool(played[chunk])
