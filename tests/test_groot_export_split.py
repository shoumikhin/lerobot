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

import importlib
import json
import os
import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_checkpoint_split_matches_chunk(monkeypatch, tmp_path):
    checkpoint = os.environ.get("GROOT_SPLIT_CHECKPOINT")
    dataset = os.environ.get("GROOT_SPLIT_DATASET")
    if not checkpoint or not dataset:
        pytest.skip("Set GROOT_SPLIT_CHECKPOINT and GROOT_SPLIT_DATASET for the checkpoint check.")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples" / "export"))
    from groot_recipe import GrootChunk, GrootExport, free_memory, part_names

    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.rollout.inference.export.engine import ProgramChain

    class Program:
        def __init__(self, module):
            self.module = module

        def run_device(self, *inputs):
            with torch.no_grad():
                return self.module(*(torch.as_tensor(x, device="cuda") for x in inputs))

        def __call__(self, *inputs):
            return self.run_device(*inputs).float().cpu().numpy()

    torch.manual_seed(14)
    np.random.seed(14)
    args = Namespace(
        output_dir=tmp_path,
        policy_path=checkpoint,
        backend="eager",
        tolerance=0.0,
        dataset=dataset,
        dataset_root=os.environ.get("GROOT_SPLIT_DATASET_ROOT"),
    )
    export = GrootExport(args, split=True)
    programs, metadata = {}, {}
    for name, module, inputs, input_names, output_names in export.parts():
        programs[name] = Program(module)
        metadata[name] = {"file": f"{name}.pte", "inputs": input_names, "outputs": output_names}
        assert len(inputs) == len(input_names)
        with torch.no_grad():
            outputs = module(*inputs)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        assert len(outputs) == len(output_names)
        print(f"CHECKED {name}", flush=True)
        del module, outputs
        free_memory()
    assert list(programs) == part_names(checkpoint)
    export.write(metadata)
    info = json.loads((tmp_path / "export.json").read_text())
    chain = ProgramChain(programs, info)
    chained = torch.from_numpy(chain(*(x.cpu().numpy() for x in export.inputs)))
    actual = torch.from_numpy(export.expected_actions)
    torch.testing.assert_close(chained, actual, rtol=0, atol=0)
    del chain, programs
    free_memory()
    policy = GrootPolicy.from_pretrained(checkpoint).cuda().eval()
    reference = GrootChunk(policy, export.batch, export.postprocessor, export.frame_processor).eval()
    with torch.no_grad():
        expected = reference(*export.inputs).float().cpu()
    difference = (actual - expected).abs().max().item()
    print(f"SPLIT_VS_CHUNK max_abs={difference:.9g} shape={tuple(actual.shape)}", flush=True)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(chained, expected, rtol=0, atol=0)


@pytest.fixture
def recipe(monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("diffusers")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples" / "export"))
    return importlib.import_module("groot_recipe")


def tiny_head(alternate=True, interleave=True):
    from lerobot.policies.groot.groot_n1_7 import GR00TN17ActionHead, GR00TN17Config

    config = GR00TN17Config(
        backbone_embedding_dim=32,
        hidden_size=16,
        input_embedding_dim=32,
        max_state_dim=6,
        max_action_dim=4,
        action_horizon=3,
        max_num_embodiments=3,
        max_seq_len=16,
        num_inference_timesteps=3,
        use_alternate_vl_dit=alternate,
        diffusion_model_cfg={
            "num_layers": 5,
            "num_attention_heads": 4,
            "attention_head_dim": 8,
            "output_dim": 16,
            "dropout": 0.0,
            "positional_embeddings": None,
            "interleave_self_attention": interleave,
        },
        vl_self_attention_cfg={
            "num_layers": 1,
            "num_attention_heads": 4,
            "attention_head_dim": 8,
            "dropout": 0.0,
            "positional_embeddings": None,
        },
    )
    return GR00TN17ActionHead(config).eval()


def constants():
    image_mask = torch.tensor([[False, True, True, False, False]])
    return {
        "image_mask": image_mask,
        "backbone_attention_mask": torch.tensor([[True, True, True, True, False]]),
        "embodiment_id": torch.tensor([2]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        "position_ids": torch.arange(5)[None, None].expand(3, 1, -1),
        "visual_pos_masks": image_mask,
        "image_token_mask": image_mask[..., None].expand(1, 5, 32),
        "text_embeds": torch.randn(1, 5, 32),
    }


def test_vision_part_preserves_deepstack_and_state(recipe):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    vision = Qwen3VLVisionModel(
        Qwen3VLVisionConfig(
            depth=3,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0, 1, 2],
        )
    ).eval()

    class Observation(torch.nn.Module):
        def forward(self, pixels, state):
            return pixels, state

    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    pixels, state = torch.randn(32, 24), torch.randn(1, 1, 6)
    part = recipe.GrootVision(vision, Observation(), grid, False).eval()
    reference = recipe.Qwen3VisionForExport(vision, grid).eval()
    with torch.no_grad():
        images, deepstack = reference(pixels)
        actual = part(pixels, state)
        captured = torch.export.export(part, (pixels, state)).module()(pixels, state)
    for got, saved, expected in zip(actual, captured, (images, *deepstack, state), strict=True):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        torch.testing.assert_close(saved, expected, rtol=0, atol=0)


def test_language_slices_preserve_global_deepstack_indices_and_pre_norm_output(recipe):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    text = Qwen3VLTextModel(
        Qwen3VLTextConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 2]},
        )
    ).eval()
    head, fixed = tiny_head(), constants()
    images, state = torch.randn(2, 32), torch.randn(1, 1, 6)
    deepstack = tuple(torch.randn(2, 32) for _ in range(3))
    with torch.no_grad():
        embeddings = fixed["text_embeds"].masked_scatter(fixed["image_token_mask"], images)
        expected = recipe.LLMForExport(text)(
            embeddings, fixed["attention_mask"], fixed["position_ids"], fixed["visual_pos_masks"], *deepstack
        )
        expected = head.vl_self_attention(head.vlln(expected))
        expected_state = head.state_encoder(state, fixed["embodiment_id"])
        hidden = images
        for start, stop in ((0, 1), (1, 3), (3, 4)):
            part = recipe.GrootLanguageModel(text, head, fixed, start, stop, False).eval()
            inputs = (hidden, state, *deepstack)
            hidden = part(*inputs)
            saved = torch.export.export(part, inputs).module()(*inputs)
            torch.testing.assert_close(saved, hidden, rtol=0, atol=0)
    torch.testing.assert_close(hidden, (expected, expected_state), rtol=0, atol=0)


@pytest.mark.parametrize("alternate,interleave", [(False, False), (False, True), (True, True)])
def test_step_slices_and_program_chain_match_action_head(recipe, alternate, interleave):
    from transformers.feature_extraction_utils import BatchFeature

    from lerobot.rollout.inference.export.engine import ProgramChain

    torch.manual_seed(14)
    head, fixed = tiny_head(alternate, interleave), constants()
    features, state, noise = torch.randn(1, 5, 32), torch.randn(1, 1, 32), torch.randn(3, 4)
    with torch.no_grad():
        expected = head.get_action_with_features(
            features,
            state,
            fixed["embodiment_id"],
            BatchFeature(
                {
                    "image_mask": fixed["image_mask"],
                    "backbone_attention_mask": fixed["backbone_attention_mask"],
                }
            ),
            BatchFeature({}),
            noise=noise[None],
        )["action_pred"][0]
    split_head = deepcopy(head)
    parts = [
        recipe.GrootStep(split_head, fixed, start, stop, False).eval()
        for start, stop in ((0, 2), (2, 3), (3, 5))
    ]
    for part in parts:
        for layer in part.modules():
            if isinstance(layer, recipe.CategorySpecificLinear):
                assert layer.W.shape[0] == layer.b.shape[0] == 1
    calls = []

    class Program:
        def __init__(self, module, name):
            self.module, self.name = module, name

        def __call__(self, *arrays):
            calls.append(self.name)
            with torch.no_grad():
                result = self.module(*(torch.from_numpy(x) for x in arrays))
            if isinstance(result, tuple):
                return tuple(x.numpy() for x in result)
            return result.numpy()

    programs = {f"step_{i}": Program(part, f"step_{i}") for i, part in enumerate(parts)}
    programs["actions"] = Program(recipe.GrootActions(2, 3, torch.nn.Identity()), "actions")
    names = {
        "step_0": (["vl", "state", "x_t", "time"], ["hidden", "temb"]),
        "step_1": (["vl", "hidden", "temb"], ["hidden", "temb"]),
        "step_2": (["vl", "hidden", "temb"], ["velocity"]),
        "actions": (["x_t"], ["actions"]),
    }
    info = {
        "inputs": ["vl", "state", "noise"],
        "num_steps": 3,
        "dt": 1 / 3,
        "timesteps": [0, 333, 666],
        "programs": {
            name: {"inputs": inputs, "outputs": outputs} for name, (inputs, outputs) in names.items()
        },
    }
    chain = ProgramChain(programs, info)
    actual = chain.initialize(features.numpy(), state.numpy(), noise.numpy())
    np.testing.assert_allclose(actual, expected[:2, :3].numpy(), rtol=0, atol=1e-6)
    assert calls == ["step_0", "step_1", "step_2"] * 3 + ["actions"]
    inputs = (features, state, noise, torch.tensor([333.0]))
    with torch.no_grad():
        for part in parts:
            result = part(*inputs)
            saved = torch.export.export(part, inputs).module()(*inputs)
            torch.testing.assert_close(saved, result, rtol=0, atol=0)
            if not part.last:
                inputs = (features, *result)


def test_export_parts_uses_a_fresh_process_for_each_part(recipe, monkeypatch, tmp_path):
    names = ["vision", "language_model_0", "step_0", "step_1", "actions"]
    calls = []
    monkeypatch.setattr(recipe, "part_names", lambda path: names)
    monkeypatch.setattr(recipe.subprocess, "run", lambda command, check: calls.append((command, check)))
    monkeypatch.setattr(sys, "argv", ["export", "--policy.path=checkpoint"])
    output = tmp_path / "export"
    args = Namespace(part=None, policy_path="checkpoint", output_dir=output, job_name="groot")
    recipe.export_parts(args, "export", "{name}.pte", None)
    assert output.is_dir()
    assert calls == [
        (
            [
                sys.executable,
                "export",
                "--policy.path=checkpoint",
                f"--output_dir={output}",
                f"--part={name}",
            ],
            True,
        )
        for name in names
    ]


@pytest.mark.parametrize("device_resident", [False, True])
def test_export_parts_saves_the_device_contract(recipe, monkeypatch, tmp_path, device_resident):
    export = recipe.GrootExport.__new__(recipe.GrootExport)
    export.__dict__.update(
        output_dir=tmp_path,
        backend="executorch_tensorrt",
        frame={"observation.state": np.zeros(2, np.float32)},
        noise=torch.zeros(3, 4),
        expected_actions=np.zeros((3, 2), np.float32),
        config=SimpleNamespace(save_pretrained=lambda path: None),
        input_features={},
        input_names=["observation.state", "noise"],
        image_resize=None,
        task="pick",
        tolerance=5.0,
        num_steps=2,
        timesteps=[0, 500],
        start=recipe.time.perf_counter(),
    )
    monkeypatch.setattr(recipe, "part_names", lambda path: ["actions"])
    monkeypatch.setattr(recipe, "GrootExport", lambda args, split: export)
    monkeypatch.setattr(
        export,
        "parts",
        lambda: iter([("actions", torch.nn.Identity(), (export.noise,), ["x_t"], ["action"])]),
    )
    compiled = []
    args = Namespace(part="actions", policy_path="checkpoint")
    recipe.export_parts(
        args, "export", "{name}.pte", lambda *args: compiled.append(args[-1]), device_resident=device_resident
    )
    info = json.loads((tmp_path / "export.json").read_text())
    assert compiled == [tmp_path / "actions.pte"]
    assert info["device_resident"] is device_resident
    assert info["programs"]["actions"]["file"] == "actions.pte"
    assert info["dt"] == 0.5 and info["timesteps"] == [0, 500]


def test_part_loader_preserves_precision_without_changing_policy_config(recipe, monkeypatch):
    calls = []
    export = recipe.GrootExport.__new__(recipe.GrootExport)
    export.config = SimpleNamespace(use_bf16=True, dtype=torch.float32)
    export.model_file = "checkpoint"
    monkeypatch.setattr(
        recipe.GrootPolicy,
        "load_checkpoint_modules",
        lambda path, config, modules: calls.append((path, config, modules)),
    )
    export.load_part(recipe.VISION)
    assert export.config.use_bf16
    path, config, modules = calls[0]
    assert path == export.model_file and modules == [recipe.VISION]
    assert not config.use_bf16 and config.dtype == torch.float32


def test_actions_apply_the_postprocessor_after_slicing(recipe):
    class Postprocessor(torch.nn.Module):
        def forward(self, actions):
            return actions * 7 + 3

    part = recipe.GrootActions(2, 3, Postprocessor())
    values = torch.arange(20).reshape(4, 5).float()
    torch.testing.assert_close(part(values), values[:2, :3] * 7 + 3)
    saved = torch.export.export(part, (values,)).module()(values)
    torch.testing.assert_close(saved, part(values))


def test_part_names_cover_remainders_and_reject_unknown_parts(recipe, monkeypatch):
    monkeypatch.setattr(
        recipe.PreTrainedConfig, "from_pretrained", lambda path: SimpleNamespace(base_model_path="base")
    )
    monkeypatch.setattr(
        recipe.GR00TN17Config,
        "from_pretrained",
        lambda path: SimpleNamespace(select_layer=16, diffusion_model_cfg={"num_layers": 25}),
    )
    assert recipe.part_names("checkpoint") == [
        "vision",
        "language_model_0",
        "language_model_1",
        "language_model_2",
        "step_0",
        "step_1",
        "step_2",
        "step_3",
        "actions",
    ]
    with pytest.raises(ValueError, match="Unknown part"):
        recipe.export_parts(Namespace(part="bad", policy_path="checkpoint"), "export", "{name}.pte", None)
