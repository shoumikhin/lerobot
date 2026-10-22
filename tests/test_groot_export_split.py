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
import runpy
import sys
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.export_helpers import action_pipeline

pytest.importorskip("datasets", reason="the export examples require lerobot[dataset]")
pytest.importorskip("av", reason="the export examples require lerobot[dataset]")


@pytest.fixture
def recipe(monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("diffusers")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples" / "export"))
    return importlib.import_module("groot")


def whole_language_model(
    recipe, text, embeddings, attention_mask, position_ids, visual_pos_masks, *deepstack
):
    """All layers in one run, with the recipe's mask and deepstack add: the last layer's output before the norm."""
    hidden = embeddings
    mask = recipe.causal_mask(hidden.dtype, hidden.device, *hidden.shape[:2], attention_mask)
    positions = text.rotary_emb(hidden, position_ids)
    for index, layer in enumerate(text.layers):
        hidden = layer(
            hidden,
            attention_mask=mask,
            position_ids=position_ids[0],
            past_key_values=None,
            position_embeddings=positions,
        )
        if index < len(deepstack):
            rows = recipe.image_rows(visual_pos_masks)
            hidden = recipe.deepstack_add(hidden, visual_pos_masks, rows, deepstack[index].to(hidden.dtype))
    return hidden


def tiny_head(alternate=True, interleave=True, num_layers=5):
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
            "num_layers": num_layers,
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


def tiny_vision():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    config = Qwen3VLVisionConfig(
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
    return Qwen3VLVisionModel(config).eval()


def tiny_text(num_layers=4):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    config = Qwen3VLTextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 2]},
    )
    return Qwen3VLTextModel(config).eval()


# A prompt of a text token, two image tokens, then a task of one token padded to two.
REAL = 4


def constants():
    return {
        "image_mask": torch.tensor([[False, True, True, False, False]]),
        "embodiment_id": torch.tensor([2]),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
        "position_ids": torch.arange(5)[None, None].expand(3, 1, -1),
        "task_positions": torch.arange(2),
        "head": 3,
        "head_embeds": torch.randn(1, 3, 32),
        # Not a recipe constant: what the rollout gives the programs for the task.
        "task_embeddings": torch.cat((torch.randn(1, 32), torch.zeros(1, 32))),
        "task_length": torch.tensor([REAL - 3]),
    }


def image_embeddings(fixed, images):
    """The text embeddings with the image features at the image tokens, as Qwen3-VL places them."""
    text = torch.cat((fixed["head_embeds"], fixed["task_embeddings"][None]), dim=1)
    return text.masked_scatter(fixed["image_mask"][..., None].expand(1, 5, 32), images)


def test_vision_part_preserves_deepstack_and_state(recipe):
    vision = tiny_vision()

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
    text = tiny_text()
    head, fixed = tiny_head(), constants()
    images, state = torch.randn(2, 32), torch.randn(1, 1, 6)
    deepstack = tuple(torch.randn(2, 32) for _ in range(3))
    with torch.no_grad():
        # The policy runs the prompt without the task's padding.
        embeddings = image_embeddings(fixed, images)[:, :REAL]
        expected = whole_language_model(
            recipe,
            text,
            embeddings,
            torch.ones(1, REAL, dtype=torch.long),
            fixed["position_ids"][..., :REAL],
            fixed["image_mask"][:, :REAL],
            *deepstack,
        )
        expected = head.vl_self_attention(head.vlln(expected))
        expected_state = head.state_encoder(state, fixed["embodiment_id"])
        hidden = images
        for start, stop in ((0, 1), (1, 3), (3, 4)):
            part = recipe.GrootLanguageModel(text, head, fixed, start, stop, False, len(deepstack)).eval()
            inputs = (
                hidden,
                *(deepstack[i] for i in part.deepstack_layers),
                *((fixed["task_embeddings"],) if start == 0 else ()),
                *((state, fixed["task_length"]) if part.last else ()),
            )
            hidden = part(*inputs)
            saved = torch.export.export(part, inputs).module()(*inputs)
            torch.testing.assert_close(saved, hidden, rtol=0, atol=0)
    features, state_features = hidden
    torch.testing.assert_close(features[:, :REAL], expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(state_features, expected_state, rtol=0, atol=0)


def test_language_slices_have_no_data_dependent_shapes(recipe):
    """TensorRT allocates GPU memory on every run for a data-dependent shape, which a CUDA graph cannot record."""
    text = tiny_text()
    head, fixed = tiny_head(), constants()
    deepstack = tuple(torch.randn(2, 32) for _ in range(3))
    part = recipe.GrootLanguageModel(text, head, fixed, 0, 4, False, len(deepstack)).eval()
    inputs = (
        torch.randn(2, 32),
        *deepstack,
        fixed["task_embeddings"],
        torch.randn(1, 1, 6),
        fixed["task_length"],
    )
    with torch.no_grad():
        program = torch.export.export(part, inputs)
    # Autocast regions are submodules of the exported graph.
    graphs = [
        module.graph for module in program.graph_module.modules() if isinstance(module, torch.fx.GraphModule)
    ]
    targets = {str(node.target) for graph in graphs for node in graph.nodes if node.op == "call_function"}
    assert not {"aten.masked_scatter.default", "aten.nonzero.default"} & targets


def test_last_language_slice_encodes_a_float32_state_with_bfloat16_weights(recipe):
    """A checkpoint loaded in bfloat16 with use_bf16 off has bfloat16 weights, but the robot's state is float32."""
    text, head = tiny_text().to(torch.bfloat16), tiny_head().to(torch.bfloat16)
    fixed = constants()
    hidden, state = torch.randn(1, 5, 32, dtype=torch.bfloat16), torch.randn(1, 1, 6)
    with torch.no_grad():
        expected = head.state_encoder(state.to(torch.bfloat16), fixed["embodiment_id"])
        part = recipe.GrootLanguageModel(text, head, fixed, 3, 4, False, 3).eval()
        _, state_features = part(hidden, state, fixed["task_length"])
    torch.testing.assert_close(state_features, expected, rtol=0, atol=0)


def test_vision_wrapper_matches_the_hugging_face_vision_model(recipe):
    torch.manual_seed(14)
    vision = tiny_vision()
    grid, pixels = torch.tensor([[1, 4, 4], [1, 4, 4]]), torch.randn(32, 24)
    with torch.no_grad():
        expected = vision(pixels, grid)
        images, deepstack = recipe.Qwen3VisionForExport(vision, grid).eval()(pixels)
    torch.testing.assert_close(images, expected.pooler_output, rtol=0, atol=1e-6)
    torch.testing.assert_close(deepstack, torch.stack(expected.deepstack_features), rtol=0, atol=1e-6)


def test_language_model_layers_match_the_hugging_face_text_model(recipe):
    torch.manual_seed(14)
    text = tiny_text()
    fixed = constants()
    deepstack = [torch.randn(2, 32) for _ in range(3)]
    embeddings = image_embeddings(fixed, torch.randn(2, 32))
    last = []
    # GR00T reads the last layer's output, before the model's final norm.
    text.layers[-1].register_forward_hook(lambda module, args, output: last.append(output))
    padding = torch.tensor([[1, 1, 1, 1, 0]])
    with torch.no_grad():
        text(
            inputs_embeds=embeddings,
            attention_mask=padding,
            position_ids=fixed["position_ids"],
            visual_pos_masks=fixed["image_mask"],
            deepstack_visual_embeds=deepstack,
        )
        actual = whole_language_model(
            recipe, text, embeddings, padding, fixed["position_ids"], fixed["image_mask"], *deepstack
        )
    expected = last[0][0] if isinstance(last[0], tuple) else last[0]
    valid = padding[0].bool()
    torch.testing.assert_close(actual[:, valid], expected[:, valid], rtol=0, atol=1e-5)


def test_language_slices_take_only_the_inputs_their_layers_read(recipe):
    text = tiny_text(8)
    head, fixed = tiny_head(), constants()
    slices = {
        (start, stop): recipe.GrootLanguageModel(text, head, fixed, start, stop, False, 3)
        for start, stop in ((0, 2), (2, 6), (6, 8))
    }
    assert [part.deepstack_layers for part in slices.values()] == [[0, 1], [2], []]
    assert [part.last for part in slices.values()] == [False, False, True]


@pytest.mark.parametrize("alternate,interleave", [(False, False), (False, True), (True, True)])
def test_step_slices_and_program_chain_match_action_head(recipe, alternate, interleave):
    from transformers.feature_extraction_utils import BatchFeature

    from lerobot.inference.export.engine import ProgramChain

    torch.manual_seed(14)
    head, fixed = tiny_head(alternate, interleave), constants()
    features, state, noise = torch.randn(1, 5, 32), torch.randn(1, 1, 32), torch.randn(3, 4)
    with torch.no_grad():
        # The policy runs the prompt without the task's padding.
        expected = head.get_action_with_features(
            features[:, :REAL],
            state,
            fixed["embodiment_id"],
            BatchFeature(
                {
                    "image_mask": fixed["image_mask"][:, :REAL],
                    "backbone_attention_mask": torch.ones(1, REAL, dtype=torch.bool),
                }
            ),
            BatchFeature({}),
            noise=noise[None],
        )["action_pred"][0]
    split_head = deepcopy(head)
    parts = [
        recipe.GrootStep(split_head, fixed, start, stop, False, 1 / 3).eval()
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

        def run_device(self, *values):
            calls.append(self.name)
            with torch.no_grad():
                result = self.module(*(torch.as_tensor(x) for x in values))
            return result if isinstance(result, tuple) else (result,)

        def __call__(self, *values):
            (output,) = self.run_device(*values)
            return output.numpy()

        def upload(self, value):
            return value

    programs = {f"step_{i}": Program(part, f"step_{i}") for i, part in enumerate(parts)}
    programs["actions"] = Program(recipe.ChunkActions(2, 3, action_pipeline()), "actions")
    names = {
        "step_0": (["vl", "state", "x_t", "timestep", "task_length"], ["hidden", "temb"]),
        "step_1": (["vl", "hidden", "temb", "task_length"], ["hidden"]),
        "step_2": (["vl", "hidden", "temb", "x_t", "task_length"], ["x_next"]),
        "actions": (["x_t", "robot_state"], ["actions"]),
    }
    info = {
        "inputs": ["vl", "state", "robot_state", "noise"],
        "steps": ["step_0", "step_1", "step_2"],
        "timesteps": [0, 333, 666],
        "programs": {
            name: {"inputs": inputs, "outputs": outputs} for name, (inputs, outputs) in names.items()
        },
    }
    chain = ProgramChain(programs, info)
    chain.hold({"task_length": fixed["task_length"].numpy()})
    actual = chain(features.numpy(), state.numpy(), np.zeros(6, np.float32), noise.numpy())
    np.testing.assert_allclose(actual, expected[:2, :3].numpy(), rtol=0, atol=1e-6)
    assert calls == ["step_0", "step_1", "step_2"] * 3 + ["actions"]
    length = fixed["task_length"]
    inputs = (features, state, noise, torch.tensor([333.0]), length)
    with torch.no_grad():
        for part in parts:
            result = part(*inputs)
            saved = torch.export.export(part, inputs).module()(*inputs)
            torch.testing.assert_close(saved, result, rtol=0, atol=0)
            if not part.last:
                hidden, temb = result if part.start == 0 else (result, inputs[-2])
                inputs = (
                    (features, hidden, temb, length)
                    if not parts[parts.index(part) + 1].last
                    else (features, hidden, temb, noise, length)
                )


def test_diffusion_parts_export_to_onnx_and_run_as_chain(recipe, monkeypatch, tmp_path):
    onnx = pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    from lerobot.inference.export.engine import ProgramChain
    from lerobot.utils.constants import ACTION

    class Vision(torch.nn.Module):
        def forward(self, features, state, robot_state):
            return features, state

    class LanguageModel(torch.nn.Module):
        last = True
        deepstack_layers = []

        def forward(self, features, task_embeddings, state, task_length):
            return features, state

    torch.manual_seed(14)
    head = tiny_head(num_layers=25)
    features, state, noise = torch.randn(1, 5, 32), torch.randn(1, 1, 32), torch.randn(3, 4)
    text = SimpleNamespace(get_input_embeddings=lambda: torch.nn.Embedding(32, 32))
    exporter = recipe.Exporter.__new__(recipe.Exporter)
    exporter.__dict__.update(
        policy=SimpleNamespace(_groot_model=SimpleNamespace(backbone=SimpleNamespace(visual=None))),
        observation=None,
        image_resize=None,
        batch={"image_grid_thw": None, "input_ids": torch.zeros(1, 5, dtype=torch.long)},
        config=SimpleNamespace(use_bf16=False, output_features={ACTION: SimpleNamespace(shape=(3,))}),
        model_config=SimpleNamespace(
            select_layer=1, diffusion_model_cfg={"num_layers": 25}, add_pos_embed=False
        ),
        constants=constants(),
        prompt={"max_tokens": 2},
        noise=noise,
        timesteps=[0, 333, 666],
        num_steps=3,
        n_action_steps=2,
        chunk_actions=recipe.chunk_actions(2, 3, action_pipeline(), action_pipeline()),
        frame={
            "features": features.numpy(),
            "state": state.numpy(),
            "observation.state": np.zeros(3, np.float32),
        },
    )
    monkeypatch.setattr(recipe, "resize_images", lambda frame, geometry: frame)
    # Keep the prefix tiny while exercising the real recipe's diffusion modules and names.
    monkeypatch.setattr(recipe, "GrootVision", lambda *args: Vision())
    monkeypatch.setattr(recipe, "GrootLanguageModel", lambda *args: LanguageModel())
    monkeypatch.setattr(
        exporter,
        "load_modules",
        lambda *args: SimpleNamespace(
            _groot_model=SimpleNamespace(
                backbone=SimpleNamespace(language_model=text), action_head=deepcopy(head)
            )
        ),
    )
    sessions = {}

    def check_engine(onnx_path, engine_path):
        onnx.checker.check_model(onnx.load(onnx_path), full_check=True)
        sessions[engine_path.stem] = onnxruntime.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )

    monkeypatch.setitem(sys.modules, "build_engine", SimpleNamespace(build_engine=check_engine))
    compile_part = runpy.run_path(str(Path(recipe.__file__).with_name("to_onnx_tensorrt.py")))["compile_part"]
    programs, metadata = {}, {}

    class Program:
        def __init__(self, session, input_names):
            self.session, self.input_names = session, input_names

        def run_device(self, *arrays):
            return tuple(
                self.session.run(None, dict(zip(self.input_names, map(np.asarray, arrays), strict=True)))
            )

        def __call__(self, *arrays):
            (output,) = self.run_device(*arrays)
            return output

        def upload(self, value):
            return value

    for part in exporter.parts():
        if not (part.name.startswith("step_") or part.name == "actions"):
            continue
        compile_part(part, tmp_path / f"{part.name}.engine")
        session = sessions[part.name]
        assert [value.name for value in session.get_inputs()] == part.input_names
        assert [value.name for value in session.get_outputs()] == part.output_names
        programs[part.name] = Program(session, part.input_names)
        metadata[part.name] = {"inputs": part.input_names, "outputs": part.output_names}
        with torch.no_grad():
            expected = part.module(*part.inputs)
        expected = expected if isinstance(expected, tuple) else (expected,)
        actual = session.run(None, dict(zip(part.input_names, [x.numpy() for x in part.inputs], strict=True)))
        for got, want in zip(actual, expected, strict=True):
            np.testing.assert_allclose(got, want.numpy(), rtol=1e-5, atol=1e-5)
    assert list(programs) == ["step_0", "step_1", "step_2", "step_3", "actions"]
    info = {
        "inputs": ["vl_embeds", "state_features", "observation.state", "noise"],
        "programs": metadata,
        "steps": ["step_0", "step_1", "step_2", "step_3"],
        "timesteps": exporter.timesteps,
    }
    robot_state = exporter.frame["observation.state"]
    chain = ProgramChain(programs, info)
    chain.hold({"task_length": np.array([2])})
    actual = chain(features.numpy(), state.numpy(), robot_state, noise.numpy())
    np.testing.assert_allclose(actual, exporter.expected_actions, rtol=1e-5, atol=1e-5)


def test_part_loader_preserves_precision_without_changing_policy_config(recipe, monkeypatch):
    calls = []
    exporter = recipe.Exporter.__new__(recipe.Exporter)
    exporter.config = SimpleNamespace(use_bf16=True, dtype=torch.float32)
    exporter.model_file = "checkpoint"
    monkeypatch.setattr(
        recipe.GrootPolicy,
        "load_checkpoint_modules",
        lambda path, config, modules: calls.append((path, config, modules)) or torch.nn.Module(),
    )
    exporter.load_modules(recipe.VISION)
    assert exporter.config.use_bf16
    path, config, modules = calls[0]
    assert path == exporter.model_file and modules == [recipe.VISION]
    assert not config.use_bf16 and config.dtype == torch.float32


@pytest.mark.parametrize("use_bf16", [True, False])
def test_part_loader_holds_only_linear_weights_in_bfloat16_under_autocast(recipe, monkeypatch, use_bf16):
    exporter = recipe.Exporter.__new__(recipe.Exporter)
    exporter.config = SimpleNamespace(use_bf16=use_bf16, dtype=torch.float32)
    exporter.model_file = "checkpoint"
    head = tiny_head()
    monkeypatch.setattr(recipe.GrootPolicy, "load_checkpoint_modules", lambda path, config, modules: head)
    exporter.load_modules(recipe.ACTION_HEAD)
    expected = torch.bfloat16 if use_bf16 else torch.float32
    linear = [m for m in head.modules() if isinstance(m, torch.nn.Linear)]
    assert linear and all(p.dtype == expected for m in linear for p in m.parameters())
    # Norms, embeddings and per-embodiment layers keep the float32 the residual additions need.
    others = [p for m in head.modules() if not isinstance(m, torch.nn.Linear) for p in m.parameters(False)]
    assert others and all(p.dtype == torch.float32 for p in others)


def test_bfloat16_linear_weights_compute_what_autocast_computes(recipe):
    torch.manual_seed(15)
    fixed = constants()
    inputs = (
        torch.randn(1, 5, 32),
        torch.randn(1, 1, 32),
        torch.randn(3, 4),
        torch.tensor([333.0]),
        fixed["task_length"],
    )
    autocast, held = (recipe.GrootStep(tiny_head(), fixed, 0, 5, True, 1 / 3).eval() for _ in range(2))
    held.load_state_dict(autocast.state_dict())
    recipe.hold_linear_weights_in_bfloat16(held)
    linears = [layer for layer in held.modules() if isinstance(layer, torch.nn.Linear)]
    assert linears and all(layer.weight.dtype == torch.bfloat16 for layer in linears)
    with torch.no_grad():
        torch.testing.assert_close(held(*inputs), autocast(*inputs), rtol=0, atol=0)


def test_actions_apply_the_postprocessor_after_slicing(recipe):
    from lerobot.processor import ProcessorStep, TransitionKey

    class ScaleByHorizon(ProcessorStep):
        """Depends on the chunk's length, so slicing after it gives a different answer than before it."""

        def __call__(self, transition):
            actions = transition[TransitionKey.ACTION]
            return {**transition, TransitionKey.ACTION: actions * actions.shape[1] + 3}

        def transform_features(self, features):
            return features

    part = recipe.ChunkActions(2, 3, action_pipeline(ScaleByHorizon()))
    values = torch.arange(20).reshape(4, 5).float()
    state = torch.zeros(6)
    torch.testing.assert_close(part(values, state), values[:2, :3] * 2 + 3)
    saved = torch.export.export(part, (values, state)).module()(values, state)
    torch.testing.assert_close(saved, part(values, state))


def n17_decoder(relative=False, percentiles=False, per_step=False, transform=None):
    """An N1.7 decode step for a two-group arm, a one-number gripper and an end effector pose."""
    from lerobot.policies.groot.processor_groot import GrootN17ActionDecodeStep

    rng = np.random.default_rng(0)

    def stats(dim):
        low = rng.uniform(-3, 0, (4, dim) if per_step else dim)
        return {"min": low, "max": low + rng.uniform(1, 3, low.shape), "q01": low + 0.1, "q99": low + 0.9}

    groups = {"arm": 3, "gripper": 1, "eef": 9}
    raw_stats = {
        "action": {k: {n: v.tolist() for n, v in stats(d).items()} for k, d in groups.items()},
        "relative_action": {k: {n: v.tolist() for n, v in stats(d).items()} for k, d in groups.items()},
    }
    configs = [
        {"rep": "relative" if relative else "absolute", "type": "non_eef", "state_key": "joints"},
        {"rep": "absolute"},
        {"rep": "relative" if relative else "absolute", "type": "eef", "format": "xyz_rot6d"},
    ]
    return GrootN17ActionDecodeStep(
        env_action_dim=12,
        raw_stats=raw_stats,
        modality_config={
            "action": {"modality_keys": list(groups), "action_configs": configs, "delta_indices": [0, 1, 2]}
        },
        use_percentiles=percentiles,
        use_relative_action=relative,
        action_decode_transform=transform,
    )


def n17_pack():
    """What the decoding reads from the packing step: the state's groups, the joints then the end effector."""
    state = {"joints": {"min": [0.0] * 3}, "eef": {"min": [0.0] * 9}}
    return SimpleNamespace(
        raw_stats={"state": state}, modality_config={"state": {"modality_keys": list(state)}}
    )


@pytest.mark.parametrize(
    "options",
    [{}, {"percentiles": True}, {"per_step": True}, {"relative": True}, {"transform": "libero"}],
)
def test_n17_decoding_in_tensors_matches_the_policy_step(recipe, options):
    """The program's decoding computes what GR00T N1.7's own NumPy step computes, for every kind of group."""
    torch.manual_seed(0)
    step = n17_decoder(**options)
    module = recipe.N17ChunkActions(4, recipe.N17ActionDecode(step, n17_pack(), 4))
    actions = torch.rand(6, 16) * 2.4 - 1.2
    pose = torch.cat((torch.randn(3), torch.randn(6)))
    state = torch.cat((torch.randn(3), pose))
    groups = {"joints": state[None, :3].numpy(), "eef": state[None, 3:].numpy()}
    step.pack_step = SimpleNamespace(get_cached_raw_state=lambda: groups)
    expected = action_pipeline(step)(actions[None, :4])[0]
    torch.testing.assert_close(module(actions, state), expected, rtol=1e-5, atol=1e-5)
    saved = torch.export.export(module, (actions, state)).module()
    moved = torch.rand(6, 16) * 2 - 1
    torch.testing.assert_close(saved(moved, state), module(moved, state))


def test_n17_decoding_refuses_what_the_policy_step_refuses(recipe):
    step = n17_decoder(transform="unknown")
    with pytest.raises(ValueError, match="Unsupported N1.7 action decode transform"):
        recipe.N17ActionDecode(step, n17_pack(), 4)


def test_n17_decoding_takes_the_state_only_for_relative_actions(recipe):
    for relative in (False, True):
        step = n17_decoder(relative=relative)
        module = recipe.N17ChunkActions(4, recipe.N17ActionDecode(step, n17_pack(), 4))
        part = module.part(torch.zeros(6, 16), torch.zeros(12))
        assert part.input_names == (["x_t", "observation.state"] if relative else ["x_t"])


def test_n17_decoding_exports_to_onnx_and_follows_new_actions(recipe, tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    module = recipe.N17ChunkActions(
        4, recipe.N17ActionDecode(n17_decoder(relative=True), n17_pack(), 4)
    ).eval()
    actions, state = torch.rand(6, 16) * 2 - 1, torch.randn(12)
    path = tmp_path / "actions.onnx"
    torch.onnx.export(
        module, (actions, state), path, dynamo=True, input_names=["x", "state"], output_names=["a"]
    )
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    moved = torch.rand(6, 16) * 2 - 1
    with torch.no_grad():
        expected = module(moved, state).numpy()
    np.testing.assert_allclose(
        session.run(None, {"x": moved.numpy(), "state": state.numpy()})[0], expected, atol=1e-5
    )


def test_part_names_cover_remainders_and_reject_unknown_parts(recipe, monkeypatch):
    monkeypatch.setattr(
        recipe.PreTrainedConfig, "from_pretrained", lambda path: SimpleNamespace(base_model_path="base")
    )
    monkeypatch.setattr(
        recipe.GR00TN17Config,
        "from_pretrained",
        lambda path: SimpleNamespace(select_layer=16, diffusion_model_cfg={"num_layers": 25}),
    )
    assert recipe.part_names(Namespace(policy_path="checkpoint")) == [
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
    common = importlib.import_module("common")
    with pytest.raises(SystemExit, match="Unknown part"):
        common.run_route("script", Namespace(part="bad", policy_path="checkpoint"), recipe, None, ".pte")
