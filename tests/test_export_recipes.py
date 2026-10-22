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

"""Each recipe exported from start to end on the CPU, with a small random policy.

The compiler is replaced by a runner that replays the module the recipe hands it, so the test covers the
recipe's parts, the folder it writes and the rollout engine's startup check, without a GPU.
"""

import importlib
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.inference.export import ExportInferenceEngine, engine as export_engine
from tests.export_helpers import action_pipeline
from tests.utils import require_cuda, require_hf_token

pytest.importorskip("datasets", reason="the export examples require lerobot[dataset]")
pytest.importorskip("av", reason="the export examples require lerobot[dataset]")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "export"


@pytest.fixture
def recipes(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    yield
    for name in ("common", "act", "smolvla", "pi05", "groot"):
        sys.modules.pop(name, None)


class Replay:
    """Stands in for a compiled program: runs the module the recipe exported, on host arrays."""

    def __init__(self, module):
        self.module = module.eval()

    def run_device(self, *inputs):
        with torch.no_grad():
            outputs = self.module(*(torch.as_tensor(np.asarray(x)) for x in inputs))
        return outputs if isinstance(outputs, tuple) else (outputs,)

    def __call__(self, *inputs):
        (output,) = self.run_device(*inputs)
        return output.numpy()

    def upload(self, value):
        return value


def export_with_replay(recipe_name, args, monkeypatch):
    """Export through the shared part loop, compiling each part into a `Replay` the engine then loads."""
    common = importlib.import_module("common")
    # The replay runs the parts on host arrays, so the recipe runs as on a machine without a GPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    recipe = importlib.import_module(recipe_name)
    compiled = {}

    def compile_part(part, path):
        # The rollout hands every input on the GPU; the export places there only the inputs a program reads.
        program = torch.export.export(part.module, part.inputs)
        user_inputs = set(program.graph_signature.user_inputs)
        unread = [n.name for n in program.graph.nodes if n.name in user_inputs and not n.users]
        assert not unread, f"{part.name} takes inputs it never reads: {unread}"
        path.write_bytes(b"")
        compiled[path.name] = Replay(part.module)

    names = recipe.part_names(args)
    for name in names:
        args.part = name
        common.run_route("unused", args, recipe, compile_part, ".bin")
    monkeypatch.setattr(
        export_engine,
        "load_programs",
        lambda folder, info: {name: compiled[p["file"]] for name, p in info["programs"].items()},
    )
    return ExportInferenceEngine(args.output_dir, task=getattr(args, "task", None) or "")


def act_checkpoint(path):
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    torch.manual_seed(0)
    config = ACTConfig(
        device="cpu",
        chunk_size=4,
        # Fewer actions played than predicted, so a program that skips the trim cannot pass.
        n_action_steps=2,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=4,
        pretrained_backbone_weights=None,
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    stats = {
        "observation.images.front": {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.2)},
        "observation.state": {"mean": torch.zeros(3), "std": torch.ones(3)},
        "action": {"mean": torch.tensor([1.0, -1.0, 0.0]), "std": torch.tensor([2.0, 2.0, 2.0])},
    }
    policy = ACTPolicy(config)
    policy.save_pretrained(path)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)
    return path


def test_act_exports_and_reproduces_its_test_case(recipes, tmp_path, monkeypatch):
    checkpoint = act_checkpoint(tmp_path / "checkpoint")
    args = Namespace(
        policy_path=str(checkpoint), output_dir=tmp_path / "folder", tolerance=1e-5, route="onnx_tensorrt"
    )
    args.output_dir.mkdir()
    engine = export_with_replay("act", args, monkeypatch)
    info = __import__("json").loads((args.output_dir / "export.json").read_text())
    assert list(info["programs"]) == ["model"]
    assert info["inputs"] == ["observation.images.front", "observation.state"]
    assert "noise_shape" not in info
    frame = {
        "observation.images.front": np.zeros((32, 32, 3), np.uint8),
        "observation.state": np.zeros(3, np.float32),
    }
    actions = [engine.get_action(frame) for _ in range(4)]
    assert all(action.shape == (3,) for action in actions)


def test_test_case_images_change_smoothly_like_a_camera_image(recipes):
    """Random pixels make a vision-language policy's bfloat16 actions swing, so the test case avoids them."""

    from common import IMAGE_CELLS, frame_from_policy

    camera = "observation.images.front"
    config = SimpleNamespace(
        input_features={
            camera: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        image_features={camera},
    )
    image = frame_from_policy(config)[camera].astype(int)
    assert image.shape == (48, 64, 3) and image.std() > 0
    # Bilinear blending across cells of 8 pixels moves each pixel at most 1/8 of the range from its neighbor.
    for axis, cells in ((0, IMAGE_CELLS[0]), (1, IMAGE_CELLS[1])):
        assert np.abs(np.diff(image, axis=axis)).max() <= 255 * cells / image.shape[axis] + 1


def test_act_with_temporal_ensembling_is_refused(recipes, monkeypatch):
    act = importlib.import_module("act")
    policy = SimpleNamespace(config=SimpleNamespace(temporal_ensemble_coeff=0.01))
    policy.to = lambda device: policy
    policy.eval = lambda: policy
    monkeypatch.setattr(act.ACTPolicy, "from_pretrained", lambda path: policy)
    with pytest.raises(SystemExit, match="temporal_ensemble_coeff"):
        act.Exporter(Namespace(policy_path="checkpoint"))


def tiny_smolvlm(path):
    """A local copy of SmolVLM2's config, tokenizer and processor, with a small vision and language model."""
    import json
    import shutil

    from huggingface_hub import snapshot_download

    # SmolVLM2 is public; only its config, tokenizer and processor files are needed, not its weights.
    source = Path(
        snapshot_download("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", allow_patterns=["*.json", "*.txt"])
    )
    path.mkdir()
    for file in source.iterdir():
        if file.is_file() and file.suffix in (".json", ".txt"):
            shutil.copy(file, path / file.name)
    config = json.loads((path / "config.json").read_text())
    config["text_config"].update(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    # A 64-pixel image in 4-pixel patches: a 16 x 16 grid, which the connector shrinks by its scale factor.
    config["vision_config"].update(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=64,
        patch_size=4,
    )
    (path / "config.json").write_text(json.dumps(config))
    return path


def dataset_metadata(features, tasks):
    import pandas as pd

    return SimpleNamespace(features=features, tasks=pd.DataFrame(index=tasks))


def smolvla_checkpoint(path, vlm):
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor import RenameObservationsProcessorStep

    torch.manual_seed(0)
    config = SmolVLAConfig(
        device="cpu",
        vlm_model_name=str(vlm),
        load_vlm_weights=False,
        num_vlm_layers=2,
        chunk_size=4,
        n_action_steps=2,
        max_state_dim=6,
        max_action_dim=6,
        num_steps=2,
        resize_imgs_with_padding=(64, 64),
        input_features={
            "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    stats = {
        "observation.state": {"mean": torch.zeros(3), "std": torch.ones(3)},
        "action": {"mean": torch.tensor([1.0, -1.0, 0.0]), "std": torch.tensor([2.0, 2.0, 2.0])},
    }
    SmolVLAPolicy(config).save_pretrained(path)
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    # As lerobot-train saves it: the robot's camera names map to the checkpoint's placeholders.
    rename = next(s for s in preprocessor.steps if isinstance(s, RenameObservationsProcessorStep))
    rename.rename_map = {"observation.images.front": "observation.images.camera1"}
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)
    return path


def test_smolvla_exports_for_the_robots_cameras_and_reproduces_its_test_case(recipes, tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    vlm = tiny_smolvlm(tmp_path / "vlm")
    checkpoint = smolvla_checkpoint(tmp_path / "checkpoint", vlm)
    features = {
        # The robot names its camera differently from the checkpoint's placeholder, and stores it channel first.
        "observation.images.front": {
            "dtype": "video",
            "shape": (3, 32, 32),
            "names": ["channels", "height", "width"],
        },
        # A camera the dataset recorded but the policy was not trained on.
        "observation.images.side": {
            "dtype": "video",
            "shape": (32, 32, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (3,), "names": None},
        "action": {"dtype": "float32", "shape": (3,), "names": None},
    }
    metadata = dataset_metadata(features, ["pick the cup", "open the drawer"])
    monkeypatch.setattr(
        importlib.import_module("common"), "LeRobotDatasetMetadata", lambda repo_id, root=None: metadata
    )
    args = Namespace(
        policy_path=str(checkpoint),
        output_dir=tmp_path / "folder",
        tolerance=1e-4,
        route="executorch_tensorrt",
        dataset="user/dataset",
        dataset_root=None,
        task="open the drawer",
    )
    args.output_dir.mkdir()
    engine = export_with_replay("smolvla", args, monkeypatch)
    info = __import__("json").loads((args.output_dir / "export.json").read_text())
    assert info["inputs"] == ["observation.images.front", "observation.state", "noise"]
    assert info["task"] == "open the drawer"
    assert info["noise_shape"] == [4, 6]
    frame = {
        "observation.images.front": np.zeros((32, 32, 3), np.uint8),
        "observation.state": np.zeros(3, np.float32),
    }
    assert engine.get_action(frame).shape == (3,)


def test_smolvla_recipe_runs_a_checkpoint_saved_with_compile_model_uncompiled(recipes, tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from lerobot.configs import PreTrainedConfig

    vlm = tiny_smolvlm(tmp_path / "vlm")
    checkpoint = smolvla_checkpoint(tmp_path / "checkpoint", vlm)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.compile_model = True
    config.save_pretrained(checkpoint)
    features = {
        "observation.images.front": {
            "dtype": "video",
            "shape": (32, 32, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (3,), "names": None},
        "action": {"dtype": "float32", "shape": (3,), "names": None},
    }
    metadata = dataset_metadata(features, ["pick the cup"])
    monkeypatch.setattr(
        importlib.import_module("common"), "LeRobotDatasetMetadata", lambda repo_id, root=None: metadata
    )
    args = Namespace(policy_path=str(checkpoint), dataset="user/dataset", dataset_root=None, task=None)
    exporter = importlib.import_module("smolvla").Exporter(args)
    # The policy compiles its sampler by replacing the method on the instance.
    assert "sample_actions" not in vars(exporter.policy.model)


def pi05_checkpoint(path, monkeypatch, dtype=torch.float32, cameras=("observation.images.base_0_rgb",)):
    """A pi0.5 checkpoint with a four-layer Gemma, so the chain splits its language model into two groups."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Config, PI05Policy, modeling_pi05
    from tests.policies.pi0_pi05.utils.meta_load import save_checkpoint, use_tiny_backbone

    use_tiny_backbone(monkeypatch, modeling_pi05)
    tiny = modeling_pi05.GemmaConfig(width=16, depth=4, mlp_dim=32, num_heads=1, num_kv_heads=1, head_dim=16)
    monkeypatch.setattr(modeling_pi05, "get_gemma_config", lambda variant: tiny)
    paligemma = modeling_pi05.PaliGemmaForConditionalGenerationWithPiGemma

    def small_projector(config):
        # pi0.5 fixes the image projector at Gemma 2B's width; the four-layer Gemma here is narrower.
        config.vision_config.projection_dim = tiny.width
        config.vision_config.intermediate_size = 32
        return paligemma(config=config)

    monkeypatch.setattr(modeling_pi05, "PaliGemmaForConditionalGenerationWithPiGemma", small_projector)
    config = PI05Config(
        device="cpu",
        dtype=dtype,
        image_resolution=(28, 28),
        chunk_size=4,
        n_action_steps=2,
        max_state_dim=8,
        max_action_dim=8,
        num_inference_steps=2,
        tokenizer_max_length=48,
        input_features={
            **{name: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 28, 28)) for name in cameras},
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    renames = {f"time_mlp_in.{name}": f"action_time_mlp_in.{name}" for name in ("weight", "bias")}
    save_checkpoint(PI05Policy, config, path, renames)
    config.save_pretrained(path)
    stats = {
        "observation.state": {
            "mean": torch.zeros(3),
            "std": torch.ones(3),
            "q01": -torch.ones(3),
            "q99": torch.ones(3),
        },
        "action": {"mean": torch.zeros(3), "std": torch.ones(3), "q01": -torch.ones(3), "q99": torch.ones(3)},
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(path)
    postprocessor.save_pretrained(path)
    return path, config


@require_hf_token
def test_pi05_chain_reproduces_the_policy_for_any_task(recipes, tmp_path, monkeypatch):
    """The chained parts compute what `PI05Policy.predict_action_chunk` computes, from the same frame and noise,
    for the task of the export and for another one given at run time."""
    pytest.importorskip("transformers")
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.utils import prepare_observation_for_inference

    torch.manual_seed(0)
    checkpoint, config = pi05_checkpoint(tmp_path / "checkpoint", monkeypatch)
    task = "pick up the cube"
    args = Namespace(
        policy_path=str(checkpoint),
        output_dir=tmp_path / "folder",
        tolerance=1e-4,
        route="executorch_tensorrt",
        task=task,
        cameras=None,
    )
    args.output_dir.mkdir()
    engine = export_with_replay("pi05", args, monkeypatch)
    info = __import__("json").loads((args.output_dir / "export.json").read_text())
    assert list(info["programs"]) == ["embed", "language_model_0", "language_model_1", "step", "actions"]
    assert info["steps"] == ["step"] and info["timesteps"] == [1.0, 0.5]

    case = __import__("safetensors.numpy", fromlist=["load_file"]).load_file(
        args.output_dir / "test_case.safetensors"
    )
    policy = PI05Policy.from_pretrained(checkpoint, config=config).eval()
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(checkpoint))
    frame = {name: case[name] for name in ("observation.images.base_0_rgb", "observation.state")}

    def policy_actions(task, noise):
        with torch.no_grad():
            batch = preprocessor(prepare_observation_for_inference(dict(frame), torch.device("cpu"), task))
            actions = policy.predict_action_chunk(batch, noise=torch.from_numpy(noise)[None])
            return postprocessor(actions[:, : config.n_action_steps])[0].numpy()

    np.testing.assert_allclose(
        case["expected_actions"], policy_actions(task, case["noise"]), rtol=0, atol=1e-4
    )
    other = "open_the drawer\n"
    assert engine.set_task(other)
    np.random.seed(0)
    actions = np.stack([engine.get_action(frame) for _ in range(config.n_action_steps)])
    noise = np.random.RandomState(0).standard_normal(case["noise"].shape).astype(np.float32)
    np.testing.assert_allclose(actions, policy_actions(other, noise), rtol=0, atol=1e-4)
    # The task changes the actions, so the comparison above would catch a program that ignores it.
    assert not np.allclose(policy_actions(task, noise), policy_actions(other, noise), atol=1e-3)


@require_hf_token
@pytest.mark.parametrize(
    "exported,sent",
    [
        (None, ["observation.images.wrist"]),
        (None, ["observation.images.base_0_rgb"]),
        (["observation.images.wrist"], ["observation.images.wrist"]),
    ],
)
def test_pi05_runs_any_cameras_the_robot_sends_as_the_policy_does(
    recipes, tmp_path, monkeypatch, exported, sent
):
    """A camera the robot does not send, or one the export leaves out with --cameras, is filled in as the policy
    fills in a camera missing from its batch."""
    pytest.importorskip("transformers")
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.utils import prepare_observation_for_inference

    torch.manual_seed(0)
    cameras = ("observation.images.base_0_rgb", "observation.images.wrist")
    checkpoint, config = pi05_checkpoint(tmp_path / "checkpoint", monkeypatch, cameras=cameras)
    args = Namespace(
        policy_path=str(checkpoint),
        output_dir=tmp_path / "folder",
        tolerance=1e-4,
        route="executorch_tensorrt",
        task="pick up the cube",
        cameras=exported,
    )
    args.output_dir.mkdir()
    engine = export_with_replay("pi05", args, monkeypatch)
    case = __import__("safetensors.numpy", fromlist=["load_file"]).load_file(
        args.output_dir / "test_case.safetensors"
    )
    frame = {name: case[name] for name in (*sent, "observation.state")}
    np.random.seed(0)
    actions = np.stack([engine.get_action(frame) for _ in range(config.n_action_steps)])
    noise = np.random.RandomState(0).standard_normal(case["noise"].shape).astype(np.float32)
    policy = PI05Policy.from_pretrained(checkpoint, config=config).eval()
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(checkpoint))
    with torch.no_grad():
        batch = preprocessor(prepare_observation_for_inference(dict(frame), torch.device("cpu"), args.task))
        expected = policy.predict_action_chunk(batch, noise=torch.from_numpy(noise)[None])
        expected = postprocessor(expected[:, : config.n_action_steps])[0].numpy()
    np.testing.assert_allclose(actions, expected, rtol=0, atol=1e-4)
    with pytest.raises(ValueError, match="none of the policy's cameras"):
        engine.get_action({"observation.state": case["observation.state"]})


@require_cuda
@require_hf_token
def test_pi05_parts_hold_linear_weights_in_bfloat16_and_compute_what_autocast_computes(
    recipes, tmp_path, monkeypatch
):
    pytest.importorskip("transformers")
    torch.manual_seed(0)
    checkpoint, _ = pi05_checkpoint(tmp_path / "checkpoint", monkeypatch, dtype=torch.bfloat16)
    pi05 = importlib.import_module("pi05")
    exporter = pi05.Exporter(Namespace(policy_path=str(checkpoint), task="pick up the cube", cameras=None))
    with monkeypatch.context() as patch:
        patch.setattr(pi05, "hold_linear_weights_in_bfloat16", lambda module: None)
        stock_step = pi05.PI05Step(exporter.load_modules(*pi05.STEP_MODULES).model, exporter.dt)
    stock = {
        "embed": pi05.PI05Embed(exporter.load_modules(*pi05.EMBED_MODULES), exporter.observation),
        "step": stock_step,
    }
    for part in exporter.parts():
        if part.name in stock:
            linears = [layer for layer in part.module.modules() if isinstance(layer, torch.nn.Linear)]
            assert linears and all(layer.weight.dtype == torch.bfloat16 for layer in linears), part.name
            with torch.no_grad():
                torch.testing.assert_close(
                    part.module(*part.inputs), stock.pop(part.name)(*part.inputs), rtol=0, atol=0
                )
    assert not stock


@require_hf_token
def test_pi05_prompt_matches_the_policy_prompt_on_bin_edges(recipes):
    """The program's prompt equals pi0.5's own prompt step, for states on the bin edges and between them."""
    pytest.importorskip("transformers")
    from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
    from lerobot.processor import TokenizerProcessorStep, TransitionKey

    pi05 = importlib.import_module("pi05")
    tokenizer = TokenizerProcessorStep(
        tokenizer_name="google/paligemma-3b-pt-224", max_length=64, padding_side="right", padding="max_length"
    )
    state = torch.tensor([[-1.0, -1.0 + 1 / 128, 0.0, 0.5 - 1e-6, 1.0, 0.3]])
    # Row i holds i, so the rows the prompt picks spell out its token ids.
    embedding = torch.nn.Embedding(len(tokenizer.input_tokenizer), 1)
    embedding.weight.data = torch.arange(len(tokenizer.input_tokenizer), dtype=torch.float32)[:, None]
    prompt = pi05.StatePrompt(tokenizer, embedding, state.shape[1])
    for task in ("pick up the cube", " stack_the\nblocks "):
        clean = task.strip().replace("_", " ").replace("\n", " ")
        ids = tokenizer.input_tokenizer.encode(pi05.TASK_TEXT.format(task=clean))
        task_embeddings = torch.zeros(prompt.max_task_tokens, 1)
        task_embeddings[: len(ids), 0] = torch.tensor(ids, dtype=torch.float32)
        rows, mask = prompt(state, task_embeddings, torch.tensor([len(ids)]))
        transition = {
            TransitionKey.OBSERVATION: {"observation.state": state},
            TransitionKey.COMPLEMENTARY_DATA: {"task": [task]},
        }
        transition = Pi05PrepareStateTokenizerProcessorStep(max_state_dim=6)(transition)
        expected = tokenizer(transition)[TransitionKey.OBSERVATION]
        torch.testing.assert_close(rows[..., 0].long(), expected["observation.language.tokens"])
        torch.testing.assert_close(mask, expected["observation.language.attention_mask"].bool())


def test_embedding_tables_are_saved_in_bfloat16_only_when_that_is_exact(recipes, tmp_path):
    from lerobot.inference.export.prompt import read_table

    common = importlib.import_module("common")
    exact = torch.randn(5, 4).to(torch.bfloat16).float()
    inexact = exact + 1e-4
    for table, raw_dtype in ((exact, np.uint16), (inexact, np.float32)):
        common.save_embedding_table(table, tmp_path / "table.safetensors")
        saved = read_table(tmp_path / "table.safetensors")
        assert saved.dtype == raw_dtype
        if raw_dtype == np.uint16:
            saved = (saved.astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(saved, table.numpy())


@pytest.mark.parametrize(
    "shape,names",
    [((12, 16, 3), ["height", "width", "channels"]), ((3, 12, 16), ["channels", "height", "width"])],
)
def test_robot_frame_cameras_are_hwc_whatever_the_dataset_layout(recipes, shape, names):
    common = importlib.import_module("common")
    features = {
        "observation.images.front": {"dtype": "video", "shape": shape, "names": names},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": None},
        "action": {"dtype": "float32", "shape": (6,), "names": None},
    }
    frame = common.frame_from_dataset(features)
    assert frame["observation.images.front"].shape == (12, 16, 3)
    assert frame["observation.images.front"].dtype == np.uint8
    assert frame["observation.state"].shape == (6,) and "action" not in frame


def test_task_must_be_one_of_the_dataset_tasks(recipes):
    import pandas as pd

    common = importlib.import_module("common")
    metadata = SimpleNamespace(tasks=pd.DataFrame(index=["pick the cup", "open the drawer"]))
    assert common.dataset_task(metadata, None) == "pick the cup"
    assert common.dataset_task(metadata, "open the drawer") == "open the drawer"
    with pytest.raises(ValueError, match="one of the dataset's tasks"):
        common.dataset_task(metadata, "close the drawer")


def test_relative_actions_follow_the_robot_state_after_export(recipes):
    common = importlib.import_module("common")
    from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep, TransitionKey

    relative = RelativeActionsProcessorStep(enabled=True)
    relative({TransitionKey.OBSERVATION: {"observation.state": torch.full((1, 3), 50.0)}})
    postprocessor = action_pipeline(AbsoluteActionsProcessorStep(enabled=True, relative_step=relative))
    part = common.ChunkActions(2, 3, postprocessor)
    offsets = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    program = torch.export.export(part, (offsets, torch.full((3,), 50.0))).module()
    moved = torch.tensor([-5.0, 0.0, 5.0])
    torch.testing.assert_close(program(offsets, moved), offsets[:2] + moved)


@pytest.mark.parametrize("relative", [False, True])
def test_actions_program_takes_the_state_only_when_it_reads_it(recipes, relative):
    """An input the program never reads stays on the host, where the rollout cannot hand it one."""
    common = importlib.import_module("common")
    from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep

    relative_step = RelativeActionsProcessorStep(enabled=True)
    absolute_step = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative_step)
    postprocessor = action_pipeline(absolute_step) if relative else action_pipeline()
    part = common.ChunkActions(2, 3, postprocessor).part(torch.zeros(3, 3), torch.zeros(3))
    inputs, names = part.inputs, part.input_names
    program = torch.export.export(part.module, inputs)
    user_inputs = set(program.graph_signature.user_inputs)
    unread = [n.name for n in program.graph.nodes if n.name in user_inputs and not n.users]
    assert unread == []
    assert names == (["x_t", "observation.state"] if relative else ["x_t"])
