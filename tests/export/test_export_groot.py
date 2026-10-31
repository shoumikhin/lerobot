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

"""The GR00T N1.7 export recipe on a tiny random GR00T with its own tiny Qwen3-VL processor."""

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("diffusers")
pytest.importorskip("cv2", exc_type=ImportError)
# The policy check runs GR00T's own input step, which needs dm-tree from the groot extra.
pytest.importorskip("tree")

from lerobot.bundle.manifest import Bundle  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.export.configs import ExportConfig  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.groot.configuration_groot import GrootConfig  # noqa: E402
from lerobot.policies.groot.export_groot import GrootExportRecipe, image_stages  # noqa: E402
from lerobot.policies.groot.processor_groot import (  # noqa: E402
    GrootN17VLMEncodeStep,
    _transform_n1_7_image_for_vlm_albumentations,
    _transform_n1_7_image_for_vlm_torch,
)
from lerobot.scripts.lerobot_export import export  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from tests.export.plugins import use_plugins  # noqa: E402
from tests.export.tiny_groot import MODES, TASK, save_checkpoint  # noqa: E402
from tests.export.torch_backend import TorchTestBackend  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def torch_test_backend():
    """Add the test backend to the export's steps, which run in processes of their own."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        use_plugins(monkeypatch, "torch_backend")
        yield


def test_a_bfloat16_policy_without_autocast_exports(tmp_path, capfd):
    """With autocast off, the parts feed the policy's bfloat16 layers bfloat16 values, as its own forward does.

    Its noisy actions are bfloat16 too, rounded at every one of 16 denoising steps, as the chain rounds them.
    """
    checkpoint = save_checkpoint(tmp_path, "albumentations", dtype=torch.bfloat16, num_inference_timesteps=16)
    export(
        [
            "--backend.type=torch_test",
            f"--policy.path={checkpoint}",
            f"--output_dir={tmp_path / 'bundle'}",
            f"--task={TASK}",
            "--tolerance=1e-2",
        ]
    )
    assert "Test case against the whole policy" in capfd.readouterr().err


def test_a_float32_policy_under_bfloat16_autocast_exports(tmp_path, capfd):
    """`use_bf16`, the default, runs the parts under bfloat16 autocast, as the policy's own forward does."""
    checkpoint = save_checkpoint(tmp_path, "albumentations", use_bf16=True)
    export(
        [
            "--backend.type=torch_test",
            f"--policy.path={checkpoint}",
            f"--output_dir={tmp_path / 'bundle'}",
            f"--task={TASK}",
            "--tolerance=1e-4",
        ]
    )
    assert "Test case against the whole policy" in capfd.readouterr().err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_a_bfloat16_policy_under_bfloat16_autocast_on_a_gpu_exports(tmp_path):
    """On a GPU, autocast runs the policy's layer norm in float32, so its vision-language features stay float32
    and so do the noisy actions it holds in their dtype: the chain must not round each update to bfloat16."""

    class CudaTestBackend(TorchTestBackend):
        device = "cuda"

    checkpoint = save_checkpoint(
        tmp_path, "torchvision", dtype=torch.bfloat16, use_bf16=True, num_inference_timesteps=16
    )
    config = GrootConfig.from_pretrained(checkpoint)
    config.device = "cuda"
    # Measured on a GPU: 0 with float32 noisy actions, 2.9e-2 with every update rounded to bfloat16.
    recipe = GrootExportRecipe(
        ExportConfig(
            backend=CudaTestBackend(),
            policy=config,
            output_dir=tmp_path / "bundle",
            task=TASK,
            tolerance=1e-3,
        )
    )
    list(recipe.parts())
    recipe.test_case()


@pytest.mark.parametrize(
    "checkpoint_args",
    [
        # GR00T reads its last language layer before Qwen3-VL adds that layer's deepstack features.
        {"select_layer": 2},
        # The backbone keeps all 7 of its layers when select_layer is above that.
        {"select_layer": 9},
        # A DiT that cross-attends to the image and text tokens together.
        {"alternate_vl_dit": False},
    ],
)
def test_a_policy_with_another_backbone_depth_or_dit_exports(tmp_path, caplog, checkpoint_args):
    checkpoint = save_checkpoint(tmp_path, "torchvision", **checkpoint_args)
    recipe = GrootExportRecipe(_export_config(GrootConfig.from_pretrained(checkpoint), tmp_path))
    list(recipe.parts())
    with caplog.at_level("INFO"):
        recipe.test_case()
    assert "Test case against the whole policy" in caplog.text


def test_the_tiny_checkpoint_does_not_depend_on_the_random_state(tmp_path):
    """So a test's result does not depend on the tests that ran before it."""
    from safetensors.torch import load_file

    weights = []
    for seed in (101, 202):
        torch.manual_seed(seed)
        weights.append(load_file(save_checkpoint(tmp_path / str(seed), "torchvision") / "model.safetensors"))
    assert weights[0].keys() == weights[1].keys()
    assert all(torch.equal(weights[0][name], weights[1][name]) for name in weights[0])


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return save_checkpoint(tmp_path_factory.mktemp("groot"), "torchvision")


def test_the_parts_read_the_commit_the_export_started_from(checkpoint, tmp_path, monkeypatch):
    """Each part reads the checkpoint again, so a branch that moves during the export must not mix two
    versions."""
    import shutil

    from huggingface_hub import constants
    from safetensors.torch import load_file, save_file

    from lerobot.policies.groot.export_groot import ACTION_HEAD

    repo = tmp_path / "hub" / "models--user--tiny-groot"
    first, second = "1" * 40, "2" * 40
    for commit in (first, second):
        shutil.copytree(checkpoint, repo / "snapshots" / commit)
    weights = load_file(checkpoint / "model.safetensors")
    save_file({name: w + 1 for name, w in weights.items()}, repo / "snapshots" / second / "model.safetensors")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(first)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    config = GrootConfig.from_pretrained(checkpoint)
    config.pretrained_path = "user/tiny-groot"
    recipe = GrootExportRecipe(_export_config(config, tmp_path))
    (repo / "refs" / "main").write_text(second)
    decoder = recipe._load_modules(f"{ACTION_HEAD}.action_decoder")._groot_model.action_head.action_decoder
    for name, value in decoder.named_parameters():
        assert torch.equal(value, weights[f"{ACTION_HEAD}.action_decoder.{name}"].to(value))


def test_the_task_padding_stays_out_of_the_action_heads_self_attention(checkpoint, tmp_path):
    """The action head's self-attention attends both ways, so only the prompt mask keeps the task's padding
    out of the features of the real tokens."""
    recipe = GrootExportRecipe(_export_config(GrootConfig.from_pretrained(checkpoint), tmp_path))
    part = next(p for p in recipe.parts() if p.name.startswith("language_model") and p.module.last)
    hidden, *inputs, task_length = part.example_inputs
    real = recipe._prompt["head"] + int(task_length)
    assert real < hidden.shape[1]
    padded = hidden.clone()
    padded[:, real:] = 10 * torch.randn_like(padded[:, real:])
    with torch.no_grad():
        features, _ = part.module(hidden, *inputs, task_length)
        with_other_padding, _ = part.module(padded, *inputs, task_length)
    torch.testing.assert_close(with_other_padding[:, :real], features[:, :real], rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "policy, processor, export_args, error",
    [
        ({"use_flash_attention": True}, {}, {}, "flash attention does not take"),
        ({"num_inference_timesteps": 0}, {}, {}, "at least one step"),
        (
            {"input_features": {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(9,))}},
            {},
            {},
            "State dimension 9 exceeds max_state_dim 8",
        ),
        ({}, {"groot_n1_7_pack_inputs_v1": {"video_horizon": 2}}, {}, "not a video history"),
        (
            {},
            {"groot_n1_7_vlm_encode_v1": {"image_target_size": [20, 20]}},
            {},
            "only the first resize.*each side a multiple of 8",
        ),
        ({}, {}, {"max_task_tokens": 3}, "raise --max_task_tokens to at least that"),
    ],
)
def test_export_refuses_what_it_does_not_reproduce(
    checkpoint, tmp_path, policy, processor, export_args, error
):
    config = _with_saved_processor(checkpoint, tmp_path, processor, policy)
    with pytest.raises(ValueError, match=error):
        GrootExportRecipe(_export_config(config, tmp_path, **export_args))


@pytest.mark.parametrize("seed", [2, 4, 5, 1000])
def test_the_image_check_on_the_cpu_does_not_depend_on_the_test_frame(checkpoint, tmp_path, seed):
    """torchvision's 8-bit CPU kernel moved some random frames of this letterbox 4 levels from the program,
    which reproduces its float32 resize, as on a GPU."""
    letterbox = {"image_target_size": [24, 24], "shortest_image_edge": 28, "letter_box_transform": True}
    letterbox |= {"image_crop_size": None, "crop_fraction": None}
    camera = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 30, 20))
    cameras = dict.fromkeys(MODES["torchvision"][1], camera)
    config = _with_saved_processor(
        checkpoint, tmp_path, {"groot_n1_7_vlm_encode_v1": letterbox}, {"input_features": cameras}
    )
    GrootExportRecipe(_export_config(config, tmp_path, seed=seed))


def _with_saved_processor(checkpoint, tmp_path, processor, policy=None):
    """A copy of `checkpoint` whose saved preprocessor steps take `processor`'s settings: its config, with
    `policy`'s settings."""
    import json
    import shutil

    copied = shutil.copytree(checkpoint, tmp_path / "checkpoint")
    preprocessor_file = copied / "policy_preprocessor.json"
    saved = json.loads(preprocessor_file.read_text())
    for step in saved["steps"]:
        step["config"].update(processor.get(step["registry_name"], {}))
    preprocessor_file.write_text(json.dumps(saved))
    config = GrootConfig.from_pretrained(copied)
    for name, value in (policy or {}).items():
        value = {**config.input_features, **value} if name == "input_features" else value
        setattr(config, name, value)
    return config


def _export_config(config, tmp_path, **export_args):
    return ExportConfig(
        backend=TorchTestBackend(), policy=config, output_dir=tmp_path / "bundle", task=TASK, **export_args
    )


@pytest.mark.parametrize("shift", [1.0, float("nan")])
def test_export_refuses_parts_that_compute_other_actions_than_the_policy(
    checkpoint, tmp_path, monkeypatch, shift
):
    from lerobot.policies.groot.modeling_groot import GrootPolicy

    recipe = GrootExportRecipe(_export_config(GrootConfig.from_pretrained(checkpoint), tmp_path))
    list(recipe.parts())
    predict = GrootPolicy.predict_action_chunk
    monkeypatch.setattr(
        GrootPolicy, "predict_action_chunk", lambda *args, **kwargs: predict(*args, **kwargs) + shift
    )
    with pytest.raises(ValueError, match="compute other actions than GR00T's predict_action_chunk"):
        recipe.test_case()


@pytest.mark.parametrize("formalize", [True, False])
def test_the_bundles_prompt_ops_give_the_processors_tokens(checkpoint, tmp_path, formalize):
    """The export checks its prompt ops against the processor on the known tasks and a few edge cases."""
    config = _with_saved_processor(
        checkpoint, tmp_path, {"groot_n1_7_pack_inputs_v1": {"formalize_language": formalize}}
    )
    recipe = GrootExportRecipe(_export_config(config, tmp_path))
    assert (["lower"] in recipe._prompt_ops) == formalize


def test_export_refuses_prompt_ops_that_give_other_tokens(checkpoint, tmp_path, monkeypatch):
    from lerobot.policies.groot import export_groot

    # The processor of this checkpoint lowers the task's case first.
    monkeypatch.setattr(
        export_groot, "_prompt_ops", lambda formalize: [["template", export_groot.TASK_TEMPLATE]]
    )
    with pytest.raises(ValueError, match=f"turns the task '{TASK}' into other tokens than the bundle's"):
        GrootExportRecipe(_export_config(GrootConfig.from_pretrained(checkpoint), tmp_path))


@pytest.mark.parametrize(
    "pixel_shift, state_shift, error",
    [
        (0.0, 1e-3, "state differs from the saved processor's by 0.001"),
        (0.0, float("nan"), "state differs from the saved processor's by nan"),
        (float("nan"), 0.0, "resized images differ from the saved processor's by nan levels"),
        # Dozens of levels, which a tolerance far above two would let through.
        (1.0, 0.0, r"resized images differ from the saved processor's by [\d.]+ levels, more than 2\.0"),
    ],
)
def test_export_refuses_images_or_a_state_the_saved_processor_does_not_make(
    checkpoint, tmp_path, monkeypatch, pixel_shift, state_shift, error
):
    from lerobot.policies.groot.export_groot import _ImagesAndState

    images_and_state = _ImagesAndState.forward

    def shifted(self, *frame):
        pixels, state = images_and_state(self, *frame)
        return pixels + pixel_shift, state + state_shift

    monkeypatch.setattr(_ImagesAndState, "forward", shifted)
    with pytest.raises(ValueError, match=error):
        GrootExportRecipe(_export_config(GrootConfig.from_pretrained(checkpoint), tmp_path))


def test_a_subclass_sets_the_bfloat16_tolerance(checkpoint, tmp_path):
    class LooserRecipe(GrootExportRecipe):
        bfloat16_tolerance = 0.2

    config = GrootConfig.from_pretrained(checkpoint)
    config.use_bf16 = True
    assert LooserRecipe(_export_config(config, tmp_path)).tolerance == 0.2


JOINTS = "observation.joints"


def _dataset_that_names_the_state_joints(root):
    return _tiny_dataset(root, JOINTS, [TASK])


def _tiny_dataset(root, state, tasks, depth=()):
    """The metadata of a dataset of the torchvision cameras, the state under the name `state`, and `tasks`.

    The cameras `depth` are marked as depth maps.
    """
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets.io_utils import write_info

    image = {"dtype": "image", "names": ["height", "width", "channels"]}
    features = {
        **{
            name: {
                **image,
                "shape": (h, w, 1 if name in depth else c),
                "info": {"is_depth_map": name in depth},
            }
            for name, (c, h, w) in MODES["torchvision"][1].items()
        },
        state: {"dtype": "float32", "shape": (5,), "names": None},
        ACTION: {"dtype": "float32", "shape": (5,), "names": [f"joint_{i}" for i in range(5)]},
    }
    metadata = LeRobotDatasetMetadata.create(
        "user/tiny", fps=10, features=features, root=root, use_videos=False
    )
    if tasks:
        metadata.save_episode_tasks(tasks)
        metadata.info.total_tasks = len(tasks)
        write_info(metadata.info, root)
    return metadata


@pytest.mark.dataset
@pytest.mark.parametrize("tasks", [[], [TASK, "pick up the red cube then stop " * 3]])
def test_export_keeps_the_dataset_tasks_that_fit_the_token_budget(checkpoint, tmp_path, tasks):
    """A dataset without episodes has no tasks yet, and a task longer than --max_task_tokens is left out."""
    from lerobot.configs.default import DatasetConfig

    root = tmp_path / "dataset"
    _tiny_dataset(root, OBS_STATE, tasks)
    cfg = _export_config(GrootConfig.from_pretrained(checkpoint), tmp_path, max_task_tokens=12)
    cfg.dataset = DatasetConfig(repo_id="user/tiny", root=str(root))
    assert list(GrootExportRecipe(cfg)._prompt["known_tasks"]) == [TASK]


@pytest.mark.dataset
def test_export_refuses_a_camera_the_dataset_marks_as_a_depth_map(checkpoint, tmp_path):
    from lerobot.configs.default import DatasetConfig

    root = tmp_path / "dataset"
    camera = next(iter(MODES["torchvision"][1]))
    _tiny_dataset(root, OBS_STATE, [TASK], depth=[camera])
    cfg = _export_config(GrootConfig.from_pretrained(checkpoint), tmp_path)
    cfg.dataset = DatasetConfig(repo_id="user/tiny", root=str(root))
    with pytest.raises(ValueError, match=rf"reads the depth maps \['{camera}'\]"):
        GrootExportRecipe(cfg)


@pytest.mark.dataset
def test_export_takes_a_depth_camera_the_checkpoint_does_not_read(checkpoint, tmp_path):
    from lerobot.configs.default import DatasetConfig

    root = tmp_path / "dataset"
    _tiny_dataset(root, OBS_STATE, [TASK], depth=["observation.images.wrist"])
    pack = {"groot_n1_7_pack_inputs_v1": {"video_modality_keys": ["front"]}}
    cfg = _export_config(_with_saved_processor(checkpoint, tmp_path, pack), tmp_path)
    cfg.dataset = DatasetConfig(repo_id="user/tiny", root=str(root))
    assert "observation.images.wrist" not in GrootExportRecipe(cfg)._frame


@pytest.mark.dataset
def test_export_refuses_a_dataset_without_the_state(checkpoint, tmp_path):
    """The checkpoint's preprocessor saved no rename from the dataset's `observation.joints`."""
    from lerobot.configs.default import DatasetConfig

    root = tmp_path / "dataset"
    _dataset_that_names_the_state_joints(root)
    cfg = _export_config(GrootConfig.from_pretrained(checkpoint), tmp_path)
    cfg.dataset = DatasetConfig(repo_id="user/tiny", root=str(root))
    with pytest.raises(
        ValueError, match=r"The dataset gives nothing the policy reads as \['observation.state'\]"
    ):
        GrootExportRecipe(cfg)


@pytest.mark.dataset
def test_export_takes_the_state_under_the_name_the_saved_preprocessor_renames(tmp_path, capfd):
    """The dataset names the state `observation.joints`, which the checkpoint's preprocessor renames."""
    from lerobot.processor import RenameObservationsProcessorStep

    root = tmp_path / "dataset"
    metadata = _dataset_that_names_the_state_joints(root)
    # With relative actions, the actions part reads the state too.
    checkpoint = save_checkpoint(tmp_path, "torchvision", dataset_meta=metadata, use_relative_actions=True)
    config = GrootConfig.from_pretrained(checkpoint)
    preprocessor, _ = make_pre_post_processors(config, checkpoint)
    preprocessor.steps.insert(0, RenameObservationsProcessorStep(rename_map={JOINTS: OBS_STATE}))
    preprocessor.save_pretrained(checkpoint)
    bundle = tmp_path / "bundle"
    export(
        [
            "--backend.type=torch_test",
            f"--policy.path={checkpoint}",
            f"--output_dir={bundle}",
            "--dataset.repo_id=user/tiny",
            f"--dataset.root={root}",
            "--tolerance=1e-4",
        ]
    )
    assert "Test case against the whole policy" in capfd.readouterr().err
    runs = Bundle.read(bundle).manifest.runs
    assert [run.name for run in runs if JOINTS in run.inputs] == ["vision", "actions"]
    assert not any(OBS_STATE in run.inputs for run in runs)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "mode,height,width,geometry",
    [
        (
            "albumentations",
            32,
            32,
            {"image_target_size": [16, 16], "shortest_image_edge": 16, "crop_fraction": 0.75},
        ),
        (
            "albumentations",
            24,
            40,
            {"image_target_size": [20, 20], "crop_fraction": 0.9, "letter_box_transform": True},
        ),
        ("torchvision", 24, 40, {"image_target_size": [16, 16], "image_crop_size": [12, 12]}),
        (
            "torchvision",
            30,
            20,
            {"image_target_size": [24, 24], "shortest_image_edge": 28, "letter_box_transform": True},
        ),
    ],
)
def test_image_stages_reproduce_the_processor_pixels(mode, height, width, geometry):
    """The matrices, rounded per stage, give the processor's own 8-bit pixels within one level. torchvision's
    8-bit CPU kernel, which rounds and clamps between its passes, is within two, or 12 next to a letterbox."""
    encode = GrootN17VLMEncodeStep(use_albumentations=mode == "albumentations", **geometry)
    stages = image_stages(encode, height, width)
    rng = np.random.default_rng(0)
    for _ in range(3):
        image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        args = {
            key: getattr(encode, key)
            for key in (
                "image_crop_size",
                "image_target_size",
                "shortest_image_edge",
                "crop_fraction",
                "letter_box_transform",
            )
        }
        if mode == "albumentations":
            expected = _transform_n1_7_image_for_vlm_albumentations(image, **args).astype(np.float64)
        else:
            chw = torch.from_numpy(image).permute(2, 0, 1).contiguous()
            expected = (
                _transform_n1_7_image_for_vlm_torch(chw, **args).permute(1, 2, 0).numpy().astype(np.float64)
            )
        actual = torch.from_numpy(image).permute(2, 0, 1).float()
        for rows, columns in stages:
            actual = (rows @ actual @ columns.T).clamp(0, 255).round()
        actual = actual.permute(1, 2, 0).numpy()
        assert actual.shape == expected.shape
        if mode == "albumentations":
            assert np.abs(actual - expected).max() <= 1
        else:
            # torchvision's 8-bit CPU kernel also rounds between its two passes, and clamps there, which next to
            # a letterbox's black bars moves pixels by more, 10 levels on these images; a wrong matrix moves
            # them by hundreds. Its GPU path rounds once, as the program does.
            assert np.abs(actual - expected).max() <= (12 if encode.letter_box_transform else 2)
        if mode == "torchvision" and torch.cuda.is_available():
            chw = torch.from_numpy(image).permute(2, 0, 1).contiguous().cuda()
            on_gpu = _transform_n1_7_image_for_vlm_torch(chw, **args).permute(1, 2, 0).cpu().numpy()
            assert np.abs(actual - on_gpu).max() <= 1


def test_resize_matrices_are_read_off_a_few_lines_at_a_time():
    """All the lines of a full-size camera at once would take gigabytes."""
    from lerobot.policies.groot.export_groot import LINE_BATCH_PIXELS, _separable_map

    rng = np.random.default_rng(0)
    height, width = 320, 320
    rows, columns = rng.random((16, height)), rng.random((12, width))
    sizes = []

    def transform(lines):
        sizes.append(lines.size)
        return (rows @ lines @ columns.T).astype(np.float32)

    read_rows, read_columns = _separable_map(transform, (height, width))
    assert max(sizes) <= LINE_BATCH_PIXELS < height * height * width
    # An exporter warns about the strides of a matrix that is not contiguous.
    assert read_rows.is_contiguous() and read_columns.is_contiguous()
    image = rng.random((height, width))
    actual = read_rows.double() @ torch.from_numpy(image) @ read_columns.double().T
    np.testing.assert_allclose(actual.numpy(), rows @ image @ columns.T, rtol=1e-4)


@pytest.mark.parametrize("joints, horizon", [("joints", 4), ("arm.joints", 4), ("joints", 2)])
def test_n17_action_decode_matches_the_processor_step(joints, horizon):
    """Unnormalizing, a relative joint group, a relative end-effector pose, and the LIBERO gripper.

    The statistics per chunk step hold more steps than the chunk decodes, the chunk may hold more steps than
    the action horizon, and the gripper spans both sides of the LIBERO threshold, 0.5.
    """
    from lerobot.policies.groot.export_groot import N17ActionDecode
    from lerobot.policies.groot.processor_groot import GrootN17ActionDecodeStep, GrootN17PackInputsStep
    from lerobot.processor import (
        PolicyProcessorPipeline,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    rng = np.random.default_rng(0)

    def stats(dim, horizon=None):
        shape = (horizon, dim) if horizon else (dim,)
        low = rng.uniform(-2, 0, shape)
        return {
            "min": low.tolist(),
            "max": (low + rng.uniform(1, 3, shape)).tolist(),
            "q01": (low + 0.1).tolist(),
            "q99": (low + 0.9).tolist(),
            "mean": [0.0] * dim,
        }

    keys, dims = ["eef", joints, "gripper"], [9, 2, 1]
    raw_stats = {
        "action": {key: stats(dim) for key, dim in zip(keys, dims, strict=True)},
        "relative_action": {"eef": stats(9, 6), joints: stats(2, 6)},
        "state": {"eef": stats(9), joints: stats(2)},
    }
    raw_stats["action"]["gripper"].update({"q01": [0.0], "q99": [1.0]})
    modality_config = {
        "action": {
            "modality_keys": keys,
            "delta_indices": list(range(horizon)),
            "action_configs": [
                {"rep": "relative", "type": "eef", "format": "xyz+rot6d"},
                {"rep": "relative", "type": "non_eef"},
                {"rep": "absolute", "type": "non_eef"},
            ],
        },
        "state": {"modality_keys": ["eef", joints]},
    }
    pack = GrootN17PackInputsStep(raw_stats=raw_stats, modality_config=modality_config)
    decode = GrootN17ActionDecodeStep(
        env_action_dim=12,
        raw_stats=raw_stats,
        modality_config=modality_config,
        use_percentiles=True,
        use_relative_action=True,
        action_decode_transform="libero",
        pack_step=pack,
    )
    state = rng.standard_normal(11).astype(np.float32)
    pack._last_raw_state = {"eef": state[None, :9], joints: state[None, 9:]}
    actions = torch.from_numpy(rng.uniform(-1.2, 1.2, (1, 4, 16)).astype(np.float32))
    # The gripper decodes to 0.25, 0.5, 0.75 and 1: below, at and above the threshold.
    actions[0, :, 11] = torch.tensor([-0.5, 0.0, 0.5, 1.0])
    pipeline = PolicyProcessorPipeline(
        steps=[decode], to_transition=policy_action_to_transition, to_output=transition_to_policy_action
    )
    expected = pipeline(actions.clone())
    actual = N17ActionDecode(decode, pack, 4)(actions, torch.from_numpy(state)[None])
    torch.testing.assert_close(actual, expected.float(), rtol=0, atol=1e-5)


def test_n17_action_decode_sizes_the_state_groups_as_the_packing_step():
    """Statistics saved as arrays, or with a row per chunk step, size a group by their last axis."""
    from lerobot.policies.groot.export_groot import N17ActionDecode
    from lerobot.policies.groot.processor_groot import GrootN17ActionDecodeStep, GrootN17PackInputsStep

    stats = {"min": [-1.0] * 2, "max": [1.0] * 2, "mean": [0.0] * 2}
    raw_stats = {
        "action": {"arm": stats},
        "relative_action": {"arm": stats},
        "state": {"eef": {"min": np.zeros(9), "max": np.ones(9)}, "arm": {"min": [[0.0, 0.0]] * 3}},
    }
    modality_config = {
        "action": {
            "modality_keys": ["arm"],
            "delta_indices": [0],
            "action_configs": [{"rep": "relative", "type": "non_eef"}],
        },
        "state": {"modality_keys": ["eef", "arm"]},
    }
    pack = GrootN17PackInputsStep(raw_stats=raw_stats, modality_config=modality_config)
    decode = GrootN17ActionDecodeStep(
        raw_stats=raw_stats, modality_config=modality_config, use_relative_action=True, pack_step=pack
    )
    assert N17ActionDecode(decode, pack, 1).state_groups == {"eef": (0, 9), "arm": (9, 2)}


@pytest.mark.parametrize("shift", [1.0, float("nan")])
def test_n17_decode_check_refuses_actions_the_processor_step_does_not_make(shift):
    from lerobot.processor import (
        PolicyProcessorPipeline,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    recipe = object.__new__(GrootExportRecipe)
    recipe._preprocess = lambda frame, task: None
    recipe._frame, recipe.task, recipe._steps, recipe._action_dim = {}, TASK, 4, 3
    # An empty postprocessor returns the chunk as it is, so the decoded test chunk must be zeros.
    recipe.postprocessor = PolicyProcessorPipeline(
        steps=[], to_transition=policy_action_to_transition, to_output=transition_to_policy_action
    )
    recipe._actions = np.full((4, 3), shift, np.float32)
    recipe._action_scale = lambda: np.ones(3)
    with pytest.raises(ValueError, match="N17ActionDecode moved the test chunk's actions"):
        recipe._check_n17_decode(torch.zeros(4, 6))


@pytest.mark.parametrize("widening", [False, True])
def test_action_scale_of_a_relative_pose_comes_from_its_statistics(widening):
    """The state's rotation turns the xyz, so they share their largest scale, and a rotation is made of unit
    vectors, so its scale is 1. With statistics per chunk step, the widest step sets the scale."""
    from lerobot.policies.groot.export_groot import N17ActionDecode, _N17ChunkActions
    from lerobot.policies.groot.processor_groot import GrootN17ActionDecodeStep, GrootN17PackInputsStep

    # x, y and z span 2, 4 and 6, so their scales are 1, 2 and 3.
    spans = [1.0, 2.0, 3.0] + [1.0] * 6
    stats = {"min": [-a for a in spans], "max": spans, "mean": [0.0] * 9}
    # Step k spans 1 + k times as much.
    per_step = {
        "min": [[-a * (1 + k) for a in spans] for k in range(4)],
        "max": [[a * (1 + k) for a in spans] for k in range(4)],
    }
    raw_stats = {
        "action": {"eef": stats},
        "relative_action": {"eef": per_step if widening else stats},
        "state": {"eef": stats},
    }
    modality_config = {
        "action": {
            "modality_keys": ["eef"],
            "delta_indices": list(range(4)),
            "action_configs": [{"rep": "relative", "type": "eef", "format": "xyz+rot6d"}],
        },
        "state": {"modality_keys": ["eef"]},
    }
    pack = GrootN17PackInputsStep(raw_stats=raw_stats, modality_config=modality_config)
    decode = GrootN17ActionDecodeStep(
        env_action_dim=9,
        raw_stats=raw_stats,
        modality_config=modality_config,
        use_percentiles=False,
        use_relative_action=True,
        pack_step=pack,
    )
    recipe = object.__new__(GrootExportRecipe)
    recipe._chunk_actions = _N17ChunkActions(4, 9, N17ActionDecode(decode, pack, 4))
    xyz = 3.0 * (4 if widening else 1)
    np.testing.assert_allclose(recipe._action_scale(), [xyz] * 3 + [1.0] * 6, rtol=1e-6)


def test_action_scale_is_that_of_the_widest_chunk_step():
    """Actions normalized with statistics per chunk step can spread more at later steps."""
    from types import SimpleNamespace

    recipe = object.__new__(GrootExportRecipe)
    recipe._model_config = SimpleNamespace(max_action_dim=3)
    recipe._frame = {OBS_STATE: np.zeros(3, np.float32)}
    recipe._images = SimpleNamespace(state_name=OBS_STATE)
    recipe._device = torch.device("cpu")
    recipe._steps = 4
    recipe._chunk_actions = lambda actions, state: actions * torch.arange(1.0, 5.0)[:, None]
    np.testing.assert_allclose(recipe._action_scale(), [4.0] * 3)


def test_rope_frequencies_are_the_exact_ones_or_refused():
    """A bfloat16 policy rounds its rope frequencies, and the bundle uses the exact ones; others are refused."""
    from lerobot.policies.groot.export_groot import _exact_inv_freq

    exact = 1.0 / 5e6 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16)
    rounded = exact.to(torch.bfloat16).float()
    assert not torch.equal(rounded, exact)
    assert torch.equal(_exact_inv_freq(rounded, exact), exact)
    with pytest.raises(ValueError, match="rope frequencies are not the ones"):
        _exact_inv_freq(exact * 1.1, exact)


def test_export_refuses_another_rope_or_attention():
    from types import SimpleNamespace

    from diffusers.models.attention_processor import Attention, AttnProcessor

    from lerobot.policies.groot.export_groot import _check_cross_attention, _text_rotary

    with pytest.raises(ValueError, match="computes the default rope, not 'yarn'"):
        _text_rotary(SimpleNamespace(rope_type="yarn"), torch.zeros(3, 1, 4))
    _check_cross_attention(Attention(query_dim=8, heads=2, dim_head=4))
    for other in (
        {"qk_norm": "layer_norm"},
        {"residual_connection": True},
        {"cross_attention_norm": "layer_norm", "cross_attention_dim": 8},
        {"processor": AttnProcessor()},
    ):
        with pytest.raises(ValueError, match="Only diffusers' default attention processor"):
            _check_cross_attention(Attention(query_dim=8, heads=2, dim_head=4, **other))


@pytest.mark.parametrize(
    "action_config, transform, stats_change, error",
    [
        (
            {"rep": "relative", "type": "eef", "format": "xyz+quat"},
            None,
            {},
            "Unsupported relative N1.7 action",
        ),
        ({"rep": "absolute", "type": "non_eef"}, "libero", {}, "needs a scalar gripper action group"),
        ({"rep": "absolute", "type": "non_eef"}, "mirror", {}, "Unsupported N1.7 action decode transform"),
        ({"rep": "absolute", "type": "non_eef"}, None, {"action": {}}, "no action group with statistics"),
        ({"rep": "relative", "type": "non_eef", "state_key": "arm"}, None, {}, r"state groups \['arm'\]"),
        ({"rep": "absolute", "type": "non_eef"}, None, None, "no statistics or modality config"),
    ],
)
def test_n17_action_decode_refuses_what_it_does_not_reproduce(action_config, transform, stats_change, error):
    from lerobot.policies.groot.export_groot import N17ActionDecode
    from lerobot.policies.groot.processor_groot import GrootN17ActionDecodeStep, GrootN17PackInputsStep

    stats = {"min": [-1.0] * 9, "max": [1.0] * 9, "mean": [0.0] * 9}
    raw_stats = {"action": {"eef": stats}, "relative_action": {"eef": stats}, "state": {"eef": stats}}
    modality_config = {
        "action": {"modality_keys": ["eef"], "delta_indices": [0], "action_configs": [action_config]},
        "state": {"modality_keys": ["eef"]},
    }
    pack = GrootN17PackInputsStep(raw_stats=raw_stats, modality_config=modality_config)
    decode = GrootN17ActionDecodeStep(
        raw_stats=raw_stats, modality_config=modality_config, use_relative_action=True, pack_step=pack
    )
    decode.action_decode_transform = transform
    decode.raw_stats = None if stats_change is None else {**raw_stats, **stats_change}
    with pytest.raises(ValueError, match=error):
        N17ActionDecode(decode, pack, 1)


def test_export_refuses_a_processor_it_does_not_reproduce():
    from lerobot.policies.groot.export_groot import _chunk_actions
    from lerobot.policies.groot.processor_groot import GrootN17ActionDecodeStep, GrootN17PackInputsStep
    from lerobot.processor import PolicyProcessorPipeline, RenameObservationsProcessorStep

    other = RenameObservationsProcessorStep(rename_map={})
    preprocessor = PolicyProcessorPipeline(steps=[other])
    with pytest.raises(ValueError, match="no GrootN17PackInputsStep, so it is not a GR00T N1.7 one"):
        _chunk_actions(4, 5, preprocessor, PolicyProcessorPipeline(steps=[GrootN17ActionDecodeStep()]))
    preprocessor = PolicyProcessorPipeline(steps=[GrootN17PackInputsStep()])
    with pytest.raises(ValueError, match="must be the postprocessor's only step"):
        _chunk_actions(4, 5, preprocessor, PolicyProcessorPipeline(steps=[GrootN17ActionDecodeStep(), other]))


def test_chunk_actions_add_the_state_the_program_takes():
    """GR00T's generic relative fallback: the actions part adds the robot state it takes, as the postprocessor
    adds the state its preprocessor cached, and keeps the gripper absolute."""
    from lerobot.policies.groot.export_groot import _ChunkActions
    from lerobot.processor import (
        AbsoluteActionsProcessorStep,
        PolicyProcessorPipeline,
        RelativeActionsProcessorStep,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    relative = RelativeActionsProcessorStep(
        enabled=True, exclude_joints=["gripper"], action_names=["shoulder", "elbow", "gripper"]
    )
    postprocessor = PolicyProcessorPipeline(
        steps=[AbsoluteActionsProcessorStep(enabled=True, relative_step=relative)],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    state = torch.tensor([0.5, -1.0, 2.0])
    relative._last_state = state[None]
    actions = torch.randn(6, 4)
    chunk_actions = _ChunkActions(4, 3, postprocessor)
    assert chunk_actions.reads_state
    expected = actions[:4, :3] + torch.tensor([0.5, -1.0, 0.0])
    torch.testing.assert_close(chunk_actions(actions, state), expected)
    torch.testing.assert_close(postprocessor(actions[None, :4, :3].clone())[0], expected)


def _vision_tower(deepstack):
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    vision = {"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 32}
    vision |= {
        "patch_size": 4,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "num_position_embeddings": 16,
    }
    torch.manual_seed(0)
    return Qwen3VLVisionModel(
        Qwen3VLConfig(vision_config={**vision, "deepstack_visual_indexes": deepstack}).vision_config
    )


@pytest.mark.parametrize("deepstack", [[], [0, 1]])
def test_vision_tower_computes_what_the_towers_own_forward_does(deepstack):
    """One output per deepstack layer, equal to the tower's own forward."""
    from lerobot.policies.groot.export_groot import _VisionTower

    tower = _vision_tower(deepstack).eval()
    grid = torch.tensor([[1, 4, 4]])
    pixels = torch.randn(16, 3 * 2 * 4 * 4)
    with torch.no_grad():
        # Before the export's tower is built, as that sets the rotary frequencies the tower shares.
        expected = tower(pixels, grid_thw=grid)
        features = _VisionTower(tower, grid)(pixels)
    torch.testing.assert_close(
        features, (expected.pooler_output, *expected.deepstack_features), rtol=0, atol=0
    )


def test_vision_tower_keeps_its_residual_stream_in_float32_under_bfloat16_autocast():
    """Not the tower's own forward: transformers 5.9 and later round the sum of the patch and positional
    embeddings to bfloat16 there, while the export adds them in float32, as NVIDIA's GR00T and older
    transformers do. So the export equals the tower with a float32 patch embedding, not the tower itself."""
    from lerobot.policies.groot.export_groot import _VisionTower

    tower = _vision_tower([0, 1]).eval()
    grid = torch.tensor([[1, 4, 4]])
    pixels = torch.randn(16, 3 * 2 * 4 * 4)
    with torch.no_grad(), torch.autocast("cpu", torch.bfloat16):
        rounded = tower(pixels, grid_thw=grid).pooler_output
        hook = tower.patch_embed.register_forward_hook(lambda module, args, output: output.float())
        widened = tower(pixels, grid_thw=grid)
        hook.remove()
        features = _VisionTower(tower, grid)(pixels)
    torch.testing.assert_close(features, (widened.pooler_output, *widened.deepstack_features), rtol=0, atol=0)
    assert not torch.equal(features[0], rounded)
