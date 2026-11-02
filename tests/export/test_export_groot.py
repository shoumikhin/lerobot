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
pytest.importorskip("cv2")

from lerobot.bundle.manifest import Bundle  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.groot.configuration_groot import GrootConfig  # noqa: E402
from lerobot.policies.groot.export_groot import image_stages  # noqa: E402
from lerobot.policies.groot.processor_groot import (  # noqa: E402
    GrootN17VLMEncodeStep,
    _transform_n1_7_image_for_vlm_albumentations,
    _transform_n1_7_image_for_vlm_torch,
)
from lerobot.scripts.lerobot_export import export  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from tests.export.plugins import use_plugins  # noqa: E402
from tests.export.tiny_groot import MODES, TASK, save_checkpoint  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def torch_test_backend():
    """Add the test backend to the export's steps, which run in processes of their own."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        use_plugins(monkeypatch, "torch_backend")
        yield


def test_a_bfloat16_policy_without_autocast_exports(tmp_path, capfd):
    """With autocast off, the parts feed the policy's bfloat16 layers bfloat16 values, as its own forward does."""
    checkpoint = save_checkpoint(tmp_path, "albumentations", dtype=torch.bfloat16)
    export(
        [
            "--backend.type=torch_test",
            f"--policy.path={checkpoint}",
            f"--output_dir={tmp_path / 'bundle'}",
            f"--task={TASK}",
        ]
    )
    assert "Policy check: the chained parts are" in capfd.readouterr().err


@pytest.mark.dataset
def test_export_takes_the_state_under_the_name_the_saved_preprocessor_renames(tmp_path):
    """The dataset names the state `observation.joints`, which the checkpoint's preprocessor renames."""
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets.io_utils import write_info
    from lerobot.processor import RenameObservationsProcessorStep

    joints = "observation.joints"
    image = {"dtype": "image", "names": ["height", "width", "channels"]}
    features = {
        **{name: {**image, "shape": (h, w, c)} for name, (c, h, w) in MODES["torchvision"][1].items()},
        joints: {"dtype": "float32", "shape": (5,), "names": None},
        ACTION: {"dtype": "float32", "shape": (5,), "names": [f"joint_{i}" for i in range(5)]},
    }
    root = tmp_path / "dataset"
    metadata = LeRobotDatasetMetadata.create(
        "user/tiny", fps=10, features=features, root=root, use_videos=False
    )
    metadata.save_episode_tasks([TASK])
    metadata.info.total_tasks = 1
    write_info(metadata.info, root)
    # With relative actions, the actions part reads the state too.
    checkpoint = save_checkpoint(tmp_path, "torchvision", dataset_meta=metadata, use_relative_actions=True)
    config = GrootConfig.from_pretrained(checkpoint)
    preprocessor, _ = make_pre_post_processors(config, checkpoint)
    preprocessor.steps.insert(0, RenameObservationsProcessorStep(rename_map={joints: OBS_STATE}))
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
    runs = Bundle.read(bundle).manifest.runs
    assert [run.name for run in runs if joints in run.inputs] == ["vision", "actions"]
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
    """The matrices, rounded per stage, give the processor's own 8-bit pixels within one level."""
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
        elif not encode.letter_box_transform:
            # torchvision's 8-bit CPU kernel also rounds between its two passes, and clamps there, which next to
            # a letterbox's black bars moves pixels by more. Its GPU path rounds once, as the program does.
            assert np.abs(actual - expected).max() <= 2
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
    image = rng.random((height, width))
    actual = read_rows.double() @ torch.from_numpy(image) @ read_columns.double().T
    np.testing.assert_allclose(actual.numpy(), rows @ image @ columns.T, rtol=1e-4)


def test_n17_action_decode_matches_the_processor_step():
    """Unnormalizing, a relative joint group, a relative end-effector pose, and the LIBERO gripper."""
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

    keys, dims = ["eef", "joints", "gripper"], [9, 2, 1]
    raw_stats = {
        "action": {key: stats(dim) for key, dim in zip(keys, dims, strict=True)},
        "relative_action": {"eef": stats(9, 4), "joints": stats(2, 4)},
        "state": {"eef": stats(9), "joints": stats(2)},
    }
    modality_config = {
        "action": {
            "modality_keys": keys,
            "delta_indices": list(range(4)),
            "action_configs": [
                {"rep": "relative", "type": "eef", "format": "xyz+rot6d"},
                {"rep": "relative", "type": "non_eef"},
                {"rep": "absolute", "type": "non_eef"},
            ],
        },
        "state": {"modality_keys": ["eef", "joints"]},
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
    pack._last_raw_state = {"eef": state[None, :9], "joints": state[None, 9:]}
    actions = torch.from_numpy(rng.uniform(-1.2, 1.2, (1, 4, 16)).astype(np.float32))
    pipeline = PolicyProcessorPipeline(
        steps=[decode], to_transition=policy_action_to_transition, to_output=transition_to_policy_action
    )
    expected = pipeline(actions.clone())
    actual = N17ActionDecode(decode, pack, 4)(actions, torch.from_numpy(state)[None])
    torch.testing.assert_close(actual, expected.float(), rtol=0, atol=1e-5)


def test_action_scale_of_a_relative_rotation_is_one():
    """Constant actions are no rotation, so the scale of a relative end-effector rotation is not measured."""
    from types import SimpleNamespace

    from lerobot.policies.groot.export_groot import GrootExportRecipe, N17ActionDecode, _N17ChunkActions
    from lerobot.policies.groot.processor_groot import GrootN17ActionDecodeStep, GrootN17PackInputsStep

    stats = {"min": [-1.0] * 9, "max": [1.0] * 9, "mean": [0.0] * 9}
    raw_stats = {"action": {"eef": stats}, "relative_action": {"eef": stats}, "state": {"eef": stats}}
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
    recipe._model_config = SimpleNamespace(max_action_dim=9)
    recipe._frame = {OBS_STATE: np.array([0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0], np.float32)}
    recipe._images = SimpleNamespace(state_name=OBS_STATE)
    recipe._device = torch.device("cpu")
    recipe._steps = 4
    recipe._chunk_actions = _N17ChunkActions(4, 9, N17ActionDecode(decode, pack, 4))
    # The state's rotation is the identity, so the xyz moves by one per normalized unit, as its statistics say.
    np.testing.assert_allclose(recipe._action_scale(), np.ones(9, np.float32), rtol=1e-6)
