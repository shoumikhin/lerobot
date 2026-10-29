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

"""The default export recipe, on a tiny ACT built in memory."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import Tensor

import lerobot.export.recipe as recipe_module
import lerobot.policies.factory as factory
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.export.recipe import (
    ActionChunkRecipe,
    export_task,
    frame_shapes,
    measure_action_scale,
    robot_features,
)
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor import ProcessorStep, TransitionKey
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

IMAGE = f"{OBS_IMAGES}.front"
STATS = {
    IMAGE: {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.2)},
    OBS_STATE: {"mean": torch.zeros(3), "std": torch.ones(3)},
    ACTION: {"mean": torch.full((3,), 10.0), "std": torch.tensor([30.0, 3.0, 0.3])},
}


def act_config() -> ACTConfig:
    return ACTConfig(
        input_features={
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        chunk_size=4,
        n_action_steps=4,
        use_vae=False,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        pretrained_backbone_weights=None,
        device="cpu",
    )


def export_config(task: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task=task, tolerance=None
    )


def make(
    policy_class=ACTPolicy, task: str = "", extra_step: ProcessorStep | None = None
) -> ActionChunkRecipe:
    torch.manual_seed(0)
    config = act_config()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=STATS)
    if extra_step is not None:
        preprocessor.steps.insert(0, extra_step)
    return ActionChunkRecipe(policy_class(config).eval(), preprocessor, postprocessor, export_config(task))


class TaskCodes(ProcessorStep):
    """Reads the task like a tokenizer that keeps only its first `limit` characters."""

    limit = 4

    def __call__(self, transition):
        task = transition[TransitionKey.COMPLEMENTARY_DATA]["task"]
        codes = [ord(c) for c in task[: self.limit].ljust(self.limit)]
        observation = {**transition[TransitionKey.OBSERVATION], "observation.task_codes": torch.tensor(codes)}
        return {**transition, TransitionKey.OBSERVATION: observation}

    def transform_features(self, features):
        return features


class NoisyACT(ACTPolicy):
    """ACT that draws a starting noise of its own shape, (chunk_size, 5), and adds its first column."""

    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = torch.randn(1, self.config.chunk_size, 5)
        return super().predict_action_chunk(batch) + noise[..., :1]


class TwoNoisesACT(ACTPolicy):
    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        return super().predict_action_chunk(batch) + torch.randn(1, 4, 1) + torch.randn(1, 4, 1)


def test_recipe_measures_the_action_scale_from_the_postprocessor():
    np.testing.assert_allclose(make().test_case().action_scale, [30.0, 3.0, 0.3], rtol=1e-6)


def test_recipe_takes_the_tolerance_and_the_seed_it_is_given():
    states = []
    for seed in (0, 7):
        torch.manual_seed(0)
        config = act_config()
        cfg = export_config()
        cfg.seed, cfg.tolerance = seed, 0.01
        recipe = ActionChunkRecipe(
            ACTPolicy(config).eval(), *make_pre_post_processors(config, dataset_stats=STATS), cfg
        )
        assert recipe.tolerance == 0.01
        states.append(recipe.test_case().frame[OBS_STATE])
    assert not np.array_equal(*states)


def test_recipe_keeps_a_task_given_even_when_a_tokenizer_cuts_it_short():
    recipe = make(task="pick cube pick cube", extra_step=TaskCodes())
    assert recipe.task == "pick cube pick cube"
    with pytest.raises(ValueError, match="reads the task"):
        make(extra_step=TaskCodes())
    assert make().task is None


def test_the_dataset_is_found_where_training_finds_it(monkeypatch):
    import lerobot.datasets
    import lerobot.datasets.storage
    from lerobot.export.recipe import dataset_metadata

    loaded = []
    load = lambda *args, **kwargs: loaded.append((args, kwargs))  # noqa: E731
    monkeypatch.setattr(lerobot.datasets.storage, "load_dataset_metadata", load)
    monkeypatch.setattr(lerobot.datasets, "LeRobotDatasetMetadata", None)
    root = "hf://buckets/user/data"
    dataset = SimpleNamespace(repo_id="user/data", root=root, revision="v1", repo_type="bucket")
    dataset_metadata(SimpleNamespace(dataset=dataset))
    assert loaded == [(("user/data",), {"root": root, "revision": "v1", "repo_type": "bucket"})]


def test_recipe_asks_for_the_task_of_a_dataset_without_tasks():
    with pytest.raises(ValueError, match="its dataset has none: give the one to export with --task"):
        export_task(export_config(), SimpleNamespace(tasks=None), "act")


def test_a_joint_that_never_moved_gets_a_small_action_scale_and_not_zero():
    # Still at 5 and at -5, and one that moves.
    scale = measure_action_scale(
        lambda value: torch.tensor([5.0, -5.0, 5.0]) + torch.tensor([0.0, 0.0, 30.0]) * value
    )
    np.testing.assert_allclose(scale, [1e-4 * 5.0 + 1e-6, 1e-4 * 5.0 + 1e-6, 30.0], rtol=1e-6)


def test_a_part_traces_its_inputs_contiguous():
    from lerobot.export.recipe import Part

    part = Part("p", torch.nn.Identity(), (torch.arange(6.0).view(2, 3).T,), ["a"], ["y"])
    assert part.example_inputs[0].is_contiguous()
    torch.testing.assert_close(part.example_inputs[0], torch.arange(6.0).view(2, 3).T)


def test_a_part_refuses_one_tensor_for_two_inputs():
    from lerobot.export.recipe import Part

    x = torch.zeros(2)
    with pytest.raises(ValueError, match="gives one tensor for its inputs a and b"):
        Part("p", torch.nn.Identity(), (x, torch.ones(2), x), ["a", "c", "b"], ["y"])
    # Views of one tensor are separate inputs.
    Part("p", torch.nn.Identity(), tuple(torch.zeros(2, 2).unbind(0)), ["a", "b"], ["y"])


def test_recipe_asks_the_policy_for_its_noise_shape():
    recipe = make(NoisyACT)
    assert recipe.test_case().noise.shape == (4, 5)
    [part] = recipe.parts()
    assert part.inputs[-1] == "noise"


def test_recipe_refuses_a_policy_that_draws_two_noises():
    with pytest.raises(ValueError, match="draws 2 random tensors"):
        make(TwoNoisesACT)


class UniformDrawACT(ACTPolicy):
    """ACT that adds a uniform draw of its own, so each chunk it computes is another one."""

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        return super().predict_action_chunk(batch) + torch.rand(1)


def test_recipe_refuses_a_policy_that_computes_another_chunk_each_time():
    with pytest.raises(ValueError, match="computes another chunk from the same frame and noise"):
        make(UniformDrawACT)


class NoChunkSpecACT(ACTPolicy):
    def chunk_inference_spec(self):
        raise ValueError("n_action_steps is larger than chunk_size")


def test_recipe_refuses_a_checkpoint_whose_chunk_spec_fails():
    with pytest.raises(ValueError, match="checkpoint cannot be exported: n_action_steps is larger"):
        make(NoChunkSpecACT)


def test_recipe_does_not_ask_the_dataset_for_empty_camera_slots():
    empty = f"{OBS_IMAGES}.empty_camera_0"
    config = SmolVLAConfig(
        empty_cameras=1,
        input_features={
            IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            empty: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
    )
    metadata = SimpleNamespace(
        features={
            IMAGE: {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]},
            OBS_STATE: {"dtype": "float32", "shape": (3,), "names": None},
        },
    )
    for given in (metadata, None):
        features = robot_features(config, SimpleNamespace(steps=[]), given)
        assert frame_shapes(features) == {IMAGE: (32, 32, 3), OBS_STATE: (3,)}, given


def test_a_strict_load_refuses_a_peft_adapter_that_lacks_a_weight(tmp_path):
    peft = pytest.importorskip("peft")
    from safetensors.torch import load_file, save_file

    config = act_config()
    ACTPolicy(config).save_pretrained(tmp_path / "base")
    lora = peft.LoraConfig(target_modules=["out_proj"])
    adapter = peft.get_peft_model(ACTPolicy.from_pretrained(tmp_path / "base"), lora)
    adapter.peft_config["default"].base_model_name_or_path = str(tmp_path / "base")
    adapter.save_pretrained(tmp_path / "adapter")
    config.use_peft, config.pretrained_path = True, tmp_path / "adapter"
    factory.load_pretrained_policy(config, strict=True)
    weights_file = tmp_path / "adapter" / "adapter_model.safetensors"
    save_file(dict(sorted(load_file(weights_file).items())[1:]), weights_file)
    with pytest.raises(ValueError, match="lacks 1 of its weights"):
        factory.load_pretrained_policy(config, strict=True)


def test_an_image_recorded_channels_last_keeps_its_shape():
    features = {
        IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 24, 32)),
        f"{IMAGE}_hwc": PolicyFeature(type=FeatureType.VISUAL, shape=(24, 32, 3)),
    }
    assert frame_shapes(features) == {IMAGE: (24, 32, 3), f"{IMAGE}_hwc": (24, 32, 3)}


@pytest.mark.parametrize(
    ("meminfo", "pages", "spare"),
    [
        # 64 GB installed, of which Linux counts 6.1 GB available.
        ("MemTotal:       62500000 kB\nMemAvailable:    5957032 kB\n", 15_625_000, "6.1 GB available"),
        # More than the checkpoint, but not twice as much.
        ("MemTotal:       62500000 kB\nMemAvailable:    8789063 kB\n", 15_625_000, "9.0 GB available"),
        # No /proc/meminfo, as on macOS: 6.1 GB installed.
        (None, 1_489_258, "6.1 GB installed"),
    ],
)
def test_a_policy_too_big_to_compare_is_skipped_with_a_log_line(
    tmp_path, monkeypatch, caplog, meminfo, pages, spare
):
    """A 6.15 GB checkpoint on a machine with less than twice that much memory to spare."""
    # Only the checkpoint's size is read, so no file that big is made.
    model_file = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=6_150_000_000))
    meminfo_file = tmp_path / "meminfo"
    if meminfo is not None:
        meminfo_file.write_text(meminfo)
    monkeypatch.setattr(recipe_module, "_MEMINFO", meminfo_file)
    monkeypatch.setattr(
        recipe_module.os, "sysconf", {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": pages}.__getitem__, raising=False
    )
    with caplog.at_level(logging.WARNING, logger=recipe_module.__name__):
        assert not recipe_module.whole_policy_fits(model_file, torch.device("cpu"), "pi0.5")
    assert (
        f"Skipped the comparison with the whole pi0.5 policy: it needs about 12.3 GB, {spare}. The "
        "bundle's own check against its test case still runs." in caplog.text
    )


def test_a_policy_is_not_compared_on_a_machine_that_does_not_say_how_much_memory_it_has(
    tmp_path, monkeypatch, caplog
):
    """As on Windows: no /proc/meminfo and no os.sysconf."""
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"0")
    monkeypatch.setattr(recipe_module, "_MEMINFO", tmp_path / "meminfo")
    monkeypatch.delattr(recipe_module.os, "sysconf", raising=False)
    with caplog.at_level(logging.WARNING, logger=recipe_module.__name__):
        assert not recipe_module.whole_policy_fits(model_file, torch.device("cpu"), "pi0.5")
    assert "this machine does not say how much memory it has" in caplog.text


def test_a_policy_export_module_must_name_its_recipe(monkeypatch):
    # A policy without one gets the one-program recipe.
    assert factory.get_export_recipe_class(act_config()) is ActionChunkRecipe
    monkeypatch.setattr(
        factory, "_import_sibling_policy_module", lambda *args: SimpleNamespace(__name__="export_act")
    )
    with pytest.raises(ValueError, match="export_act must define the export recipe ACTExportRecipe"):
        factory.get_export_recipe_class(act_config())
