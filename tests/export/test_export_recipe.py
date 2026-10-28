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
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor import ProcessorStep, TransitionKey
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from tests.export.tiny_act import IMAGE, act_config, act_stats

STATS = act_stats(action_std=(30.0, 3.0, 0.3))


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


@pytest.mark.dataset
def test_the_dataset_is_found_where_training_finds_it(monkeypatch):
    pytest.importorskip("datasets")
    import lerobot.datasets
    import lerobot.datasets.storage
    from lerobot.export.recipe import dataset_metadata

    loaded = []
    load = lambda *args, **kwargs: loaded.append((args, kwargs))  # noqa: E731
    monkeypatch.setattr(lerobot.datasets.storage, "load_dataset_metadata", load)
    monkeypatch.setattr(lerobot.datasets, "LeRobotDatasetMetadata", None)
    root = "hf://buckets/user/data"
    dataset = SimpleNamespace(
        repo_id="user/data", root=root, revision="v1", repo_type="bucket", streaming=False
    )
    dataset_metadata(SimpleNamespace(dataset=dataset))
    # Streamed, a bucket given as the root is read in bucket mode.
    streamed = SimpleNamespace(**{**vars(dataset), "repo_type": "dataset", "streaming": True})
    dataset_metadata(SimpleNamespace(dataset=streamed))
    assert loaded == [
        (("user/data",), {"root": root, "revision": "v1", "repo_type": "bucket"}),
        (("user/data",), {"root": None, "revision": "v1", "repo_type": "bucket"}),
    ]


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


def test_a_part_refuses_inputs_without_one_name_each():
    from lerobot.export.recipe import Part

    with pytest.raises(ValueError, match=r"different number of input names \(1\) and example inputs \(2\)"):
        Part("p", torch.nn.Identity(), (torch.zeros(2), torch.ones(2)), ["a"], ["y"])


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


def test_recipe_follows_renames_in_the_order_the_preprocessor_makes_them():
    from lerobot.export.recipe import preprocessor_renames
    from lerobot.processor import RenameObservationsProcessorStep

    first, middle = f"{OBS_IMAGES}.cam", f"{OBS_IMAGES}.camera"
    steps = [
        RenameObservationsProcessorStep(rename_map={first: middle, "observation.joints": OBS_STATE}),
        RenameObservationsProcessorStep(rename_map={middle: IMAGE}),
    ]
    renames = preprocessor_renames(SimpleNamespace(steps=steps))
    assert renames == {first: IMAGE, "observation.joints": OBS_STATE, middle: IMAGE}
    metadata = SimpleNamespace(
        features={
            first: {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]},
            "observation.joints": {"dtype": "float32", "shape": (3,), "names": None},
        }
    )
    features = robot_features(act_config(), SimpleNamespace(steps=steps), metadata)
    assert sorted(features) == [first, "observation.joints"]


def test_recipe_refuses_a_depth_map_the_dataset_marks():
    depth = f"{OBS_IMAGES}.front_depth"
    config = act_config()
    config.input_features = {
        **config.input_features,
        depth: PolicyFeature(type=FeatureType.VISUAL, shape=(1, 32, 32)),
    }
    features = {
        name: {"dtype": "float32", "shape": tuple(f.shape), "names": None}
        for name, f in config.input_features.items()
    }
    features[IMAGE] = {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]}
    features[depth] = {
        "dtype": "video",
        "shape": (32, 32, 1),
        "names": ["height", "width", "channels"],
        "info": {"is_depth_map": True},
    }
    with pytest.raises(ValueError, match=r"reads the depth maps \['observation.images.front_depth'\]"):
        robot_features(config, SimpleNamespace(steps=[]), SimpleNamespace(features=features))


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


def test_the_whole_policy_loads_with_torch_compile_off(tmp_path, monkeypatch):
    config = act_config()
    ACTPolicy(config).save_pretrained(tmp_path / "policy")
    config.pretrained_path = tmp_path / "policy"
    # ACT has no such field; a checkpoint of a policy that has one, such as Diffusion, can save it on.
    config.compile_model = True
    monkeypatch.setattr(recipe_module, "load_processors", lambda config: (None, None))
    load, compiled = factory.load_pretrained_policy, []

    def load_and_record(config, **kwargs):
        compiled.append(config.compile_model)
        return load(config, **kwargs)

    monkeypatch.setattr(factory, "load_pretrained_policy", load_and_record)
    recipe_module.load_policy(SimpleNamespace(policy=config, backend=SimpleNamespace(device="cpu")))
    assert compiled == [False]


def test_the_whole_policy_loads_with_its_peft_adapter_folded_in(tmp_path, monkeypatch):
    peft = pytest.importorskip("peft")
    config = act_config()
    ACTPolicy(config).save_pretrained(tmp_path / "base")
    lora = peft.LoraConfig(target_modules=["out_proj"])
    adapter = peft.get_peft_model(ACTPolicy.from_pretrained(tmp_path / "base"), lora)
    # A new LoRA adapter changes nothing; a trained one changes the weights it adapts.
    with torch.no_grad():
        for name, parameter in adapter.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.1)
    adapter.peft_config["default"].base_model_name_or_path = str(tmp_path / "base")
    adapter.save_pretrained(tmp_path / "adapter")
    merged = {name: value.clone() for name, value in adapter.merge_and_unload().state_dict().items()}
    config.use_peft, config.pretrained_path = True, tmp_path / "adapter"
    monkeypatch.setattr(recipe_module, "load_processors", lambda config: (None, None))
    cfg = SimpleNamespace(policy=config, backend=SimpleNamespace(device="cpu"))
    policy, _, _ = recipe_module.load_policy(cfg)
    assert type(policy) is ACTPolicy
    state = policy.state_dict()
    assert state.keys() == merged.keys()
    assert all(torch.equal(state[name], merged[name]) for name in merged)


def test_an_image_recorded_channels_last_keeps_its_shape():
    features = {
        IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 24, 32)),
        f"{IMAGE}_hwc": PolicyFeature(type=FeatureType.VISUAL, shape=(24, 32, 3)),
    }
    assert frame_shapes(features) == {IMAGE: (24, 32, 3), f"{IMAGE}_hwc": (24, 32, 3)}


def test_a_policy_export_module_must_name_its_recipe(monkeypatch):
    # A policy without one gets the one-program recipe.
    assert factory.get_export_recipe_class(act_config()) is ActionChunkRecipe
    monkeypatch.setattr(
        factory, "_import_sibling_policy_module", lambda *args: SimpleNamespace(__name__="export_act")
    )
    with pytest.raises(ValueError, match="export_act must define the export recipe ACTExportRecipe"):
        factory.get_export_recipe_class(act_config())
