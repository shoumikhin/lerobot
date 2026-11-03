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
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import Tensor

import lerobot.export.recipe as recipe_module
import lerobot.policies.factory as factory
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.export.recipe import (
    BFLOAT16_STEPS_TOLERANCE,
    BFLOAT16_TOLERANCE,
    FLOAT32_TOLERANCE,
    ActionChunkRecipe,
    ExportRecipe,
    ProgramCheck,
    ReplayDraws,
    export_task,
    frame_shapes,
    measure_action_scale,
    robot_features,
)
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.processor import NormalizerProcessorStep, ProcessorStep, TransitionKey
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from tests.export.tiny_act import IMAGE, act_config, act_stats

STATS = act_stats(action_std=(30.0, 3.0, 0.3))


def export_config(task: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task=task, tolerance=None
    )


def make(
    policy_class=ACTPolicy,
    task: str = "",
    extra_step: ProcessorStep | None = None,
    recipe_class: type[ActionChunkRecipe] = ActionChunkRecipe,
) -> ActionChunkRecipe:
    torch.manual_seed(0)
    config = act_config()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=STATS)
    if extra_step is not None:
        preprocessor.steps.insert(0, extra_step)
    return recipe_class(policy_class(config).eval(), preprocessor, postprocessor, export_config(task))


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


def test_a_part_run_once_per_step_passes_its_value_on_and_reads_each_steps_constants():
    from lerobot.bundle.manifest import ProgramRun
    from lerobot.export.recipe import repeated_runs

    steps = [{"timestep": np.array([1.0], np.float32)}, {"timestep": np.array([0.5], np.float32)}]
    runs, constants = repeated_runs("step", ["cache", "x", "timestep"], ["x"], steps, start={"x": "noise"})
    assert runs == [
        ProgramRun("step_0", "step", ["cache", "noise", "timestep_0"], ["x"]),
        ProgramRun("step_1", "step", ["cache", "x", "timestep_1"], ["x"]),
    ]
    assert list(constants) == ["timestep_0", "timestep_1"]
    np.testing.assert_array_equal(constants["timestep_1"], [0.5])


def test_the_relative_to_absolute_step_must_be_last_and_know_its_relative_step():
    from lerobot.export.recipe import split_absolute_step
    from lerobot.processor import (
        AbsoluteActionsProcessorStep,
        RelativeActionsProcessorStep,
        RenameObservationsProcessorStep,
    )

    paired = AbsoluteActionsProcessorStep(
        enabled=True, relative_step=RelativeActionsProcessorStep(enabled=True)
    )
    with pytest.raises(ValueError, match="must be the postprocessor's last step"):
        split_absolute_step(SimpleNamespace(steps=[paired, RenameObservationsProcessorStep()]), 3)
    with pytest.raises(ValueError, match="linked to no relative step"):
        split_absolute_step(SimpleNamespace(steps=[AbsoluteActionsProcessorStep(enabled=True)]), 3)


class FirstTwoActions(ProcessorStep):
    """Keeps the first two actions, as a postprocessor that crops a padded model width once unnormalized."""

    def __call__(self, transition):
        return {**transition, TransitionKey.ACTION: transition[TransitionKey.ACTION][..., :2]}

    def transform_features(self, features):
        return features


def test_recipe_measures_the_action_scale_at_the_width_the_postprocessor_takes():
    torch.manual_seed(0)
    config = act_config()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=STATS)
    # Right after the unnormalizer, which needs all three actions.
    postprocessor.steps.insert(1, FirstTwoActions())
    recipe = ActionChunkRecipe(ACTPolicy(config).eval(), preprocessor, postprocessor, export_config())
    np.testing.assert_allclose(recipe.test_case().action_scale, [30.0, 3.0], rtol=1e-6)


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


def test_a_part_or_a_run_cannot_take_the_name_of_the_noise_draw():
    from lerobot.bundle.manifest import ProgramRun
    from lerobot.export.recipe import Part, RunPlan

    with pytest.raises(ValueError, match="A part cannot be named noise"):
        Part("noise", torch.nn.Identity(), (torch.zeros(2),), ["a"], ["y"])
    named = ProgramRun("noise", "model", ["observation.state"], ["action"])
    running = ProgramRun("draw", "noise", [], ["noise"])
    for plan in ({"runs": [named]}, {"first_runs": [[named]]}, {"runs": [running]}):
        with pytest.raises(ValueError, match="A run cannot be named noise or run a program so named"):
            RunPlan(**plan)


def test_recipe_asks_the_policy_for_its_noise_shape():
    recipe = make(NoisyACT)
    assert recipe.test_case().noise.shape == (4, 5)
    [part] = recipe.parts()
    assert part.inputs[-1] == "noise"


def test_recipe_refuses_a_policy_that_draws_two_noises():
    with pytest.raises(ValueError, match="draws 2 random tensors"):
        make(TwoNoisesACT)


class ShiftedNoiseACT(ACTPolicy):
    """ACT that draws its starting noise around 5 with a spread of 0.5, with `torch.normal` or `Tensor.normal_`."""

    in_place = False

    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            shape = (1, self.config.chunk_size, 1)
            noise = torch.empty(shape).normal_(5.0, 0.5) if self.in_place else torch.normal(5.0, 0.5, shape)
        return super().predict_action_chunk(batch) + noise


@pytest.mark.parametrize("in_place", [False, True])
def test_recipe_refuses_a_policy_whose_noise_is_not_a_standard_normal(monkeypatch, in_place):
    monkeypatch.setattr(ShiftedNoiseACT, "in_place", in_place)
    with pytest.raises(ValueError, match="computes another chunk when it is given its starting noise"):
        make(ShiftedNoiseACT)


class ScaledDrawACT(ACTPolicy):
    """ACT that draws a standard normal and moves it in place to around 5 with a spread of 0.5."""

    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = torch.randn(1, self.config.chunk_size, 1).mul_(0.5).add_(5)
        return super().predict_action_chunk(batch) + noise


def test_recipe_refuses_a_policy_that_changes_its_standard_normal_draw():
    with pytest.raises(ValueError, match="changes the noise it draws before using it"):
        make(ScaledDrawACT)


class TransformerNoiseACT(ACTPolicy):
    """ACT that passes its starting noise through a PyTorch transformer layer, which has a fused kernel."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.layer = torch.nn.TransformerEncoderLayer(64, 8, batch_first=True)

    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = torch.randn(1, self.config.chunk_size, 64)
        return super().predict_action_chunk(batch) + self.layer(noise)[..., :1]


def test_recipe_takes_a_policy_whose_noise_goes_through_a_fused_kernel():
    make(TransformerNoiseACT)


class OneNumberNoiseACT(ACTPolicy):
    """ACT that draws one number of noise, for each sample (`shape` (1,)) or in all (`shape` ())."""

    shape: tuple[int, ...] = ()

    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        if noise is None:
            noise = torch.randn(self.shape)
        return super().predict_action_chunk(batch) + noise


@pytest.mark.parametrize("shape", [(), (1,)])
def test_recipe_refuses_a_policy_whose_noise_is_one_number(monkeypatch, shape):
    monkeypatch.setattr(OneNumberNoiseACT, "shape", shape)
    with pytest.raises(ValueError, match="not one starting noise for one sample"):
        make(OneNumberNoiseACT)


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


class NonFiniteACT(NoisyACT):
    """ACT that draws a starting noise and adds `value`, NaN or infinity, to its actions."""

    value = float("nan")

    def predict_action_chunk(self, batch: dict[str, Tensor], *, noise: Tensor | None = None) -> Tensor:
        return super().predict_action_chunk(batch, noise=noise) + self.value


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_recipe_refuses_a_policy_whose_actions_are_not_finite(monkeypatch, value):
    monkeypatch.setattr(NonFiniteACT, "value", value)
    with pytest.raises(ValueError, match="computes actions that are not finite"):
        make(NonFiniteACT)


def test_replayed_draws_read_the_noise_in_order():
    noise = torch.arange(6.0).reshape(3, 1, 2)
    with ReplayDraws(noise) as draws:
        drawn = [torch.randn(1, 2), torch.randn_like(torch.ones(1, 2)), torch.ones(1, 2).normal_()]
    assert torch.equal(torch.stack(drawn), noise) and draws.shapes == [(1, 2)] * 3
    with pytest.raises(ValueError, match=r"Draw 0 reads noise shaped \[1, 3\]"), ReplayDraws(noise):
        torch.randn(1, 3)
    with pytest.raises(ValueError, match="Draw 3"), ReplayDraws(noise):
        for _ in range(4):
            torch.randn(1, 2)


def test_replayed_draws_of_different_shapes_read_one_tensor_each():
    noise = [torch.randn(1, 2), torch.randn(3)]
    with ReplayDraws(noise) as draws:
        drawn = [torch.randn(1, 2), torch.randn_like(torch.ones(3))]
    assert all(map(torch.equal, drawn, noise)) and draws.shapes == [(1, 2), (3,)]
    with (
        pytest.raises(ValueError, match=r"Draw 1 reads noise shaped \[1, 2\], but the noise holds \[3\]"),
        ReplayDraws(noise),
    ):
        torch.randn(1, 2)
        torch.randn(1, 2)


def test_replayed_integers_are_equally_likely():
    noise = torch.randn(1, 2, 40000, 4, generator=torch.Generator().manual_seed(0))
    with ReplayDraws(noise) as draws:
        drawn = torch.randint(3, 7, size=(40000,))
    assert draws.shapes == [(2, 40000, 4)] and drawn.dtype == torch.int64
    counts = torch.bincount(drawn - 3, minlength=4) / 40000
    torch.testing.assert_close(counts, torch.full((4,), 0.25), atol=0.01, rtol=0)


def test_integers_are_drawn_as_usual_and_recorded_without_a_noise():
    with ReplayDraws() as draws:
        drawn = [
            torch.randint(5, (2, 3)),
            torch.randint(0, 5, size=(2,)),
            torch.randint(low=1, high=4, size=(6,)),
        ]
    assert draws.shapes == [(2, 2, 3, 5), (2, 2, 5), (2, 6, 3)]
    assert [tuple(x.shape) for x in drawn] == [(2, 3), (2,), (6,)]


def test_replayed_categories_follow_the_probabilities():
    probs = torch.tensor([[0.1, 0.2, 0.7]]).expand(20000, 3)
    noise = torch.randn(1, 2, 20000, 3, generator=torch.Generator().manual_seed(0))
    with ReplayDraws(noise):
        sampled = torch.multinomial(probs, num_samples=1)
    counts = torch.bincount(sampled[:, 0], minlength=3) / 20000
    torch.testing.assert_close(counts, probs[0], atol=0.01, rtol=0)


def test_replayed_normal_draws_keep_their_mean_and_std():
    noise = torch.tensor([0.5, -1.0, 2.0]).expand(4, 3)
    filled = torch.zeros(3)
    with ReplayDraws(noise):
        drawn = [
            torch.normal(torch.full((3,), 5.0), torch.full((3,), 2.0)),
            torch.normal(5.0, 2.0, size=(3,)),
            torch.distributions.Normal(torch.tensor(5.0), torch.tensor(2.0)).sample((3,)),
            filled.normal_(5.0, std=2.0),
        ]
    for value in drawn:
        assert torch.equal(value, torch.tensor([6.0, 3.0, 9.0])), value
    assert drawn[-1] is filled


class Draws(torch.nn.Module):
    """`draw` of its input, with the random draws reading its noise input."""

    def __init__(self, draw):
        super().__init__()
        self.draw = draw

    def forward(self, x: Tensor, noise: Tensor) -> Tensor:
        with ReplayDraws(noise):
            return self.draw(x)


@pytest.mark.parametrize(
    ("draw", "shape", "noise_shape", "op"),
    [
        (lambda logits: torch.multinomial(logits.softmax(-1), 1), (4, 8), (1, 2, 4, 8), "multinomial"),
        (lambda x: x + torch.empty_like(x).normal_(5.0, 2.0), (2, 3), (1, 2, 3), "normal"),
        (lambda values: values[torch.randint(0, 5, size=(2,))], (5, 3), (1, 2, 2, 5), "randint"),
    ],
)
def test_a_program_draws_from_its_noise_input(draw, shape, noise_shape, op):
    x, noise = torch.randn(shape), torch.randn(noise_shape)
    program = torch.export.export(Draws(draw), (x, noise), strict=False)
    # ExecuTorch cannot lower these random operations left in a program.
    assert op not in str(program.graph)
    assert torch.equal(program.module()(x, noise), Draws(draw)(x, noise))


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


def test_the_whole_policy_is_built_on_the_export_device(monkeypatch):
    monkeypatch.setattr(recipe_module, "load_processors", lambda config: (None, None))
    built_on = []

    def build(config, **kwargs):
        built_on.append(torch.get_default_device())
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(factory, "load_pretrained_policy", build)
    recipe_module.load_policy(SimpleNamespace(policy=act_config(), backend=SimpleNamespace(device="meta")))
    assert built_on == [torch.device("meta")]


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


@pytest.mark.parametrize(
    ("meminfo", "pages", "spare"),
    [
        # 64 GB installed, of which Linux counts 6.1 GB available.
        ("MemTotal:       62500000 kB\nMemAvailable:    5957032 kB\n", 15_625_000, "6.1 GB available"),
        # More than twice the checkpoint, but not three times.
        ("MemTotal:       62500000 kB\nMemAvailable:   14648438 kB\n", 15_625_000, "15.0 GB available"),
        # No /proc/meminfo, as on macOS: 6.1 GB installed.
        (None, 1_489_258, "6.1 GB installed"),
    ],
)
def test_a_policy_too_big_to_compare_is_skipped_with_a_log_line(
    tmp_path, monkeypatch, caplog, meminfo, pages, spare
):
    """A 6 GB checkpoint on a machine with less than three times that much memory to spare."""
    # Only the checkpoint's size is read, so no file that big is made.
    model_file = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=6_000_000_000))
    meminfo_file = tmp_path / "meminfo"
    if meminfo is not None:
        meminfo_file.write_text(meminfo)
    monkeypatch.setattr(recipe_module, "_MEMINFO", meminfo_file)
    monkeypatch.setattr(recipe_module, "_CGROUP", tmp_path / "no_cgroup")
    monkeypatch.setattr(
        recipe_module.os, "sysconf", {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": pages}.__getitem__, raising=False
    )
    with caplog.at_level(logging.WARNING, logger=recipe_module.__name__):
        assert not recipe_module.whole_policy_fits(model_file, torch.device("cpu"), "pi0.5", 3)
    assert (
        f"Skipped the comparison with the whole pi0.5 policy: it needs about 18.0 GB, {spare}. The "
        "bundle's own check against its test case still runs." in caplog.text
    )


@pytest.mark.parametrize(("parent_limit", "available"), [("max", "1.5 GB"), ("1500000000", "0.8 GB")])
def test_a_policy_too_big_for_its_container_is_skipped(
    tmp_path, monkeypatch, caplog, parent_limit, available
):
    """A 2 GB container on a machine with 1.8 TB available: its or its parent's tighter limit decides."""
    (tmp_path / "meminfo").write_text("MemAvailable: 1800000000 kB\n")
    (tmp_path / "cgroup").write_text("0::/user.slice/job.scope\n")
    job = tmp_path / "cgroups" / "user.slice" / "job.scope"
    job.mkdir(parents=True)
    (job.parent / "memory.max").write_text(f"{parent_limit}\n")
    (job.parent / "memory.current").write_text("700000000\n")
    (job.parent / "memory.stat").write_text("anon 700000000\ninactive_file 0\n")
    (job / "memory.max").write_text("2000000000\n")
    (job / "memory.current").write_text("600000000\n")
    # Page cache the kernel can drop counts as room.
    (job / "memory.stat").write_text("anon 500000000\ninactive_file 100000000\n")
    monkeypatch.setattr(recipe_module, "_MEMINFO", tmp_path / "meminfo")
    monkeypatch.setattr(recipe_module, "_CGROUP", tmp_path / "cgroup")
    monkeypatch.setattr(recipe_module, "_CGROUPS", tmp_path / "cgroups")
    model_file = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=1_000_000_000))
    with caplog.at_level(logging.WARNING, logger=recipe_module.__name__):
        assert not recipe_module.whole_policy_fits(model_file, torch.device("cpu"), "pi0.5", 3)
    assert f"it needs about 3.0 GB, {available} available" in caplog.text


def test_a_policy_too_big_for_host_memory_is_skipped_on_a_gpu(tmp_path, monkeypatch, caplog):
    """A 3 GB container on a GPU with 60 GB free: the whole policy passes through host memory too."""
    (tmp_path / "meminfo").write_text("MemAvailable: 1800000000 kB\n")
    (tmp_path / "cgroup").write_text("0::/job.scope\n")
    job = tmp_path / "cgroups" / "job.scope"
    job.mkdir(parents=True)
    (job / "memory.max").write_text("3000000000\n")
    (job / "memory.current").write_text("500000000\n")
    (job / "memory.stat").write_text("anon 500000000\ninactive_file 0\n")
    monkeypatch.setattr(recipe_module, "_MEMINFO", tmp_path / "meminfo")
    monkeypatch.setattr(recipe_module, "_CGROUP", tmp_path / "cgroup")
    monkeypatch.setattr(recipe_module, "_CGROUPS", tmp_path / "cgroups")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (60_000_000_000, 80_000_000_000))
    model_file = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=1_000_000_000))
    with caplog.at_level(logging.WARNING, logger=recipe_module.__name__):
        assert not recipe_module.whole_policy_fits(model_file, torch.device("cuda"), "pi0.5", 3)
    assert "it needs about 3.0 GB, 2.5 GB available on the host" in caplog.text


def test_the_whole_policy_needs_the_memory_its_recipe_measured(tmp_path, monkeypatch):
    """15 GB available: enough for a 6 GB checkpoint whose policy peaks at twice its size, not three times."""
    (tmp_path / "meminfo").write_text("MemAvailable:   14648438 kB\n")
    monkeypatch.setattr(recipe_module, "_MEMINFO", tmp_path / "meminfo")
    monkeypatch.setattr(recipe_module, "_CGROUP", tmp_path / "no_cgroup")
    model_file = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=6_000_000_000))
    assert recipe_module.whole_policy_fits(model_file, torch.device("cpu"), "pi0.5", 2)
    assert not recipe_module.whole_policy_fits(model_file, torch.device("cpu"), "pi0.5", 3)


def test_memory_that_pytorch_caches_counts_as_free_on_a_gpu(tmp_path, monkeypatch):
    """A GPU with 2 GB free, and 8 GB more that PyTorch keeps cached for the parts already run."""
    monkeypatch.setattr(recipe_module, "_MEMINFO", tmp_path / "meminfo")
    free = [2_000_000_000]
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: free.append(10_000_000_000))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free[-1], 80_000_000_000))
    model_file = SimpleNamespace(stat=lambda: SimpleNamespace(st_size=1_000_000_000))
    assert recipe_module.whole_policy_fits(model_file, torch.device("cuda"), "pi0.5", 3)


def test_a_policy_is_not_compared_on_a_machine_that_does_not_say_how_much_memory_it_has(
    tmp_path, monkeypatch, caplog
):
    """As on Windows: no /proc/meminfo and no os.sysconf."""
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"0")
    monkeypatch.setattr(recipe_module, "_MEMINFO", tmp_path / "meminfo")
    monkeypatch.delattr(recipe_module.os, "sysconf", raising=False)
    with caplog.at_level(logging.WARNING, logger=recipe_module.__name__):
        assert not recipe_module.whole_policy_fits(model_file, torch.device("cpu"), "pi0.5", 3)
    assert "this machine does not say how much memory it has" in caplog.text


def test_a_policy_export_module_must_name_its_recipe(monkeypatch):
    # A policy without one gets the one-program recipe.
    assert factory.get_export_recipe_class(act_config()) is ActionChunkRecipe
    monkeypatch.setattr(
        factory, "_import_sibling_policy_module", lambda *args: SimpleNamespace(__name__="export_act")
    )
    with pytest.raises(ValueError, match="export_act must define the export recipe ACTExportRecipe"):
        factory.get_export_recipe_class(act_config())


# In a process of its own, so that neither the CUDA tensors nor the freed host memory of earlier tests count.
LOAD_ON_THE_GPU = """
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import torch

from lerobot.export.recipe import load_policy
from lerobot.policies.act.configuration_act import ACTConfig


def anonymous_mib():
    # The memory of this process that no file backs, so not the pages of a memory-mapped checkpoint. Not
    # RssAnon, which on a Jetson also counts the GPU's allocations.
    with open("/proc/self/smaps_rollup") as smaps:
        return next(int(line.split()[1]) // 1024 for line in smaps if line.startswith("Anonymous:"))


path, weights = Path(sys.argv[1]), int(sys.argv[2])
config = ACTConfig.from_pretrained(path)
config.pretrained_path = path
torch.zeros(1, device="cuda")
torch.cuda.reset_peak_memory_stats()
start = anonymous_mib()
peak = [start]
done = threading.Event()


def sample():
    while not done.wait(0.001):
        peak[0] = max(peak[0], anonymous_mib())


sampler = threading.Thread(target=sample)
sampler.start()
try:
    policy, _, _ = load_policy(SimpleNamespace(policy=config, backend=SimpleNamespace(device="cuda")))
finally:
    done.set()
    sampler.join()
assert all(parameter.is_cuda for parameter in policy.parameters())
assert peak[0] - start < weights / 2, f"{peak[0] - start} MiB more host memory, {weights} MiB of weights"
cuda = torch.cuda.max_memory_allocated() >> 20
assert cuda < 1.5 * weights, f"{cuda} MiB of GPU memory at most, {weights} MiB of weights"
"""


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or not Path("/proc/self/smaps_rollup").exists(),
    reason="needs a CUDA GPU and Linux's memory counters",
)
def test_an_export_loads_the_policy_straight_onto_its_device(tmp_path):
    # ACT's default size, about 200 MiB of weights.
    config = ACTConfig(
        input_features=act_config().input_features,
        output_features=act_config().output_features,
        pretrained_backbone_weights=None,
        device="cpu",
    )
    policy = ACTPolicy(config)
    weights = sum(p.numel() * p.element_size() for p in policy.parameters()) >> 20
    policy.save_pretrained(tmp_path)
    for processor in make_pre_post_processors(config, dataset_stats=STATS):
        processor.save_pretrained(tmp_path)
    subprocess.run(
        [sys.executable, "-c", LOAD_ON_THE_GPU, str(tmp_path), str(weights)], check=True, timeout=600
    )


class EditedStepsRecipe(ActionChunkRecipe):
    """A recipe whose program runs the saved preprocessor with its steps changed by `edit`."""

    def program_preprocessor(self):
        return replace(self.preprocessor, steps=self.edit(self.preprocessor.steps))


class InexactStepsRecipe(EditedStepsRecipe):
    """Held to the export's tolerance rather than to rollout's actions bit for bit."""

    exact = False


class ScaleState(ProcessorStep):
    """Scales the state, as a program that computes the same thing in another order rounds it."""

    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, transition):
        observation = transition[TransitionKey.OBSERVATION]
        state = observation[OBS_STATE] * self.scale
        return {**transition, TransitionKey.OBSERVATION: {**observation, OBS_STATE: state}}

    def transform_features(self, features):
        return features


class FrozenState(ProcessorStep):
    """Keeps the first state it sees, as a program that held the test case's state as a constant would."""

    def __init__(self):
        self.state = None

    def __call__(self, transition):
        observation = transition[TransitionKey.OBSERVATION]
        if self.state is None:
            self.state = observation[OBS_STATE]
        return {**transition, TransitionKey.OBSERVATION: {**observation, OBS_STATE: self.state}}

    def transform_features(self, features):
        return features


@pytest.mark.parametrize(
    ("recipe_class", "edit", "error"),
    [
        # ACT ignores the task codes the first step makes, so the program computes the same without it.
        (EditedStepsRecipe, lambda steps: steps[1:], None),
        (
            EditedStepsRecipe,
            lambda steps: [step for step in steps if not isinstance(step, NormalizerProcessorStep)],
            "computes something else",
        ),
        # Float32 rounding: the actions are not rollout's bit for bit, but within the tolerance.
        (InexactStepsRecipe, lambda steps: [ScaleState(1 + 1e-6), *steps], None),
        (InexactStepsRecipe, lambda steps: [ScaleState(2.0), *steps], "more than the tolerance"),
        # A NaN difference is more than any tolerance too.
        (InexactStepsRecipe, lambda steps: [ScaleState(float("nan")), *steps], "more than the tolerance"),
        # Right for the test case's frame, which it sees first, and wrong for any other.
        (EditedStepsRecipe, lambda steps: [FrozenState(), *steps], "computes something else"),
    ],
)
def test_a_recipe_gives_the_program_a_preprocessor_that_computes_the_same(recipe_class, edit, error):
    recipe = make(task="pick", extra_step=TaskCodes(), recipe_class=recipe_class)
    recipe.edit = edit
    if error:
        with pytest.raises(ValueError, match=error):
            list(recipe.parts())
    else:
        [part] = recipe.parts()
        steps = [type(step) for step in part.module.preprocessor.steps]
        assert steps == [type(step) for step in edit(recipe.preprocessor.steps)]


class BareRecipe(ExportRecipe):
    """A recipe without parts, to read the tolerance it picks."""

    def parts(self):
        yield from ()

    def test_case(self):
        raise NotImplementedError


class DenoisingRecipe(BareRecipe):
    bfloat16_tolerance = BFLOAT16_STEPS_TOLERANCE


@pytest.mark.parametrize(
    ("dtype", "default", "raised"),
    [
        (torch.float32, FLOAT32_TOLERANCE, FLOAT32_TOLERANCE),
        (torch.bfloat16, BFLOAT16_TOLERANCE, BFLOAT16_STEPS_TOLERANCE),
        (torch.float16, BFLOAT16_TOLERANCE, BFLOAT16_STEPS_TOLERANCE),
    ],
)
def test_a_recipe_can_raise_the_bfloat16_tolerance(dtype, default, raised):
    policy = ACTPolicy(act_config()).to(dtype)

    assert BareRecipe(policy, None, None, export_config()).tolerance == default
    assert DenoisingRecipe(policy, None, None, export_config()).tolerance == raised


class _Scores(torch.nn.Module):
    """A bfloat16 layer, and the index of its largest output, which no rounding may change."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64).to(torch.bfloat16)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        scores = self.linear(x)
        return scores, scores.float().argmax(-1)


class _Unnormalize(torch.nn.Module):
    """Float32 arithmetic on a bfloat16 input, which no rerun rounds differently but a compiler may fuse."""

    def forward(self, x: Tensor) -> Tensor:
        return x.float() * 3.0 + 1.0


def test_a_program_check_lets_arithmetic_that_no_rerun_moves_round_by_a_step():
    torch.manual_seed(0)
    check = ProgramCheck.measure(_Unnormalize(), (torch.randn(8, 64, dtype=torch.bfloat16),))
    step = torch.finfo(torch.float32).eps
    assert check.floor == [(step, step)]
    (actions,) = check.outputs
    assert check.failures("actions", (actions * (1 + step),)) == []
    (failure,) = check.failures("actions", (actions * (1 + 4 * step),))
    assert failure.startswith("actions output 0")


class _Masked(torch.nn.Module):
    """A bfloat16 layer whose outputs an additive mask sets to minus infinity in places, as in attention."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64).to(torch.bfloat16)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x).masked_fill(torch.arange(64) % 4 == 0, float("-inf"))


def test_a_program_check_holds_an_output_with_infinities_to_its_rounding():
    torch.manual_seed(0)
    check = ProgramCheck.measure(_Masked(), (torch.randn(8, 64, dtype=torch.bfloat16),))
    assert np.isfinite(check.floor).all()
    (scores,) = check.outputs
    assert check.failures("scores", (scores.clone(),)) == []
    # The masked values stay minus infinity, and the others are far off.
    (failure,) = check.failures("scores", (scores + 100,))
    assert failure.startswith("scores output 0")
    (failure,) = check.failures("scores", (scores.nan_to_num(neginf=0.0),))
    assert failure.startswith("scores output 0")


def test_a_program_check_refuses_a_part_whose_output_holds_nan():
    with pytest.raises(ValueError, match="Output 0 of the part holds NaN in PyTorch"):
        ProgramCheck.measure(torch.nn.Identity(), (torch.tensor([1.0, float("nan")], dtype=torch.bfloat16),))


def test_a_program_check_refuses_an_output_of_another_shape_or_count():
    check = ProgramCheck((), (torch.ones(2, 3),), [(1e-3, 1e-3)])
    (failure,) = check.failures("x", (torch.ones(3, 2),))
    assert "largest relative difference inf" in failure
    assert check.failures("x", (torch.ones(2, 3), torch.ones(1))) == ["x returns 2 outputs, not 1"]


def test_a_program_check_saves_inputs_and_outputs_that_share_memory(tmp_path):
    y = torch.randn(4, dtype=torch.bfloat16)
    check = ProgramCheck((y, y[1:]), (y, y[1:]), [(0.0, 0.0), (0.0, 0.0)])
    check.save(tmp_path / "check.safetensors")
    loaded = ProgramCheck.load(tmp_path / "check.safetensors", "cpu")
    assert all(torch.equal(a, b) for a, b in zip(loaded.inputs, check.inputs, strict=True))
    assert all(torch.equal(a, b) for a, b in zip(loaded.outputs, check.outputs, strict=True))


def test_a_program_check_gives_the_part_back_in_its_own_dtype():
    # Seeded: for a few weights and inputs in a hundred, the float32 rerun picks another largest output.
    torch.manual_seed(0)
    part = _Scores()
    ProgramCheck.measure(part, (torch.randn(8, 64, dtype=torch.bfloat16),))
    assert {parameter.dtype for parameter in part.parameters()} == {torch.bfloat16}


class _WritesItsInput(torch.nn.Module):
    """A bfloat16 layer that first writes into its input, as a part that updates a cache does."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64).to(torch.bfloat16)

    def forward(self, x: Tensor) -> Tensor:
        x.add_(1)
        return self.linear(x)


def test_a_program_check_keeps_the_inputs_a_part_writes_into():
    torch.manual_seed(0)
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    check = ProgramCheck.measure(_WritesItsInput(), (x.clone(),))
    assert torch.equal(check.inputs[0], x)
    assert check.floor[0][1] < torch.finfo(torch.bfloat16).eps


class _VariesBetweenRuns(torch.nn.Module):
    """A layer whose result moves by up to `spread` of its size per run.

    `dtype` is in one place, `held_in`: its weights, a buffer that scales their result, its input (which the
    test makes) or its output. The rest is float32.
    """

    def __init__(self, spread: float, dtype: torch.dtype, held_in: str):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64, dtype=dtype if held_in == "weights" else torch.float32)
        self.register_buffer("scale", torch.ones(64, dtype=dtype if held_in == "buffer" else torch.float32))
        self.output_dtype = dtype if held_in == "output" else torch.float32
        self.spread = spread

    def forward(self, x: Tensor) -> Tensor:
        y = self.linear(x.to(self.linear.weight.dtype))
        y = (y.to(self.scale.dtype) * self.scale).float()
        return (y + self.spread * y.abs().max() * torch.rand_like(y)).to(self.output_dtype)


# GPU atomic adds move a float32 output by 1e-6 and a bfloat16 one by about half a step, 4e-3; a part that
# draws random numbers moves it by more.
@pytest.mark.parametrize(
    ("spread", "dtype", "held_in", "refused"),
    [
        (1e-6, torch.float32, "weights", False),
        (1e-2, torch.float32, "weights", True),
        (4e-3, torch.bfloat16, "weights", False),
        (4e-3, torch.bfloat16, "buffer", False),
        (4e-3, torch.bfloat16, "input", False),
        (4e-3, torch.bfloat16, "output", False),
        (0.1, torch.bfloat16, "weights", True),
    ],
)
def test_a_program_check_takes_kernels_that_vary_between_runs_but_not_random_numbers(
    spread, dtype, held_in, refused
):
    torch.manual_seed(0)
    part = _VariesBetweenRuns(spread, dtype, held_in)
    x = torch.randn(8, 64, dtype=dtype if held_in == "input" else torch.float32)
    if refused:
        with pytest.raises(ValueError, match="changes by more than rounding does when it runs again"):
            ProgramCheck.measure(part, (x,))
    else:
        ProgramCheck.measure(part, (x,))


def test_a_program_check_refuses_a_part_that_keeps_dropout_on():
    torch.manual_seed(0)
    part = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Dropout(0.1)).to(torch.bfloat16)
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    with pytest.raises(
        ValueError, match="Output 0 of the part changes by more than rounding does when it runs"
    ):
        ProgramCheck.measure(part.train(), (x,))
    ProgramCheck.measure(part.eval(), (x,))


class _ReturnsItsBuffer(torch.nn.Module):
    """A bfloat16 layer that writes its result into a float32 buffer of its own and returns that buffer."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64).to(torch.bfloat16)
        self.register_buffer("result", torch.zeros(8, 64))

    def forward(self, x: Tensor) -> Tensor:
        return self.result.copy_(self.linear(x))


def test_a_program_check_keeps_what_a_part_returned_before_its_reruns():
    torch.manual_seed(0)
    part, x = _ReturnsItsBuffer(), torch.randn(8, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = part(x).clone()
    # The float32 rerun writes other values into the buffer.
    check = ProgramCheck.measure(part, (x,))
    assert torch.equal(check.outputs[0], expected)


@pytest.mark.parametrize(
    ("floor", "spread"),
    [((0.01, 1.0), False), ((1.0, 0.01), True)],
    ids=["largest-off", "all-slightly-off"],
)
def test_a_program_check_holds_each_difference_to_three_times_its_rounding(floor, spread):
    """The largest and the mean difference each fail a program alone: one far value, or all a little off.

    Each case gives the check it does not test a floor 100 times larger, so only the tested check can fail.
    """
    reference = torch.tensor([1.0, -0.5, 0.25, 0.5], dtype=torch.float64)
    check = ProgramCheck((), (reference,), [floor])
    for times, passes in ((2.9, True), (3.1, False)):
        if spread:
            outputs = reference + times * 0.01 * reference.abs().mean()
        else:
            outputs = reference.clone()
            outputs[0] += times * 0.01
        assert (check.failures("x", (outputs,)) == []) == passes, times


def test_a_program_check_holds_a_program_to_its_rounding(tmp_path):
    torch.manual_seed(0)
    check = ProgramCheck.measure(_Scores(), (torch.randn(8, 64, dtype=torch.bfloat16),))
    largest, mean = check.floor[0]
    # Measured: a floor of one rounding step for both would mean that no rerun moved the layer.
    assert torch.finfo(torch.bfloat16).eps <= largest < 0.05 and 0 < mean < torch.finfo(torch.bfloat16).eps
    assert check.floor[1] == (0.0, 0.0)
    scores, best = check.outputs
    assert check.failures("scores", (scores, best)) == []
    assert check.failures("scores", (scores * (1 + 2 * min(largest, mean)), best)) == []
    (failure,) = check.failures("scores", (scores * (1 + 4 * max(largest, mean)), best))
    assert failure.startswith("scores output 0")
    (failure,) = check.failures("scores", (scores, best.roll(1)))
    assert failure.startswith("scores output 1")
    check.save(tmp_path / "check.safetensors")
    loaded = ProgramCheck.load(tmp_path / "check.safetensors", "cpu")
    assert loaded.floor == check.floor
    assert all(
        torch.equal(a, b)
        for a, b in zip((*loaded.inputs, *loaded.outputs), (*check.inputs, *check.outputs), strict=True)
    )
