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

import subprocess
import sys
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
    ActionChunkRecipe,
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


def test_a_part_or_a_run_cannot_take_the_name_of_the_noise_draw():
    from lerobot.bundle.manifest import ProgramRun
    from lerobot.export.recipe import Part, RunPlan

    with pytest.raises(ValueError, match="A part cannot be named noise"):
        Part("noise", torch.nn.Identity(), (torch.zeros(2),), ["a"], ["y"])
    named = ProgramRun("noise", "model", ["observation.state"], ["action"])
    running = ProgramRun("draw", "noise", [], ["noise"])
    for plan in ({"runs": [named]}, {"runs": [running]}):
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
