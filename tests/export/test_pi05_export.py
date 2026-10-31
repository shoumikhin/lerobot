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

"""pi0.5's task tokens, end to end on a tiny pi0.5, with the test backend."""

import filecmp
import json
import weakref
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from lerobot.bundle.engine import BundleEngine  # noqa: E402
from lerobot.bundle.manifest import TASK_EMBEDDINGS, TASK_LENGTH, Bundle  # noqa: E402
from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies.pi05 import PI05Policy, export_pi05  # noqa: E402
from lerobot.processor import TokenizerProcessorStep  # noqa: E402
from lerobot.scripts.lerobot_export import export  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from tests.export.plugins import use_plugins  # noqa: E402
from tests.export.tiny_pi05 import IMAGE, save_checkpoint, use_tiny_pi05  # noqa: E402

TASK = "pick up the cube"
OTHER_TASK = "open_the drawer\n"
DATASET_FEATURES = {
    IMAGE: {"dtype": "image", "shape": (28, 28, 3), "names": ["height", "width", "channels"]},
    OBS_STATE: {"dtype": "float32", "shape": (3,), "names": None},
    ACTION: {"dtype": "float32", "shape": (3,), "names": None},
}


@pytest.fixture(scope="module", autouse=True)
def tiny_pi05_plugins():
    """Add the test backend and make pi0.5 tiny in the export's steps, which run in processes of their own."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        use_plugins(monkeypatch, "torch_backend", "tiny_pi05")
        yield


def run_export(checkpoint, output_dir, *args: str) -> Bundle:
    export(["--backend.type=torch_test", f"--policy.path={checkpoint}", f"--output_dir={output_dir}", *args])
    return Bundle.read(output_dir)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pi05")
    with pytest.MonkeyPatch.context() as patch:
        save_checkpoint(folder / "checkpoint", patch)
    return folder / "checkpoint"


def test_task_budget_follows_max_task_tokens(checkpoint, tmp_path):
    bundle = run_export(checkpoint, tmp_path / "bundle", f"--task={TASK}", "--max_task_tokens=12")
    tokens = bundle.manifest.task_tokens
    assert tokens.max_tokens == 12
    # The checkpoint's own tokenizer file, byte for byte.
    assert filecmp.cmp(
        bundle.path / tokens.tokenizer, checkpoint.parent / "tokenizer" / "tokenizer.json", False
    )


@pytest.mark.dataset
def test_export_knows_the_dataset_tasks_and_exports_the_first(checkpoint, tmp_path):
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.datasets.io_utils import write_info

    root = tmp_path / "dataset"
    metadata = LeRobotDatasetMetadata.create(
        "user/tiny", fps=10, features=DATASET_FEATURES, root=root, use_videos=False
    )
    metadata.save_episode_tasks([OTHER_TASK, TASK])
    # Recorded episodes count their tasks this way; this dataset holds tasks only.
    metadata.info.total_tasks = 2
    write_info(metadata.info, root)
    bundle = run_export(
        checkpoint, tmp_path / "bundle", "--dataset.repo_id=user/tiny", f"--dataset.root={root}"
    )
    assert bundle.manifest.task == OTHER_TASK
    known = json.loads((bundle.path / bundle.manifest.task_tokens.known_tasks).read_text())
    assert list(known) == [OTHER_TASK, TASK]
    BundleEngine(bundle)


@pytest.mark.parametrize(
    "args, error",
    [
        ([f"--task={TASK}", "--max_task_tokens=3"], "more than the 3 of --max_task_tokens"),
        ([f"--task={TASK}", "--max_task_tokens=100"], "at most"),
    ],
)
def test_export_refuses_a_task_the_prompt_cannot_hold(checkpoint, tmp_path, capfd, args, error):
    with pytest.raises(RuntimeError, match="trace step failed"):
        run_export(checkpoint, tmp_path / "bundle", *args)
    assert error in capfd.readouterr().err


@pytest.fixture
def recipe_config(checkpoint, monkeypatch):
    """What `lerobot-export` hands the recipe, for a recipe built in this process."""
    use_tiny_pi05(monkeypatch)
    # The tests that run the test case rely on its comparison with the policy, which a machine that does not
    # say how much memory it has would skip.
    monkeypatch.setattr(export_pi05, "whole_policy_fits", lambda *args: True)
    policy = PreTrainedConfig.from_pretrained(checkpoint)
    policy.pretrained_path = checkpoint
    return SimpleNamespace(
        policy=policy,
        backend=SimpleNamespace(device="cpu"),
        task=TASK,
        dataset=None,
        tolerance=None,
        max_task_tokens=None,
        seed=0,
    )


@pytest.mark.parametrize("task", [" \n ", "_ _"])
def test_recipe_refuses_a_task_with_no_text(recipe_config, task):
    recipe_config.task = task
    with pytest.raises(ValueError, match="is empty once cleaned"):
        export_pi05.PI05ExportRecipe(recipe_config)


@pytest.mark.dataset
def test_recipe_exports_the_given_task_beside_empty_or_no_dataset_tasks(recipe_config, tmp_path, monkeypatch):
    from lerobot.datasets import LeRobotDatasetMetadata

    # A dataset without episodes has no tasks; one with a blank task gets a prompt from the policy too.
    metadata = LeRobotDatasetMetadata.create(
        "user/tiny", fps=10, features=DATASET_FEATURES, root=tmp_path / "dataset", use_videos=False
    )
    monkeypatch.setattr(export_pi05, "dataset_metadata", lambda cfg: metadata)
    assert export_pi05.PI05ExportRecipe(recipe_config).task == TASK
    metadata.save_episode_tasks([" ", TASK])
    assert export_pi05.PI05ExportRecipe(recipe_config).task == TASK


@pytest.mark.dataset
def test_recipe_leaves_out_the_known_tasks_longer_than_the_task_budget(recipe_config, tmp_path, monkeypatch):
    from lerobot.datasets import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata.create(
        "user/tiny", fps=10, features=DATASET_FEATURES, root=tmp_path / "dataset", use_videos=False
    )
    metadata.save_episode_tasks([TASK, " ".join([TASK] * 4)])
    monkeypatch.setattr(export_pi05, "dataset_metadata", lambda cfg: metadata)
    recipe_config.max_task_tokens = 12
    # The bundle refuses a task longer than its budget.
    assert list(export_pi05.PI05ExportRecipe(recipe_config).plan().task_tokens.known_tasks) == [TASK]


def test_the_task_budget_defaults_to_the_longest_known_task_with_room_to_spare(recipe_config):
    task_tokens = export_pi05.PI05ExportRecipe(recipe_config).plan().task_tokens
    assert task_tokens.max_tokens == len(task_tokens.known_tasks[TASK]) + export_pi05.TASK_TOKEN_MARGIN


def test_recipe_refusal_names_the_prompts_limit_when_max_task_tokens_is_not_given(recipe_config):
    recipe_config.task = " ".join(["pick"] * 60)
    with pytest.raises(ValueError, match="more than the [0-9]+ the policy's prompt leaves"):
        export_pi05.PI05ExportRecipe(recipe_config)


def test_a_tokenizer_that_pads_on_the_left_gives_the_same_actions_as_the_policy(recipe_config, monkeypatch):
    # The padding takes no part in attention and shifts no position, so its side does not change the prompt.
    load_processors = export_pi05.load_processors

    def left_padded(config):
        preprocessor, postprocessor = load_processors(config)
        next(s for s in preprocessor.steps if isinstance(s, TokenizerProcessorStep)).padding_side = "left"
        return preprocessor, postprocessor

    monkeypatch.setattr(export_pi05, "load_processors", left_padded)
    recipe = export_pi05.PI05ExportRecipe(recipe_config)
    list(recipe.parts())
    recipe.test_case()


def test_the_step_takes_a_rounded_timestep_as_its_own(recipe_config):
    # A runtime may hand the step its timestep in a lower precision, such as bfloat16.
    parts = export_pi05.PI05ExportRecipe(recipe_config).parts()
    step = next(part for part in parts if part.name == "step")
    *inputs, _ = step.example_inputs
    # The second of the ten steps.
    timestep = torch.tensor([0.9])
    rounded = timestep.to(torch.bfloat16).float()
    assert rounded != timestep
    with torch.no_grad():
        x_next = step.module(*inputs, rounded)
        torch.testing.assert_close(x_next, step.module(*inputs, timestep), rtol=0, atol=0)


def test_recipe_refuses_a_checkpoint_whose_chunk_spec_fails(recipe_config):
    # Observation memory needs past frames at the control rate, which one chunk from the current frame lacks.
    recipe_config.policy.use_visual_memory = True
    with pytest.raises(ValueError, match="This pi05 checkpoint cannot be exported: Chunk inference cannot"):
        export_pi05.PI05ExportRecipe(recipe_config)


def test_recipe_refuses_a_tokenizer_step_without_a_tokenizer_name(recipe_config, monkeypatch):
    load_processors = export_pi05.load_processors

    def unnamed(config):
        preprocessor, postprocessor = load_processors(config)
        next(s for s in preprocessor.steps if isinstance(s, TokenizerProcessorStep)).tokenizer_name = None
        return preprocessor, postprocessor

    monkeypatch.setattr(export_pi05, "load_processors", unnamed)
    with pytest.raises(ValueError, match="names no tokenizer"):
        export_pi05.PI05ExportRecipe(recipe_config)


def test_recipe_refuses_a_processor_that_writes_another_prompt(recipe_config, monkeypatch):
    ops = [*export_pi05.TASK_PROMPT_OPS[:-1], ["template", "Goal: {task}, State:"]]
    monkeypatch.setattr(export_pi05, "TASK_PROMPT_OPS", ops)
    with pytest.raises(ValueError, match="processor makes the prompt"):
        export_pi05.PI05ExportRecipe(recipe_config)


def test_recipe_needs_its_parts_before_its_test_case(recipe_config):
    with pytest.raises(RuntimeError, match="which has not run yet"):
        export_pi05.PI05ExportRecipe(recipe_config).test_case()


def test_the_embed_parts_task_rows_hold_no_link_to_the_embedding_table(recipe_config):
    # A row that requires grad keeps the whole table it was copied from alive through the later parts.
    embed = next(export_pi05.PI05ExportRecipe(recipe_config).parts())
    assert not any(x.requires_grad for x in embed.example_inputs)


def test_recipe_refuses_a_prompt_with_other_tokens_than_the_processor(recipe_config, monkeypatch):
    token_ids = export_pi05.StatePrompt.token_ids
    monkeypatch.setattr(export_pi05.StatePrompt, "token_ids", lambda *args: [*token_ids(*args), 0])
    with pytest.raises(ValueError, match="holds other tokens than pi0.5's processor"):
        list(export_pi05.PI05ExportRecipe(recipe_config).parts())


def test_the_prompt_writes_the_processors_bin_for_a_nan_state(recipe_config, monkeypatch):
    # The prompt check compares the program's prompt with the processor's for this state value too.
    monkeypatch.setattr(export_pi05, "EXTREME_STATE", float("nan"))
    next(export_pi05.PI05ExportRecipe(recipe_config).parts())


@pytest.mark.parametrize("shift", [1.0, float("nan")])
def test_recipe_refuses_parts_that_compute_other_actions_than_the_policy(recipe_config, monkeypatch, shift):
    recipe = export_pi05.PI05ExportRecipe(recipe_config)
    list(recipe.parts())
    predict = PI05Policy.predict_action_chunk
    monkeypatch.setattr(
        PI05Policy, "predict_action_chunk", lambda *args, **kwargs: predict(*args, **kwargs) + shift
    )
    with pytest.raises(ValueError, match="compute other actions than pi0.5's predict_action_chunk"):
        recipe.test_case()


def test_a_large_tolerance_does_not_loosen_the_check_against_the_policy(recipe_config, monkeypatch):
    # --tolerance allows for the compiler, and the chain and the policy both run in PyTorch.
    recipe_config.tolerance = 10.0
    recipe = export_pi05.PI05ExportRecipe(recipe_config)
    list(recipe.parts())
    predict = PI05Policy.predict_action_chunk
    monkeypatch.setattr(
        PI05Policy, "predict_action_chunk", lambda *args, **kwargs: predict(*args, **kwargs) + 1
    )
    with pytest.raises(ValueError, match="compute other actions than pi0.5's predict_action_chunk"):
        recipe.test_case()


def test_an_empty_camera_slot_is_left_out_with_the_same_actions_as_the_policy(recipe_config):
    # The policy fills the slot with an image it leaves out of attention, after the cameras the frame holds.
    recipe_config.policy.empty_cameras = 1
    recipe_config.policy.validate_features()
    recipe = export_pi05.PI05ExportRecipe(recipe_config)
    embed, *_ = recipe.parts()
    assert embed.inputs == [IMAGE, OBS_STATE, TASK_EMBEDDINGS, TASK_LENGTH]
    recipe.test_case()


def test_recipe_refuses_parts_that_compute_actions_that_are_not_finite(recipe_config, monkeypatch):
    recipe = export_pi05.PI05ExportRecipe(recipe_config)
    forward = export_pi05.PI05Actions.forward
    monkeypatch.setattr(export_pi05.PI05Actions, "forward", lambda *args: forward(*args) * float("nan"))
    list(recipe.parts())
    with pytest.raises(ValueError, match="compute actions that are not finite"):
        recipe.test_case()


def test_a_bfloat16_chain_computes_as_the_policy(recipe_config, monkeypatch):
    """The parts cast between the dtypes of their weights as the policy does, which float32 weights never test."""
    recipe_config.policy.dtype = torch.bfloat16
    # Measured on the CPU: 9.7e-4.
    monkeypatch.setattr(export_pi05, "BFLOAT16_TOLERANCE", 2e-3)
    recipe = export_pi05.PI05ExportRecipe(recipe_config)
    list(recipe.parts())
    recipe.test_case()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_a_bfloat16_step_on_a_gpu_computes_as_the_policy(recipe_config, monkeypatch):
    """On a GPU too, a bfloat16 pi0.5 step keeps its adaptive norms and action projections in float32, as the
    policy's `denoise_step` does, so the chained parts stay within rounding of `predict_action_chunk`."""
    recipe_config.policy.dtype = torch.bfloat16
    recipe_config.policy.device = recipe_config.backend.device = "cuda"
    # Measured on a GPU: 7.6e-4 with the policy's float32 layers kept, 3.2e-3 with the whole step in bfloat16.
    monkeypatch.setattr(export_pi05, "BFLOAT16_TOLERANCE", 1.5e-3)
    recipe = export_pi05.PI05ExportRecipe(recipe_config)
    list(recipe.parts())
    recipe.test_case()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_a_bfloat16_recipe_on_a_gpu_frees_the_vision_tower_once_the_embed_part_is_built(recipe_config):
    recipe_config.policy.dtype = torch.bfloat16
    recipe_config.policy.device = recipe_config.backend.device = "cuda"
    parts = export_pi05.PI05ExportRecipe(recipe_config).parts()
    embed = next(parts)
    tower = weakref.ref(embed.module.policy.model.paligemma_with_expert.paligemma.model.vision_tower)
    # The export deletes each part before it builds the next one.
    del embed
    next(parts)
    assert tower() is None


def test_the_check_against_the_whole_policy_builds_it_on_the_export_device(monkeypatch):
    built_on = []

    class BuiltError(Exception):
        pass

    def from_pretrained(cls, *args, **kwargs):
        built_on.append(torch.get_default_device())
        raise BuiltError

    monkeypatch.setattr(export_pi05, "whole_policy_fits", lambda *args: True)
    monkeypatch.setattr(PI05Policy, "from_pretrained", classmethod(from_pretrained))
    recipe = object.__new__(export_pi05.PI05ExportRecipe)
    recipe._model_file, recipe._device = None, torch.device("meta")
    recipe._policy_config = SimpleNamespace(pretrained_path="policy", pretrained_revision=None)
    with pytest.raises(BuiltError):
        recipe._check_against_policy(None)
    assert built_on == [torch.device("meta")]
