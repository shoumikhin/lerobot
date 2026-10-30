#!/usr/bin/env python

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

"""The LaWAM export recipe, on LaWAM's preprocessor and a tiny random LaWAM on a tiny random Qwen3-VL."""

from copy import copy
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from transformers import Qwen2VLImageProcessor, Qwen3VLForConditionalGeneration  # noqa: E402

from lerobot.bundle.manifest import NOISE  # noqa: E402
from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.export.recipe import ExportRecipe, random_frame  # noqa: E402
from lerobot.policies.lawam.configuration_lawam import LaWAMConfig  # noqa: E402
from lerobot.policies.lawam.export_lawam import LaWAMActions, LaWAMExportRecipe  # noqa: E402
from lerobot.policies.lawam.modeling_lawam import LaWAMPolicy  # noqa: E402
from lerobot.policies.lawam.processor_lawam import (  # noqa: E402
    LaWAMQwenInputsProcessorStep,
    make_lawam_pre_post_processors,
)
from lerobot.policies.lawam.vlas import lawam  # noqa: E402
from lerobot.policies.lawam.vlas.qwen3vl import remove_lm_head  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402
from tests.policies.common.qwen3_vl_helpers import IMAGE_TOKEN_ID, tiny_qwen3_vl_config  # noqa: E402

PLACEHOLDER_TOKEN = 20
IMAGE_SIDE = 32


class Processor:
    """Like Qwen3-VL's processor: the image processor's pixels, and one image token per merged patch."""

    def __init__(self, placeholder: str):
        pixels = IMAGE_SIDE * IMAGE_SIDE
        self.image_processor = Qwen2VLImageProcessor(
            patch_size=4,
            merge_size=2,
            temporal_patch_size=2,
            size={"shortest_edge": pixels, "longest_edge": pixels},
        )
        self.placeholder = placeholder

    def apply_chat_template(self, messages, processor_kwargs, **kwargs):
        content = [block for row in messages for message in row for block in message["content"]]
        images = [block["image"] for block in content if block["type"] == "image"]
        vision = self.image_processor(
            images=images, do_rescale=False, device=processor_kwargs["device"], return_tensors="pt"
        )
        image_tokens = int(vision["image_grid_thw"].prod(-1).sum()) // 4
        texts = [block["text"] for block in content if block["type"] == "text"]
        placeholders = sum(text.count(self.placeholder) for text in texts)
        input_ids = torch.tensor(
            [[1, *[IMAGE_TOKEN_ID] * image_tokens, 3, *[PLACEHOLDER_TOKEN] * placeholders]]
        )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": vision["pixel_values"],
            "image_grid_thw": vision["image_grid_thw"],
        }


def make_recipe(monkeypatch: pytest.MonkeyPatch, use_state: bool = True) -> ExportRecipe:
    """The LaWAM recipe on a LaWAM whose Qwen3-VL has 7 text layers, so its language model takes two parts."""
    torch.manual_seed(0)
    qwen_config = tiny_qwen3_vl_config()
    qwen_config.text_config.num_hidden_layers = 7

    def tiny_vlm(cfg, vlm_model_id):
        vlm = Qwen3VLForConditionalGeneration(qwen_config).eval()
        remove_lm_head(vlm)
        return vlm, None, PLACEHOLDER_TOKEN

    monkeypatch.setattr(lawam, "_load_vlm_and_processor", tiny_vlm)
    identity = NormalizationMode.IDENTITY
    config = LaWAMConfig(
        device="cpu",
        chunk_size=4,
        action_horizon=4,
        n_action_steps=2,
        base_vlm="dummy-qwen",
        normalization_mapping={"VISUAL": identity, "STATE": identity, "ACTION": identity},
        input_features={
            "observation.images.front": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, IMAGE_SIDE, IMAGE_SIDE)
            ),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        lam_image_hw=(IMAGE_SIDE, IMAGE_SIDE),
        lam_dim=32,
        lam_num_heads=4,
        lam_ffn_expansion_factor=2,
        lam_enc_layers=1,
        lam_code_dim=8,
        lam_dec_layers=1,
        dinov3_hidden_size=32,
        dinov3_intermediate_size=64,
        dinov3_num_hidden_layers=2,
        dinov3_num_attention_heads=4,
        num_action_queries=1,
        flow_action_num_queries=2,
        flow_hidden_dim=32,
        flow_num_layers=2,
        flow_attention_heads=4,
        flow_vlm_dim=qwen_config.text_config.hidden_size,
        flow_use_state=use_state,
        flow_num_inference_steps=2,
    )
    preprocessor, postprocessor = make_lawam_pre_post_processors(config, dataset_meta=SimpleNamespace(fps=10))
    (step,) = (step for step in preprocessor.steps if isinstance(step, LaWAMQwenInputsProcessorStep))
    step._processor, step._placeholder_token_id = Processor(step.placeholder_token), PLACEHOLDER_TOKEN
    cfg = SimpleNamespace(
        backend=SimpleNamespace(device="cpu"), seed=0, dataset=None, task="pick up the cube", tolerance=None
    )
    return LaWAMExportRecipe(LaWAMPolicy(config).eval(), preprocessor, postprocessor, cfg)


def chained(recipe: ExportRecipe, modules: dict[str, torch.nn.Module], inputs: tuple[torch.Tensor, ...]):
    """The actions of `modules`, one per part, run in the order of the recipe's plan on the frame and noise."""
    values = dict(zip([*recipe._features, NOISE], inputs, strict=True))
    with torch.no_grad():
        for run in recipe.plan().runs:
            outputs = modules[run.program](*(values[name] for name in run.inputs))
            values.update(
                zip(run.outputs, outputs if isinstance(outputs, tuple) else (outputs,), strict=True)
            )
    return values[ACTION]


@pytest.mark.parametrize("use_state", [True, False])
def test_the_programs_compute_the_chain_on_another_frame(
    monkeypatch: pytest.MonkeyPatch, use_state: bool
) -> None:
    recipe = make_recipe(monkeypatch, use_state)

    parts = {part.name: part for part in recipe.parts()}
    programs = {
        name: torch.export.export(part.module, part.example_inputs, strict=False).module()
        for name, part in parts.items()
    }

    assert list(parts) == ["embed", "language_model_0", "language_model_1", "image_features", "actions"]
    # A program input that nothing reads would stay on the CPU and cost a copy on every chunk.
    assert ("state" in parts["actions"].inputs) == use_state
    frame = random_frame(recipe.config.input_features, torch.Generator().manual_seed(1))
    noise = torch.randn(recipe.test_case().noise.shape, generator=torch.Generator().manual_seed(2))
    inputs = tuple(torch.from_numpy(x) for x in (*frame.values(), noise.numpy()))
    eager = {name: part.module for name, part in parts.items()}
    torch.testing.assert_close(
        chained(recipe, programs, inputs), chained(recipe, eager, inputs), rtol=0, atol=0
    )


def test_each_program_holds_only_the_weights_it_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Compiled as one program, the 5 GiB of a real checkpoint need more memory to build than an 8 GB board has.
    recipe = make_recipe(monkeypatch)
    backend = recipe.policy.model.policy_backend
    weights = {p.data_ptr() for p in recipe.policy.parameters()}
    vision = {p.data_ptr() for p in backend.vlm.model.visual.parameters()}
    flow = {p.data_ptr() for p in backend.flow.parameters()}
    for part in recipe.parts():
        program = torch.export.export(part.module, part.example_inputs, strict=False)
        tensors = [*program.state_dict.values(), *program.constants.values()]
        held = {t.data_ptr() for t in tensors if isinstance(t, torch.Tensor)} & weights
        assert held == {p.data_ptr() for p in part.module.parameters()}, part.name
        assert not (held & vision and held & flow), part.name


def test_a_chain_that_computes_something_else_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = make_recipe(monkeypatch)
    forward = LaWAMActions.forward

    def without_the_prompt(self, now, latent, hidden, *inputs):
        return forward(self, now, latent, torch.zeros_like(hidden), *inputs)

    monkeypatch.setattr(LaWAMActions, "forward", without_the_prompt)
    with pytest.raises(ValueError, match="lerobot-rollout"):
        list(recipe.parts())


def test_the_policy_loads_without_the_parts_only_training_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # The latent action model's encoder and state decoder (1.5 GiB in the trained checkpoint) do not fit an 8 GB board.
    policy = make_recipe(monkeypatch).policy
    policy.save_pretrained(tmp_path)
    config = copy(policy.config)
    config.pretrained_path = str(tmp_path)

    loaded = LaWAMExportRecipe.inference_policy(config)

    lam = loaded.model.policy_backend.lam
    assert lam.encoder is None and lam.state_decoder is None
    training = ("model.policy_backend.lam.encoder.", "model.policy_backend.lam.state_decoder.")
    expected = {name: x for name, x in policy.state_dict().items() if not name.startswith(training)}
    actual = loaded.state_dict()
    assert actual.keys() == expected.keys()
    for name, x in expected.items():
        assert torch.equal(actual[name], x), name
