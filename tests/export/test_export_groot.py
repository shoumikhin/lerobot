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

"""The GR00T N1.7 export recipe, end to end on a tiny random GR00T with its own tiny Qwen3-VL processor."""

import json

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("diffusers")
cv2 = pytest.importorskip("cv2")

import tests.export.torch_backend  # noqa: E402, F401  # registers the test backend
from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.inference.bundle import Bundle, BundleInferenceEngine  # noqa: E402
from lerobot.policies import make_pre_post_processors, prepare_observation_for_inference  # noqa: E402
from lerobot.policies.groot.configuration_groot import GrootConfig  # noqa: E402
from lerobot.policies.groot.export_groot import image_stages  # noqa: E402
from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config  # noqa: E402
from lerobot.policies.groot.modeling_groot import GrootPolicy  # noqa: E402
from lerobot.policies.groot.processor_groot import (  # noqa: E402
    GrootN17VLMEncodeStep,
    _transform_n1_7_image_for_vlm_albumentations,
    _transform_n1_7_image_for_vlm_torch,
)
from lerobot.scripts.lerobot_export import export  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

SPECIAL = [
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
]
WORDS = [
    "<|endoftext|>",
    "[UNK]",
    "user",
    "pick",
    "up",
    "the",
    "red",
    "cube",
    "then",
    "stop",
    "perform",
    "task",
]
CHAT_TEMPLATE = (
    "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n"
    "{% for content in message['content'] %}{% if content['type'] == 'image' %}"
    "<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'text' %}{{ content['text'] }}"
    "{% endif %}{% endfor %}<|im_end|>\n{% endfor -%}"
)
TASK = "Pick up the red cube"
# The geometry each image mode resizes with, and the cameras its checkpoint takes.
MODES = {
    "albumentations": (
        {"image_target_size": [16, 16], "shortest_image_edge": 16, "crop_fraction": 0.75},
        {"observation.images.front": (3, 32, 32), "observation.images.wrist": (3, 32, 32)},
    ),
    "torchvision": (
        {"image_target_size": [16, 16], "image_crop_size": [12, 12]},
        {"observation.images.front": (3, 24, 40), "observation.images.wrist": (3, 24, 40)},
    ),
}


def write_backbone(path):
    """A tiny Qwen3-VL: its config, a word-level tokenizer with Qwen's special tokens, and its processors."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3VLConfig
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    vocab = {token: i for i, token in enumerate(SPECIAL + WORDS)}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(SPECIAL)
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="<|endoftext|>")
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(path)
    pixels = {"patch_size": 4, "temporal_patch_size": 2, "merge_size": 2}
    Qwen2VLImageProcessor(**pixels, min_pixels=64, max_pixels=4096).save_pretrained(path)
    Qwen3VLVideoProcessor(**pixels).save_pretrained(path)
    Qwen3VLConfig(
        image_token_id=vocab["<|image_pad|>"],
        video_token_id=vocab["<|video_pad|>"],
        vision_start_token_id=vocab["<|vision_start|>"],
        vision_end_token_id=vocab["<|vision_end|>"],
        tie_word_embeddings=True,
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 7,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "vocab_size": 64,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True},
            "tie_word_embeddings": True,
        },
        vision_config={
            "depth": 3,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 4,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "deepstack_visual_indexes": [0, 1, 2],
            "num_position_embeddings": 16,
        },
    ).save_pretrained(path)


def save_checkpoint(path, mode: str):
    """A fine-tuned GR00T: 7 Qwen3 layers (two language parts), 9 diffusion blocks (two step parts)."""
    geometry, cameras = MODES[mode]
    backbone = path / "backbone"
    write_backbone(backbone)
    base = path / "base"
    GR00TN17(
        GR00TN17Config(
            model_name=str(backbone),
            select_layer=7,
            backbone_embedding_dim=32,
            hidden_size=32,
            input_embedding_dim=32,
            max_state_dim=8,
            max_action_dim=6,
            action_horizon=4,
            max_num_embodiments=12,
            vl_self_attention_cfg={"num_layers": 1, "num_attention_heads": 2, "attention_head_dim": 16},
            diffusion_model_cfg={
                "positional_embeddings": None,
                "num_layers": 9,
                "num_attention_heads": 2,
                "attention_head_dim": 16,
                "norm_type": "ada_norm",
                "dropout": 0.0,
                "final_dropout": False,
                "output_dim": 32,
                "interleave_self_attention": True,
            },
            load_bf16=True,
        ),
        load_backbone_weights=False,
    ).to(torch.bfloat16).save_pretrained(base)
    config = GrootConfig(
        base_model_path=str(base),
        input_features={
            **{name: PolicyFeature(type=FeatureType.VISUAL, shape=shape) for name, shape in cameras.items()},
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(5,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(5,))},
        device="cpu",
        chunk_size=4,
        n_action_steps=4,
        max_state_dim=8,
        max_action_dim=6,
        use_bf16=False,
    )
    torch.manual_seed(0)
    policy = GrootPolicy(config)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(torch.randn_like(parameter), alpha=0.02)
    policy.save_pretrained(path / "finetune")
    stats = {
        OBS_STATE: {"min": torch.full((5,), -1.0), "max": torch.full((5,), 2.0)},
        ACTION: {"min": torch.full((5,), -10.0), "max": torch.full((5,), 30.0)},
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    encode = next(s for s in preprocessor.steps if isinstance(s, GrootN17VLMEncodeStep))
    encode.model_name = str(backbone)
    encode.use_albumentations = mode == "albumentations"
    encode.image_crop_size = encode.shortest_image_edge = encode.crop_fraction = None
    for key, value in geometry.items():
        setattr(encode, key, value)
    for processor in (preprocessor, postprocessor):
        processor.save_pretrained(path / "finetune")
    return path / "finetune"


@pytest.fixture(scope="module", params=list(MODES))
def exported(request, tmp_path_factory):
    path = tmp_path_factory.mktemp(f"groot_{request.param}")
    checkpoint = save_checkpoint(path, request.param)
    bundle = path / "bundle"
    export(
        [
            "--backend.type=torch_test",
            f"--policy.path={checkpoint}",
            f"--output_dir={bundle}",
            f"--task={TASK}",
        ],
        ["tests.export.torch_backend"],
    )
    return checkpoint, Bundle.read(bundle)


def test_bundle_is_the_chain_of_parts(exported):
    _, bundle = exported
    manifest = bundle.manifest
    steps = [f"step_{group}_{iteration}" for iteration in range(4) for group in range(2)]
    names = ["vision", "language_model_0", "language_model_1", "keys_values", *steps, "actions"]
    assert [p.name for p in manifest.programs] == names
    assert [p.file for p in manifest.programs[4:6]] == ["step_0.pt2", "step_1.pt2"]
    assert len({p.file for p in manifest.programs}) == 7
    # The last step's run returns the next actions; the actions part reads the last of them.
    assert manifest.programs[-2].outputs == ["x_4"] and manifest.programs[-1].inputs == ["x_4"]
    assert manifest.noise_shape == [4, 6]
    tokens = manifest.task_tokens
    known = json.loads((bundle.path / tokens.known).read_text())
    assert TASK in known and tokens.max_tokens == len(known[TASK]) + 8
    assert manifest.task == TASK


def test_bundle_plays_the_actions_of_the_pytorch_policy(exported):
    """Another frame and a task the export never saw, against the policy's own chunk under the same noise."""
    checkpoint, bundle = exported
    policy = GrootPolicy.from_pretrained(checkpoint)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, checkpoint)
    engine = BundleInferenceEngine(bundle)
    rng = np.random.default_rng(1)
    frame = {
        name: rng.integers(0, 256, (shape[0], shape[1], shape[2]), dtype=np.uint8)
        for name, shape in engine._frame_spec.items()
        if name != OBS_STATE
    }
    frame[OBS_STATE] = rng.random(5, dtype=np.float32)
    noise = rng.standard_normal((4, 6)).astype(np.float32)
    for task in (TASK, "stop then pick up the cube"):
        observation = prepare_observation_for_inference(dict(frame), "cpu", task)
        with torch.no_grad():
            chunk = policy.predict_action_chunk(
                preprocessor(observation), noise=torch.from_numpy(noise)[None]
            )
            expected = postprocessor(chunk)[0].numpy()
        actual = engine._run_chunk(frame, task, noise)
        np.testing.assert_allclose(actual / engine._scale, expected / engine._scale, atol=1e-3)


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
