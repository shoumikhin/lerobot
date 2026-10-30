# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

"""LingBot-VLA 2.0 tests on a tiny randomly initialized model, fully offline.

The Qwen3-VL config is patched to a 2-layer model and the tokenizer / image processor are
built locally, so every test runs the real processors and policy without the Hub.
"""

import json
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config, SlotMapping
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from tests.utils import skip_if_package_missing

CAMERA = "observation.images.camera_top"
WRIST = "observation.images.camera_wrist"
CHUNK = 4


def _slots(key: str, arm: list, effector: list) -> dict[str, SlotMapping]:
    def mapping(spans):
        return SlotMapping(origin_keys=[{key: {"start": s, "end": e}} for s, e in spans])

    return {f"{key}.arm.position": mapping(arm), f"{key}.effector.position": mapping(effector)}


def _slot_kwargs(arm: list, effector: list) -> dict:
    return {"state_slots": _slots(OBS_STATE, arm, effector), "action_slots": _slots(ACTION, arm, effector)}


def _ds_meta(dim: int, cameras: tuple[str, ...] = (CAMERA,)) -> SimpleNamespace:
    """The dataset metadata make_policy reads: features and stats of a ``dim``-D robot."""
    names = [f"joint_{i}" for i in range(dim - 1)] + ["gripper"]
    vector = {"dtype": "float32", "shape": (dim,), "names": names}
    image = {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]}
    features = {OBS_STATE: vector, ACTION: vector, **dict.fromkeys(cameras, image)}
    g = torch.Generator().manual_seed(dim)
    stats = {}
    for key in (OBS_STATE, ACTION):
        lo, hi = -1 - torch.rand(dim, generator=g), 1 + torch.rand(dim, generator=g)
        stats[key] = {"min": lo, "max": hi, "q01": lo, "q99": hi, "mean": lo + hi, "std": hi}
    return SimpleNamespace(features=features, stats=stats)


def _raw_batch(
    dim: int, device: str = "cpu", batch_size: int = 2, cameras: tuple[str, ...] = (CAMERA,)
) -> dict:
    g = torch.Generator().manual_seed(0)
    batch = {
        OBS_STATE: torch.randn(batch_size, dim, generator=g),
        **{camera: torch.rand(batch_size, 3, 32, 32, generator=g) for camera in cameras},
        ACTION: torch.randn(batch_size, CHUNK, dim, generator=g),
    }
    return {**{k: v.to(device) for k, v in batch.items()}, "task": ["pick up the cube"] * batch_size}


@pytest.fixture
@skip_if_package_missing("transformers")
def tiny_backbone(tmp_path, monkeypatch):
    """Patch the Qwen3-VL config to a tiny model and save an offline tokenizer + image processor."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor
    from transformers.models.qwen3_vl import Qwen3VLConfig

    import lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 as modeling

    vlm = Qwen3VLConfig()
    t, v = vlm.text_config, vlm.vision_config
    t.num_hidden_layers, t.hidden_size, t.intermediate_size = 2, 64, 128
    t.num_attention_heads, t.num_key_value_heads, t.head_dim = 4, 2, 16
    t.rope_parameters = {**t.rope_parameters, "mrope_section": [2, 3, 3]}
    t.eos_token_id = 3
    v.hidden_size, v.intermediate_size, v.num_heads, v.depth, v.out_hidden_size = 64, 128, 2, 2, 64
    v.deepstack_visual_indexes = [0, 1]
    monkeypatch.setattr(modeling.AutoConfig, "from_pretrained", lambda *args, **kwargs: vlm)

    words = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>", "user", "pick", "up", "the", "cube"]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>")
    tokenizer.chat_template = (
        "{% for m in messages %}<|im_start|> {{ m.role }} {{ m.content }} <|im_end|>{% endfor %}"
    )
    assets = tmp_path / "qwen_assets"
    tokenizer.save_pretrained(assets)
    Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2).save_pretrained(assets)
    return assets


def _tiny_config(assets, device: str = "cpu", **kwargs) -> LingbotVLAV2Config:
    return LingbotVLAV2Config(
        tokenizer_path=str(assets),
        device=device,
        dtype=torch.float32,
        expert_hidden_size=64,
        expert_intermediate_size=128,
        action_num_attention_heads=4,
        action_num_key_value_heads=2,
        action_head_dim=16,
        token_moe_layers=[0, 1],
        token_num_experts=4,
        token_top_k=2,
        token_moe_intermediate_size=32,
        token_shared_intermediate_size=32,
        chunk_size=CHUNK,
        n_action_steps=CHUNK,
        num_steps=2,
        tokenizer_max_length=16,
        resize_imgs_with_padding=(64, 64),
        image_max_pixels=64 * 64,
        image_min_pixels=64 * 64,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        },
        **kwargs,
    )


def _select_action(policy, preprocessor, postprocessor, dim: int, device: str) -> torch.Tensor:
    batch = _raw_batch(dim, device)
    batch.pop(ACTION)
    policy.reset()
    return postprocessor(policy.select_action(preprocessor(batch)))


def test_train_and_select_action(tiny_backbone):
    """7-D robot (arm 0-6, gripper 6-7, canonical ``end.position`` unmapped): train step + inference."""
    cfg = _tiny_config(tiny_backbone, **_slot_kwargs(arm=[[0, 6]], effector=[[6, 7]]))
    meta = _ds_meta(7)
    policy = make_policy(cfg, ds_meta=meta)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=meta.stats)

    loss, _ = policy.forward(preprocessor(_raw_batch(7)))
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters())

    action = _select_action(policy, preprocessor, postprocessor, 7, "cpu")
    assert action.shape == (2, 7)
    assert torch.isfinite(action).all()


def _precompute_policies(tiny_backbone, cameras, vision_boundaries):
    slots = _slot_kwargs(arm=[[0, 6]], effector=[[6, 7]])
    meta = _ds_meta(7, cameras)
    settings = {"qwen3vl_use_vision_boundaries": vision_boundaries, **slots}
    cfg = _tiny_config(tiny_backbone, **settings)
    policy = make_policy(cfg, ds_meta=meta).eval()
    fixed = make_policy(
        _tiny_config(tiny_backbone, precompute_grid_thw=True, **settings), ds_meta=meta
    ).eval()
    fixed.load_state_dict(policy.state_dict())
    preprocessor, _ = make_pre_post_processors(cfg, dataset_stats=meta.stats)
    raw = _raw_batch(7, cameras=cameras)
    raw.pop(ACTION)
    return policy, fixed, preprocessor(raw), torch.randn(2, CHUNK, cfg.max_action_dim)


@pytest.mark.parametrize("cameras", [(CAMERA,), (CAMERA, WRIST)], ids=["one-camera", "two-cameras"])
def test_precompute_grid_thw_prediction_matches_default_and_exports(tiny_backbone, monkeypatch, cameras):
    """Eager prediction keeps transformers' prefix positions; with the grid cached, prediction exports."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    from tests.policies.common.fixed_grid_checks import data_dependent_nodes

    stock_rope_index, eager_rope_calls = Qwen3VLModel.get_rope_index, []

    def counted_rope_index(self, *args, **kwargs):
        assert not torch.compiler.is_exporting(), "exported prefix positions must not use get_rope_index"
        eager_rope_calls.append(1)
        return stock_rope_index(self, *args, **kwargs)

    monkeypatch.setattr(Qwen3VLModel, "get_rope_index", counted_rope_index)
    policy, fixed, batch, noise = _precompute_policies(tiny_backbone, cameras, vision_boundaries=True)
    core = policy.model.qwenvl_with_expert
    build_prefix_position_ids, prefixes = core.build_prefix_position_ids, []

    def recorded(input_ids, attention_mask, **kwargs):
        positions = build_prefix_position_ids(input_ids, attention_mask, **kwargs)
        prefixes.append((input_ids, attention_mask, positions))
        return positions

    monkeypatch.setattr(core, "build_prefix_position_ids", recorded)
    # The sampler updates its noise in place, so every call gets its own copy.
    with torch.no_grad():
        expected = policy.predict_action_chunk(batch, noise=noise.clone()).clone()
    calls_per_prediction = len(eager_rope_calls)
    assert calls_per_prediction > 0
    # The positions export uses, computed from the grid ints of every camera, match transformers'.
    input_ids, attention_mask, positions = prefixes[0]
    every_grid = batch["image_grid_thw"].reshape(-1, 3).tolist()
    assert torch.equal(core.build_fixed_grid_position_ids(input_ids, attention_mask, every_grid), positions)

    visual = fixed.model.qwenvl_with_expert.qwenvl.model.visual
    grid_builds = []
    build_grid = visual.preprcess_grid_thw

    def recorded_build(*args, **kwargs):
        grid_builds.append(args or kwargs)
        return build_grid(*args, **kwargs)

    monkeypatch.setattr(visual, "preprcess_grid_thw", recorded_build)
    keys = sorted(key for key, value in batch.items() if isinstance(value, torch.Tensor))

    class PredictActionChunk(torch.nn.Module):
        def forward(self, noise, *tensors):
            return fixed.predict_action_chunk({**batch, **dict(zip(keys, tensors, strict=True))}, noise=noise)

    def run(module, inputs):
        return module(noise.clone(), *(inputs[key] for key in keys))

    with torch.no_grad():
        for _ in range(2):  # The second call reuses the cached grid.
            assert torch.equal(run(PredictActionChunk(), batch), expected)
        # Eager calls with the option on keep transformers' positions.
        assert len(eager_rope_calls) == 3 * calls_per_prediction
        exported = torch.export.export(
            PredictActionChunk(), (noise.clone(), *(batch[key] for key in keys)), strict=False
        )
        torch.testing.assert_close(run(exported.module(), batch), expected)
        # A grid other than the cached one fails the exported program's check.
        changed = {**batch, "image_grid_thw": batch["image_grid_thw"].clone()}
        changed["image_grid_thw"][..., 1] += 2
        with pytest.raises(RuntimeError):
            run(exported.module(), changed)
        # So does a sample with no visible prefix token, on which eager rope positions fail.
        empty = {key: batch[key].clone() for key in ("img_masks", OBS_LANGUAGE_ATTENTION_MASK)}
        for mask in empty.values():
            mask[1] = False
        with pytest.raises(RuntimeError):
            run(exported.module(), {**batch, **empty})
    assert len(grid_builds) == 1
    assert data_dependent_nodes(exported) == []

    if len(cameras) > 1:
        # A hidden camera keeps its image tokens in the prefix, masked out: the export positions take every
        # camera's grid, transformers takes the visible ones, and both must agree.
        config = core.qwenvl.config
        start = (input_ids[0] == config.vision_start_token_id).nonzero()[0, 0]
        end = (input_ids[0] == config.vision_end_token_id).nonzero()[0, 0]
        hidden = attention_mask.clone()
        hidden[0, start : end + 1] = 0
        stock = build_prefix_position_ids(input_ids, hidden, image_grid_thw=torch.tensor(every_grid[1:]))
        assert torch.equal(core.build_fixed_grid_position_ids(input_ids, hidden, every_grid), stock)
        # With cameras of two shapes and one token count, the hidden camera must not hand its grid to another.
        camera = [config.vision_start_token_id, *[config.image_token_id] * 6, config.vision_end_token_id]
        input_ids = torch.tensor([[1, *camera, *camera, 2]] * 2)
        hidden = torch.ones_like(input_ids)
        hidden[0, 1:9] = 0
        every_grid = [[1, 4, 6], [1, 6, 4]] * 2
        stock = build_prefix_position_ids(input_ids, hidden, image_grid_thw=torch.tensor(every_grid[1:]))
        assert torch.equal(core.build_fixed_grid_position_ids(input_ids, hidden, every_grid), stock)


@pytest.mark.parametrize("precompute_grid_thw", [False, True], ids=["option-off", "option-on"])
def test_adjacent_cameras_without_vision_boundaries_are_refused(tiny_backbone, precompute_grid_thw):
    """Without boundary tokens two visible cameras form one image block, which has no rope positions."""
    policy, fixed, batch, noise = _precompute_policies(
        tiny_backbone, (CAMERA, WRIST), vision_boundaries=False
    )
    with torch.no_grad(), pytest.raises(ValueError, match="at most one visible camera per sample"):
        (fixed if precompute_grid_thw else policy).predict_action_chunk(batch, noise=noise)


def test_align_query_tokens(tiny_backbone, monkeypatch):
    """A tiny RoboTwin-style align recipe appends 2 x num_task_tokens query tokens to the prefix, and a
    training step on [current, future] frames adds the distillation losses (teachers stubbed: no download)."""
    import lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 as modeling

    head = {"num_layers": 1, "num_heads": 2, "dim_head": 8, "ff_mult": 1, "num_backbone_tokens": 8}
    shared = {"share_future_depth_query": True, "use_shared_future_task_proj": True}
    align = {"mode": "query", "num_task_tokens": 4, "use_future_video": True, "depth_loss_weight": 0.004}
    align["llm"] = {"dim_out": 64, "image_token_size": 2}
    align["depth"] = {**head, "dim_out": 16, "use_future_depth": True}
    align["video"] = {**head, **shared, "dim_out": 16, "use_current_patch_loss": True}
    slots, meta = _slot_kwargs(arm=[[0, 6]], effector=[[6, 7]]), _ds_meta(7)
    cfg = _tiny_config(tiny_backbone, align_params=align, **slots)
    policy = make_policy(cfg, ds_meta=meta)
    plain = make_policy(_tiny_config(tiny_backbone, **slots), ds_meta=meta)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=meta.stats)

    images, img_masks, tokens, masks, _, grid = policy._extract_model_inputs(preprocessor(_raw_batch(7)))
    prefix_len = policy.model.embed_prefix(images, img_masks, tokens, masks, grid)[0].shape[1]
    assert prefix_len == plain.model.embed_prefix(images, img_masks, tokens, masks, grid)[0].shape[1] + 8
    assert _select_action(policy, preprocessor, postprocessor, 7, "cpu").shape == (2, 7)

    class Teachers:  # the frozen teachers' target shapes: [B, num_backbone_tokens, dim_out]
        def __init__(self, params, device):
            pass

        def targets(self, current, future, fps):
            assert current.shape == future.shape == (2, 3, 64, 64)
            names = ("depth", "future_depth", "future_video", "current_video")
            return {name: torch.randn(2, 8, 16) for name in names}

    monkeypatch.setattr(modeling, "DistillationTeachers", Teachers)
    assert cfg.image_observation_delta_indices == [0, CHUNK - 1]
    batch = _raw_batch(7)
    batch[CAMERA] = torch.stack([batch[CAMERA], batch[CAMERA].flip(-1)], dim=1)  # current, future
    policy.train()
    loss, logs = policy.forward(preprocessor(batch))
    assert {"depth_loss", "future_depth_loss", "future_video_loss", "current_video_loss"} <= logs.keys()
    loss.backward()
    assert policy.model.future_video_align_head.projector.proj_out.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("arm", "effector"),
    [([[0, 6]], [[6, 7]]), ([[0, 6], [7, 13]], [[6, 7], [13, 14]])],
    ids=["single_arm", "robotwin"],
)
def test_slot_mapping_round_trip(tiny_backbone, arm, effector):
    """Preprocessor (normalize + slot map) then postprocessor (inverse map + unnormalize) is the identity."""
    dim = max(end for _, end in arm + effector)
    cfg = _tiny_config(tiny_backbone, **_slot_kwargs(arm, effector))
    cfg.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(dim,)),
        CAMERA: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(dim,))}
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=_ds_meta(dim).stats)

    raw = _raw_batch(dim)
    canonical = preprocessor(raw)[ACTION]
    assert canonical.shape == (2, CHUNK, cfg.max_action_dim)
    # The gripper lands in the canonical effector slot (after 14 arm + 14 end-effector dims).
    effector_start = cfg.canonical_joints["arm.position"] + cfg.canonical_joints["end.position"]
    assert canonical[..., effector_start : effector_start + len(effector)].abs().sum() > 0
    torch.testing.assert_close(postprocessor(canonical), raw[ACTION], atol=1e-5, rtol=1e-5)

    # An action dim the slots leave out can't be produced by the inverse map, so it is rejected up front.
    cfg.action_slots = _slots(ACTION, [[0, 5], *arm[1:]], effector)
    with pytest.raises(ValueError, match=r"unmapped indices: \[5\]"):
        cfg.validate_features()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
    ],
)
def test_finetune_save_reload_select_action(tiny_backbone, tmp_path, device):
    """Fine-tune a 7-D checkpoint on a 6-D robot, save, then reload it the way eval/rollout do."""
    base_dir, finetuned_dir = tmp_path / "base", tmp_path / "finetuned"
    base_cfg = _tiny_config(tiny_backbone, device=device, **_slot_kwargs(arm=[[0, 6]], effector=[[6, 7]]))
    base_meta = _ds_meta(7)
    base = make_policy(base_cfg, ds_meta=base_meta)
    for obj in (base, *make_pre_post_processors(base_cfg, dataset_stats=base_meta.stats)):
        obj.save_pretrained(base_dir)

    # lerobot-train --policy.path=<base> --policy.{state,action}_slots=<6-D robot>
    cfg = PreTrainedConfig.from_pretrained(base_dir)
    cfg.pretrained_path = base_dir
    cfg.device = device
    cfg.state_slots, cfg.action_slots = _slot_kwargs(arm=[[0, 5]], effector=[[5, 6]]).values()
    cfg.tokenizer_max_length = 24  # --policy.tokenizer_max_length, applied over the saved processor's 16
    meta = _ds_meta(6)
    policy = make_policy(cfg, ds_meta=meta)
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=base_dir,
        dataset_stats=meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {
                "stats": meta.stats,
                "features": {**cfg.input_features, **cfg.output_features},
                "norm_map": cfg.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": {}},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": meta.stats,
                "features": cfg.output_features,
                "norm_map": cfg.normalization_mapping,
            },
        },
    )
    # The accelerate dataloader hands the preprocessor tensors already on the training device.
    batch = preprocessor(_raw_batch(6, device))
    assert batch[OBS_LANGUAGE_TOKENS].shape[-1] == 24
    loss, _ = policy.forward(batch)
    loss.backward()
    assert torch.isfinite(loss)
    for obj in (policy, preprocessor, postprocessor):
        obj.save_pretrained(finetuned_dir)
    saved = json.loads((finetuned_dir / "config.json").read_text())
    assert saved["input_features"][OBS_STATE]["shape"] == [6]
    assert saved["output_features"][ACTION]["shape"] == [6]

    # lerobot-eval / lerobot-rollout: everything comes from the fine-tuned checkpoint.
    cfg = PreTrainedConfig.from_pretrained(finetuned_dir)
    cfg.pretrained_path = finetuned_dir
    policy = make_policy(cfg, ds_meta=meta)
    preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=finetuned_dir)
    action = _select_action(policy, preprocessor, postprocessor, 6, device)
    assert action.shape == (2, 6)
    assert torch.isfinite(action).all()


def test_default_image_settings_match_upstream(tiny_backbone):
    """Upstream feeds 64 tokens per camera (256x256 frames at 65536 px), which the distillation heads assume."""
    from lerobot.policies.lingbot_vla_v2.processor_lingbot_vla_v2 import LingbotVLAV2ImageProcessorStep
    from lerobot.processor import TransitionKey

    defaults = LingbotVLAV2Config()
    step = LingbotVLAV2ImageProcessorStep(
        tokenizer_path=str(tiny_backbone),
        cameras=["camera_top"],
        resize_imgs_with_padding=defaults.resize_imgs_with_padding,
        image_max_pixels=defaults.image_max_pixels,
        image_min_pixels=defaults.image_min_pixels,
    )
    out = step({TransitionKey.OBSERVATION: {CAMERA: torch.rand(1, 3, 480, 640)}})
    grid = out[TransitionKey.OBSERVATION]["image_grid_thw"][0, 0]
    assert int(grid.prod()) // 4 == 64 == 8 * 8  # 2x2 merge; upstream `llm.image_token_size: 8`


@skip_if_package_missing("transformers")
def test_sparse_moe_matches_per_token_reference():
    """The padded sparse expert path equals a naive per-token top-k mixture."""
    import torch.nn.functional as F  # noqa: N812

    from lerobot.policies.lingbot_vla_v2.model_core.qwen2_action_expert import Qwen2FusedExperts

    torch.manual_seed(0)
    experts = Qwen2FusedExperts(num_experts=6, hidden_size=16, intermediate_size=8)
    hidden = torch.randn(40, 16)
    scores = torch.randn(40, 6).sigmoid()
    scores[:, 4:] = 0  # experts 4 and 5 receive no tokens
    weights, selected = torch.topk(scores, k=2, dim=-1)

    expected = torch.zeros_like(hidden)
    for k in range(selected.shape[-1]):
        e = selected[:, k]
        gate = torch.einsum("td,tjd->tj", hidden, experts.gate_proj[e])
        up = torch.einsum("td,tjd->tj", hidden, experts.up_proj[e])
        expected += weights[:, k, None] * torch.einsum("tj,tdj->td", F.silu(gate) * up, experts.down_proj[e])

    actual = experts._sparse_forward(weights, selected, hidden)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
