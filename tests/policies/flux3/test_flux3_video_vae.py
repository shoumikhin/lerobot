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
"""The video VAE attention without NATTEN: NATTEN's neighborhoods in plain PyTorch, also under torch.export."""

import itertools

import pytest
import torch

from lerobot.policies.flux3.f3 import video_vae
from lerobot.policies.flux3.f3.video_vae import Natten3D, neighborhood_attention

# The keys each query of one axis attends to, measured on NATTEN 0.21.6 and 0.21.7:
# (axis size, window, causal) -> keys.
NATTEN_WINDOWS = {
    (7, 5, False): [[0, 1, 2, 3, 4]] * 3 + [[1, 2, 3, 4, 5]] + [[2, 3, 4, 5, 6]] * 3,
    (7, 5, True): [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]],
    (6, 4, False): [[0, 1, 2, 3]] * 3 + [[1, 2, 3, 4]] + [[2, 3, 4, 5]] * 2,
}


@pytest.mark.parametrize(("size", "kernel", "causal"), list(NATTEN_WINDOWS))
def test_windows_match_natten(size, kernel, causal):
    # Zero queries attend uniformly, so one-hot values show which keys each query sees.
    q = torch.zeros(1, size, 1, size)
    v = torch.eye(size).reshape(1, size, 1, size)
    out = neighborhood_attention(q, torch.randn(1, size, 1, size), v, [kernel], [causal])[0, :, 0]
    expected = torch.zeros(size, size)
    for query, keys in enumerate(NATTEN_WINDOWS[(size, kernel, causal)]):
        expected[query, keys] = 1 / len(keys)
    torch.testing.assert_close(out, expected)


def _reference(q, k, v, kernel_size, is_causal):
    """Softmax attention of each query over the product of its per-axis windows, one query at a time."""
    spatial = q.shape[1:-2]
    out = torch.empty(q.shape, dtype=torch.float64)
    for position in itertools.product(*map(range, spatial)):
        axes = []
        for i, size, kernel, causal in zip(position, spatial, kernel_size, is_causal, strict=True):
            start = min(max(i - (kernel - 1 if causal else kernel // 2), 0), size - kernel)
            axes.append([j for j in range(start, start + kernel) if not causal or j <= i])
        keys = torch.stack([k[(slice(None), *j)] for j in itertools.product(*axes)], -2).double()
        values = torch.stack([v[(slice(None), *j)] for j in itertools.product(*axes)], -2).double()
        query = q[(slice(None), *position)].double().unsqueeze(-1)
        weights = ((keys @ query).squeeze(-1) * q.shape[-1] ** -0.5).softmax(-1)
        out[(slice(None), *position)] = (weights.unsqueeze(-2) @ values).squeeze(-2)
    return out


@pytest.mark.parametrize(
    ("spatial", "kernel_size", "is_causal"),
    [
        ((5, 7), (3, 5), (False, False)),
        ((6, 6), (4, 3), (False, False)),
        ((4, 5, 6), (3, 3, 5), (True, False, False)),
        ((5, 4, 4), (5, 3, 3), (True, True, False)),
    ],
)
@pytest.mark.parametrize("chunked", [False, True])
def test_matches_reference(spatial, kernel_size, is_causal, chunked, monkeypatch):
    if chunked:
        monkeypatch.setattr(video_vae, "_NEIGHBORHOOD_CHUNK_ELEMENTS", 1)  # one query row per chunk
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 2, *spatial, 2, 8).unbind(0)
    out = neighborhood_attention(q, k, v, kernel_size, is_causal)
    torch.testing.assert_close(out, _reference(q, k, v, kernel_size, is_causal).float())


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
    ],
)
def test_computes_in_float32_under_autocast(device):
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 1, 4, 5, 6, 2, 16, device=device).unbind(0)
    expected = neighborhood_attention(q, k, v, [3, 3, 3], [True, False, False])
    with torch.autocast(device, dtype=torch.bfloat16):
        out = neighborhood_attention(q, k, v, [3, 3, 3], [True, False, False])
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_computes_bfloat16_inputs_in_float32():
    # The VAE runs in bfloat16, and bfloat16 products would round the logits.
    torch.manual_seed(0)
    q, k, v = (4 * torch.randn(3, 1, 4, 5, 6, 2, 16)).bfloat16().unbind(0)
    out = neighborhood_attention(q, k, v, [3, 3, 3], [True, False, False])
    expected = neighborhood_attention(q.float(), k.float(), v.float(), [3, 3, 3], [True, False, False])
    torch.testing.assert_close(out, expected.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize(("frames", "moved_frame", "changed_frames"), [(1, 0, [0]), (4, 1, [1, 2, 3])])
def test_attention_block_attends_its_window(frames, moved_frame, changed_frames):
    # Moving one input token changes the outputs whose window holds it: a window of 5 rows reaches all 7 rows
    # from row 3, a window of 3 columns reaches columns 2 to 4, and a video is causal in time.
    torch.manual_seed(0)
    block = Natten3D(16, [3, 5, 3], 2, causal=True).eval()
    x = torch.randn(1, frames, 7, 7, 16)
    moved = x.clone()
    moved[0, moved_frame, 3, 3] += 1
    with torch.no_grad():
        changed = (block(moved) - block(x)).abs().amax(-1)[0] > 0
    expected = torch.zeros(frames, 7, 7, dtype=torch.bool)
    expected[changed_frames, :, 2:5] = True
    assert torch.equal(changed, expected)


@pytest.mark.parametrize("frames", [1, 4])
def test_attention_block_exports(frames):
    torch.manual_seed(0)
    block = Natten3D(32, [3, 3, 3], 2, causal=True, qk_norm=True).eval()
    x = torch.randn(1, frames, 5, 6, 32)
    program = torch.export.export(block, (x,))
    torch.testing.assert_close(program.module()(x), block(x))


def test_attention_block_gives_natten_kernels_no_cpu_tensors(monkeypatch):
    def natten_kernel(*args, **kwargs):
        raise AssertionError("NATTEN's kernel got a CPU tensor")

    monkeypatch.setattr(video_vae, "_natten_available", True)
    monkeypatch.setattr(video_vae, "na2d", natten_kernel, raising=False)
    monkeypatch.setattr(video_vae, "na3d", natten_kernel, raising=False)
    torch.manual_seed(0)
    block = Natten3D(16, [3, 3, 3], 2, causal=True).eval()
    for frames in (1, 3):
        with torch.no_grad():
            assert block(torch.randn(1, frames, 4, 4, 16)).shape == (1, frames, 4, 4, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("causal", [True, False])
def test_attention_block_gives_natten_kernels_cuda_tensors(causal, monkeypatch):
    calls = []

    def recorder(name):
        def natten_kernel(q, k, v, kernel_size, is_causal, attention_kwargs):
            assert q.is_cuda and k.is_cuda and v.is_cuda
            assert attention_kwargs == {"backend": "recorded"}
            calls.append((name, tuple(q.shape), list(kernel_size), list(is_causal)))
            return v

        return natten_kernel

    monkeypatch.setattr(video_vae, "_natten_available", True)
    monkeypatch.setattr(video_vae, "na2d", recorder("na2d"), raising=False)
    monkeypatch.setattr(video_vae, "na3d", recorder("na3d"), raising=False)
    monkeypatch.setattr(video_vae, "_natten_attention_kwargs", lambda q, k, v: {"backend": "recorded"})
    torch.manual_seed(0)
    block = Natten3D(16, [3, 5, 3], 2, causal=causal).eval().cuda()
    for frames in (1, 4):
        with torch.no_grad():
            assert block(torch.randn(1, frames, 6, 6, 16, device="cuda")).shape == (1, frames, 6, 6, 16)
    assert calls == [
        ("na2d", (1, 6, 6, 2, 8), [5, 3], [False, False]),
        ("na3d", (1, 4, 6, 6, 2, 8), [3, 5, 3], [causal, False, False]),
    ]
