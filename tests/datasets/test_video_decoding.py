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

from pathlib import Path

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("torchcodec", reason="torchcodec is required (install lerobot[dataset])")

from lerobot.datasets.video_utils import decode_video_frames  # noqa: E402

pytestmark = pytest.mark.dataset

TEST_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "encoded_videos"


def test_decode_video_frames_torchcodec():
    video_path = TEST_ARTIFACTS_DIR / "clip_4frames.mp4"
    # Out of order and repeated, so a decoder that ignores or sorts the requested timestamps fails.
    timestamps = [3 / 30, 1 / 30, 3 / 30, 0]
    frames = decode_video_frames(video_path, timestamps, tolerance_s=1e-4, backend="torchcodec")
    frames_uint8 = decode_video_frames(
        video_path, timestamps, tolerance_s=1e-4, backend="torchcodec", return_uint8=True
    )

    assert frames.shape == frames_uint8.shape == (4, 3, 64, 96)
    assert frames.device.type == frames_uint8.device.type == "cpu"
    assert frames_uint8.dtype == torch.uint8
    assert (frames >= 0).all() and (frames <= 1).all()
    torch.testing.assert_close(frames, frames_uint8.float() / 255)

    expected_uint8 = decode_video_frames(
        video_path, timestamps, tolerance_s=1e-4, backend="pyav", return_uint8=True
    )
    # The two backends can use different FFmpeg builds, and their SIMD color conversions round differently,
    # by up to 3 levels. A wrong frame differs by more than 200 levels.
    torch.testing.assert_close(frames_uint8, expected_uint8, atol=3, rtol=0)
