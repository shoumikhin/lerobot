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

"""Camera geometry for exports whose compiled programs cannot resize with antialiasing."""

import numpy as np
from PIL import Image


def resize_bicubic(image: np.ndarray, size: tuple[int, int] | list[int]) -> np.ndarray:
    """Resize a uint8 HWC image with antialiased bicubic sampling, as the policies' torchvision transforms do."""
    if image.dtype != np.uint8 or image.ndim != 3 or min(image.shape) == 0:
        raise ValueError("Expected a nonempty uint8 HWC image.")
    if len(size) != 2 or min(size) <= 0:
        raise ValueError("Expected a positive output height and width.")
    if image.shape[:2] == tuple(size):
        return image
    return np.asarray(Image.fromarray(image).resize((size[1], size[0]), Image.Resampling.BICUBIC))


def resize_images(frame: dict[str, np.ndarray], geometry: dict) -> dict[str, np.ndarray]:
    """Apply the export's saved resize/crop geometry without changing the caller's frame."""
    result = dict(frame)
    for name in geometry["cameras"]:
        image = frame[name]
        if geometry["target_size"]:
            if geometry["letterbox"]:
                height, width = image.shape[:2]
                side = max(height, width)
                image = np.pad(
                    image,
                    (
                        ((side - height) // 2, (side - height + 1) // 2),
                        ((side - width) // 2, (side - width + 1) // 2),
                        (0, 0),
                    ),
                )
            edge = geometry["resize_edge"]
            image = resize_bicubic(image, (edge, edge))
            fraction = geometry["crop_fraction"]
            if fraction is not None and 0 < fraction < 1:
                crop = max(1, round(edge * fraction))
                offset = (edge - crop) // 2
                image = image[offset : offset + crop, offset : offset + crop]
            image = resize_bicubic(image, geometry["target_size"])
        result[name] = np.ascontiguousarray(resize_bicubic(image, geometry["image_size"]))
    return result
