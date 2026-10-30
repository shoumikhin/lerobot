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

"""A tiny random Qwen2.5-VL vision tower, for the stand-in policies whose recipes build its vision grid."""

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel


def tiny_vision_tower() -> Qwen2_5_VisionTransformerPretrainedModel:
    """A vision tower of one block with full attention, in evaluation mode."""
    vision = Qwen2_5_VLVisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=32,
        fullatt_block_indexes=[0],
    )
    return Qwen2_5_VisionTransformerPretrainedModel._from_config(vision).eval()
