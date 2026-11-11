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

"""Gives Praxis-1 its fake frozen parts in the export's steps, as the test that saved the checkpoint does."""

from types import SimpleNamespace

from tests.export.tiny_praxis import use_tiny_praxis

use_tiny_praxis(SimpleNamespace(setattr=setattr))
