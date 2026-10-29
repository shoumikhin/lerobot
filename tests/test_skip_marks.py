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

"""The GPU and export jobs run only the tests marked `gpu` or `executorch`, so a skip that needs one fails unmarked."""

from types import SimpleNamespace

import pytest

from tests.conftest import pytest_runtest_makereport


class _Item:
    nodeid = "test_it"

    def __init__(self, *marks: str):
        self.marks = marks

    def get_closest_marker(self, name: str):
        return name if name in self.marks else None


def _outcome(item: _Item, reason: str) -> str:
    report = SimpleNamespace(
        skipped=True, outcome="skipped", longrepr=("test_it.py", 1, f"Skipped: {reason}")
    )
    hook = pytest_runtest_makereport(item, None)
    next(hook)
    with pytest.raises(StopIteration):
        hook.send(SimpleNamespace(get_result=lambda: report))
    return report.outcome


@pytest.mark.parametrize(
    "reason, mark", [("could not import 'executorch.exir'", "executorch"), ("requires cuda", "gpu")]
)
def test_a_skip_for_what_only_one_job_has_fails_without_its_mark(reason, mark):
    assert _outcome(_Item(), reason) == "failed"
    assert _outcome(_Item(mark), reason) == "skipped"


def test_other_skips_stay_skips():
    assert _outcome(_Item(), "python-can not available") == "skipped"
