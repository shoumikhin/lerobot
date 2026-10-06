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

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.rollout.inference.export.engine import ExportInferenceEngine, load_program


@pytest.mark.parametrize("noise_shape", [(1, 2, 3), (2, 3)])
def test_executorch_three_program_loop(tmp_path, monkeypatch, noise_shape):
    calls = []
    times = []

    def step(mask, key, sample, timestep):
        assert mask.item()
        np.testing.assert_array_equal(key, np.full((1, 3), 2.0))
        assert timestep.shape == (1,) and timestep.dtype == np.float32
        times.append(timestep.item())
        return [sample * timestep]

    programs = {
        "prefix.pte": lambda state: [np.array([[True]]), 2 * state],
        "step.pte": step,
        "actions.pte": lambda sample: [10 * sample],
    }

    def load(path, data_path=None):
        def execute(inputs):
            calls.append(path.name)
            assert all(isinstance(value, np.ndarray) for value in inputs)
            return programs[path.name](*inputs)

        return SimpleNamespace(load_method=lambda name: SimpleNamespace(execute=execute))

    runtime = SimpleNamespace(load_program=load)
    monkeypatch.setitem(sys.modules, "executorch.runtime", SimpleNamespace(Runtime=SimpleNamespace(get=lambda: runtime)))
    monkeypatch.setitem(sys.modules, "torch_tensorrt_executorch_runtime", SimpleNamespace())
    info = {
        "backend": "executorch_tensorrt",
        "programs": {name: {"file": f"{name}.pte"} for name in ("prefix", "step", "actions")},
        "num_steps": 4,
    }
    program = load_program(tmp_path, info)
    actions = program(np.ones((1, 3)), np.ones(noise_shape))
    assert actions.shape == noise_shape
    assert calls == ["prefix.pte", *(["step.pte"] * 4), "actions.pte"]
    assert times == [1.0, 0.75, 0.5, 0.25]
    expected = 10 * (1 - 0.25) * (1 - 0.1875) * (1 - 0.125) * (1 - 0.0625)
    np.testing.assert_allclose(actions, expected)


def test_executorch_output_survives_next_execution(tmp_path, monkeypatch):
    buffer = np.zeros((1, 2), dtype=np.float32)

    def execute(inputs):
        buffer[:] = inputs[0]
        return [buffer]

    runtime = SimpleNamespace(load_program=lambda *args, **kw: SimpleNamespace(
        load_method=lambda name: SimpleNamespace(execute=execute)))
    monkeypatch.setitem(sys.modules, "executorch.runtime", SimpleNamespace(Runtime=SimpleNamespace(get=lambda: runtime)))
    program = load_program(tmp_path, {"backend": "executorch_cuda", "file": "model.pte"})
    first = program(np.ones((1, 2), dtype=np.float32))
    program(np.zeros((1, 2), dtype=np.float32))
    np.testing.assert_array_equal(first, 1.0)


def test_executorch_bfloat16_compatibility_preserves_values_and_ownership(tmp_path, monkeypatch):
    import torch

    ml_dtypes = pytest.importorskip("ml_dtypes")
    buffer = torch.zeros((1, 2), dtype=torch.bfloat16)
    input_types = []

    def execute(inputs):
        input_types.append(type(inputs[0]))
        if not isinstance(inputs[0], torch.Tensor):
            raise TypeError("Tensor inputs required")
        buffer.copy_(inputs[0])
        return [buffer]

    runtime = SimpleNamespace(load_program=lambda *args, **kw: SimpleNamespace(
        load_method=lambda name: SimpleNamespace(execute=execute)))
    monkeypatch.setitem(sys.modules, "executorch.runtime", SimpleNamespace(Runtime=SimpleNamespace(get=lambda: runtime)))
    program = load_program(tmp_path, {"backend": "executorch_cuda", "file": "model.pte"})
    first = program(np.array([[1.25, -2.5]], dtype=ml_dtypes.bfloat16))
    second = program(np.zeros((1, 2), dtype=ml_dtypes.bfloat16))
    assert first.dtype == np.dtype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(first.astype(np.float32), [[1.25, -2.5]])
    np.testing.assert_array_equal(second.astype(np.float32), [[0, 0]])
    assert input_types == [np.ndarray, torch.Tensor, torch.Tensor]


def test_executorch_invalid_step_count(tmp_path):
    from lerobot.rollout.inference.export.executorch import ExecuTorchDenoisingLoop

    with pytest.raises(ValueError, match="num_steps"):
        ExecuTorchDenoisingLoop(tmp_path, {}, 0)


@pytest.mark.parametrize("image_first", [False, True])
def test_raw_frame_engine_preserves_inputs_noise_and_action_queue(tmp_path, monkeypatch, image_first):
    from safetensors.numpy import save_file

    from lerobot.rollout.inference.export import engine as module

    state = np.array([1, 2, 3], dtype=np.float32)
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    noise = np.ones((4, 3), dtype=np.float32)
    expected = np.stack([state, 2 * state])
    names = ["observation.state", "observation.images.camera"]
    if image_first:
        names.reverse()
    frame = {"observation.state": state, "observation.images.camera": image}
    info = {
        "backend": "onnx_tensorrt", "file": "model.engine", "inputs": [*names, "noise"],
        "output": "action", "noise_shape": [4, 3], "task": "pick", "task_fixed": True,
        "test_case": "case.safetensors", "tolerance": 0.0,
    }
    (tmp_path / "export.json").write_text(json.dumps(info))
    save_file({**frame, "noise": noise, "expected_actions": expected}, tmp_path / "case.safetensors")
    calls = []

    class Program:
        input_shape = frame[names[0]].shape

        def __call__(self, *arrays):
            assert len(arrays) == 3
            for name, value in zip(names, arrays[:-1], strict=True):
                np.testing.assert_array_equal(value, frame[name])
                assert value.dtype == frame[name].dtype and value.flags.c_contiguous
            assert arrays[-1].shape == (4, 3) and arrays[-1].dtype == np.float32
            calls.append(arrays[-1].copy())
            return expected

    monkeypatch.setattr(module, "load_program", lambda *args: Program())
    engine = ExportInferenceEngine(tmp_path, "pick", "replay", {"camera": "observation.images.camera"})
    np.testing.assert_array_equal(calls[0], noise)
    live = {"observation.state": state, "camera": np.asfortranarray(image),
            "unused.image": np.zeros(1, dtype=np.uint8)}
    np.testing.assert_array_equal(engine.get_action(live), state)
    np.testing.assert_array_equal(engine.get_action(live), 2 * state)
    assert len(calls) == 2
    assert not np.array_equal(calls[1], noise)
    with pytest.raises(ValueError, match="only runs the task"):
        engine._check_task("place")


def test_legacy_text_pipeline_keeps_state_not_listed_as_program_input(tmp_path, monkeypatch):
    from safetensors.numpy import save_file

    from lerobot.rollout.inference.export import engine as module

    state = np.array([1, 2], dtype=np.float32)
    (tmp_path / "export.json").write_text(json.dumps({
        "backend": "onnx_tensorrt", "file": "model.engine", "inputs": ["tokens"],
        "output": "action", "test_case": "case.safetensors", "tolerance": 0.0,
        "text_steps": "text.json",
    }))
    save_file({"observation.state": state, "expected_actions": state[None]}, tmp_path / "case.safetensors")
    def pipeline(frame):
        return {"tokens": frame["observation.state"]}

    monkeypatch.setattr("lerobot.processor.PolicyProcessorPipeline.from_pretrained", lambda *a, **kw: pipeline)
    monkeypatch.setattr(module, "load_program", lambda *a: lambda tokens: tokens[:, None])
    engine = ExportInferenceEngine(tmp_path, "pick", "replay")
    np.testing.assert_array_equal(engine.get_action({"observation.state": state}), state)
