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

"""The export engine's chain and its ExecuTorch and AOTInductor runners, with stand-ins for the runtimes."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.inference.export import engine as export_engine
from lerobot.inference.export.engine import ExportInferenceEngine, ProgramChain, load_programs
from tests.export_helpers import fake_program, write_export_folder

# DLPack's device types.
CPU, CUDA = 1, 2


class Result:
    """Stands in for ExecuTorch's own result: device memory with a host copy on request."""

    def __init__(self, array, dtype_name="float32", device=CUDA):
        self.array, self.shape, self.dtype_name = np.asarray(array), np.shape(array), dtype_name
        self.device = device

    @classmethod
    def from_host(cls, source):
        return cls(np.array(source.array if isinstance(source, cls) else source))

    def to_host(self):
        # Like the bindings, a new host copy that later runs do not overwrite.
        return self.array.copy()

    def __dlpack_device__(self):
        return (self.device, 0)


@pytest.fixture
def executorch(monkeypatch):
    """A stand-in ExecuTorch runtime; `programs` maps a file name to its function on host arrays."""
    programs, calls, options = {}, [], []

    def load(path):
        def execute(inputs):
            assert all(isinstance(value, Result) and value.device == CUDA for value in inputs), (
                "programs only see device buffers"
            )
            calls.append(path.name)
            return [Result(out) for out in programs[path.name](*(value.array for value in inputs))]

        return SimpleNamespace(load_method=lambda name: SimpleNamespace(execute=execute))

    registry = SimpleNamespace(set_option=lambda backend, opts: options.append((backend, opts)))
    runtime = SimpleNamespace(load_program=load, backend_registry=registry)
    monkeypatch.setitem(
        sys.modules,
        "executorch.runtime",
        SimpleNamespace(Runtime=SimpleNamespace(get=lambda: runtime), ExecuTorchResult=Result),
    )
    monkeypatch.setitem(sys.modules, "torch_tensorrt_executorch_runtime", SimpleNamespace())
    return SimpleNamespace(programs=programs, calls=calls, options=options)


def chain_info(names, route="executorch_tensorrt", suffix=".pte", **info):
    return {
        "route": route,
        "inputs": info.pop("inputs"),
        "programs": {
            name: {"file": f"{name}{suffix}", "inputs": inputs, "outputs": outputs}
            for name, (inputs, outputs) in names.items()
        },
        **info,
    }


def test_denoising_steps_run_in_order_and_feed_each_other(tmp_path, executorch):
    """Each step program gets the previous one's x_next as x_t, and its timestep; the engine does no math."""
    times = []

    def step(mask, key, x_t, timestep):
        assert mask.item()
        np.testing.assert_array_equal(key, np.full((1, 3), 2.0))
        assert timestep.shape == (1,) and timestep.dtype == np.float32
        times.append(timestep.item())
        return [x_t * timestep]

    executorch.programs.update(
        {
            "prefix.pte": lambda state: [np.array([[True]]), 2 * state],
            "step.pte": step,
            "actions.pte": lambda x_t: [10 * x_t],
        }
    )
    names = {
        "prefix": (["observation.state"], ["prefix_pad_masks", "past_key_0"]),
        "step": (["prefix_pad_masks", "past_key_0", "x_t", "timestep"], ["x_next"]),
        "actions": (["x_t"], ["action"]),
    }
    info = chain_info(
        names, inputs=["observation.state", "noise"], steps=["step"], timesteps=[1.0, 0.75, 0.5]
    )
    chain = ProgramChain(load_programs(tmp_path, info), info)
    actions = chain(np.ones((1, 3), np.float32), np.ones((2, 3), np.float32))
    assert executorch.calls == ["prefix.pte", "step.pte", "step.pte", "step.pte", "actions.pte"]
    assert times == [1.0, 0.75, 0.5]
    np.testing.assert_allclose(actions, np.full((2, 3), 10 * 1.0 * 0.75 * 0.5))
    assert isinstance(actions, np.ndarray)


def test_executorch_copies_the_timesteps_to_the_gpu_once(tmp_path, executorch, monkeypatch):
    """Each copy to the GPU allocates and frees device memory, which waits for the GPU."""
    uploads = []
    from_host = Result.from_host.__func__
    monkeypatch.setattr(
        Result,
        "from_host",
        classmethod(lambda cls, value: uploads.append(np.shape(value)) or from_host(cls, value)),
    )
    executorch.programs.update(
        {"step.pte": lambda x_t, timestep: [x_t * timestep], "actions.pte": lambda x_t: [x_t]}
    )
    names = {"step": (["x_t", "timestep"], ["x_next"]), "actions": (["x_t"], ["action"])}
    info = chain_info(names, inputs=["noise"], steps=["step"], timesteps=[1.0, 0.5, 0.25])
    chain = ProgramChain(load_programs(tmp_path, info), info)
    for _ in range(2):
        np.testing.assert_allclose(chain(np.ones((2, 3), np.float32)), 0.125)
    # The three timesteps when the chain loads, then only each chunk's noise.
    assert uploads == [(1,), (1,), (1,), (2, 3), (2, 3)]


def test_a_split_prefix_reads_the_values_its_inputs_name(tmp_path, executorch):
    executorch.programs.update(
        {
            "embed.pte": lambda state: [state + 1, np.array([[True]])],
            "layers_0.pte": lambda hidden, mask: [2 * hidden, 3 * hidden],
            "layers_1.pte": lambda hidden, mask: [hidden, 5 * hidden],
            "step.pte": lambda mask, key_0, key_1, x_t, timestep: [x_t * 0 + key_0 + key_1],
            "actions.pte": lambda x_t: [x_t],
        }
    )
    names = {
        "embed": (["observation.state"], ["prefix_embs", "prefix_pad_masks"]),
        "layers_0": (["prefix_embs", "prefix_pad_masks"], ["hidden_0", "past_key_0"]),
        "layers_1": (["hidden_0", "prefix_pad_masks"], ["hidden_1", "past_key_1"]),
        "step": (["prefix_pad_masks", "past_key_0", "past_key_1", "x_t", "timestep"], ["x_next"]),
        "actions": (["x_t"], ["action"]),
    }
    info = chain_info(names, inputs=["observation.state", "noise"], steps=["step"], timesteps=[1.0])
    actions = ProgramChain(load_programs(tmp_path, info), info)(
        np.array([[1.0]], np.float32), np.zeros((1, 1), np.float32)
    )
    # key_0 = 3 * (state + 1) = 6 and key_1 = 5 * 2 * (state + 1) = 20.
    np.testing.assert_allclose(actions, [[26.0]])
    assert executorch.calls == ["embed.pte", "layers_0.pte", "layers_1.pte", "step.pte", "actions.pte"]


def test_a_whole_policy_is_a_chain_of_one(tmp_path, executorch):
    executorch.programs["model.pte"] = lambda state: [2 * state]
    info = chain_info({"model": (["observation.state"], ["action"])}, inputs=["observation.state"])
    actions = ProgramChain(load_programs(tmp_path, info), info)(np.array([[1.5]], np.float32))
    np.testing.assert_array_equal(actions, [[3.0]])


@pytest.mark.parametrize(
    "make_input", [memoryview, lambda array: Result(array, device=CPU)], ids=["buffer", "host_result"]
)
def test_executorch_copies_every_host_input_to_the_gpu(tmp_path, executorch, make_input):
    """The bindings would hand a CUDA program a host pointer without checking, so nothing host-side passes."""
    executorch.programs["model.pte"] = lambda state: [2 * state]
    info = chain_info({"model": (["observation.state"], ["action"])}, inputs=["observation.state"])
    program = load_programs(tmp_path, info)["model"]
    np.testing.assert_array_equal(program(make_input(np.ones(2, np.float32))), [2, 2])


def test_executorch_programs_of_a_chain_share_one_activation_scratch(tmp_path, executorch):
    executorch.programs["model.pte"] = lambda state: [state]
    info = chain_info({"model": (["observation.state"], ["action"])}, inputs=["observation.state"])
    load_programs(tmp_path, info)
    assert executorch.options == [("TensorRTBackend", {"use_shared_activation_scratch": True})]


class HostResult:
    """Stands in for a CPU result of ExecuTorch's bindings, read through the buffer protocol.

    The protocol exports it in its own layout, except bfloat16, which the protocol cannot describe, so
    that is exported as raw bytes in storage order.
    """

    def __init__(self, storage, view):
        self._storage, self._view = storage, view

    def __buffer__(self, flags):
        if self._view.dtype.name == "bfloat16":
            return memoryview(self._storage.reshape(-1).view(np.uint8))
        return memoryview(self._view)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("channels_last", [False, True])
def test_executorch_reads_results_in_their_own_layout(dtype, channels_last):
    from lerobot.inference.export.executorch import to_host

    if dtype == "bfloat16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    expected = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2).astype(dtype)
    # A channels-last result stores each pixel's channels together and keeps the logical shape.
    storage = expected.transpose(0, 2, 3, 1).copy() if channels_last else expected.copy()
    view = storage.transpose(0, 3, 1, 2) if channels_last else storage
    result = SimpleNamespace(
        to_host=lambda: HostResult(storage, view),
        dtype_name=np.dtype(dtype).name,
        shape=view.shape,
        strides=view.strides,
    )
    host = to_host(result)
    # The bindings' to_host already made the copy, so reading it copies nothing more.
    assert np.shares_memory(host, storage)
    assert host.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(host.astype(np.float32), expected.astype(np.float32))


def test_executorch_host_copies_survive_the_next_run(tmp_path, executorch):
    buffer = np.zeros((1, 2), np.float32)

    def run(state):
        buffer[:] = state
        return [buffer]

    executorch.programs["model.pte"] = run
    info = chain_info({"model": (["observation.state"], ["action"])}, inputs=["observation.state"])
    chain = ProgramChain(load_programs(tmp_path, info), info)
    first = chain(np.ones((1, 2), np.float32))
    chain(np.zeros((1, 2), np.float32))
    np.testing.assert_array_equal(first, 1.0)


@pytest.mark.parametrize("missing", ["old_executorch", "no_backend"])
def test_executorch_names_the_packages_it_needs(tmp_path, executorch, monkeypatch, missing):
    """An ExecuTorch from before ExecuTorchResult, or no TensorRT backend, is refused when the folder loads."""
    if missing == "old_executorch":
        runtime = sys.modules["executorch.runtime"]
        monkeypatch.setitem(sys.modules, "executorch.runtime", SimpleNamespace(Runtime=runtime.Runtime))
    else:
        monkeypatch.setitem(sys.modules, "torch_tensorrt_executorch_runtime", None)
    info = chain_info({"model": (["observation.state"], ["action"])}, inputs=["observation.state"])
    with pytest.raises(ImportError, match="1.6.0.dev20261021 or newer"):
        load_programs(tmp_path, info)


def test_unknown_routes_are_refused(tmp_path):
    """A folder exported before the routes were renamed names an old route, and must be exported again."""
    with pytest.raises(ValueError, match="Unknown export route 'onnx_trt'"):
        load_programs(tmp_path, {"route": "onnx_trt", "programs": {}})


@pytest.mark.parametrize("image_first", [False, True])
def test_engine_feeds_the_raw_frame_and_fresh_noise_and_plays_the_chunk(tmp_path, monkeypatch, image_first):
    from safetensors.numpy import save_file

    state = np.array([1, 2, 3], dtype=np.float32)
    image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    noise = np.ones((4, 3), dtype=np.float32)
    expected = np.stack([state, 2 * state])
    names = ["observation.state", "observation.images.camera"]
    if image_first:
        names.reverse()
    frame = {"observation.state": state, "observation.images.camera": image}
    info = chain_info(
        {"model": ([*names, "noise"], ["action"])},
        route="onnx_tensorrt",
        suffix=".engine",
        inputs=[*names, "noise"],
        noise_shape=[4, 3],
        task="pick",
        test_case="case.safetensors",
        tolerance=0.0,
    )
    (tmp_path / "export.json").write_text(json.dumps(info))
    save_file({**frame, "noise": noise, "expected_actions": expected}, tmp_path / "case.safetensors")
    calls = []

    def program(*arrays):
        assert len(arrays) == 3
        for name, value in zip(names, arrays[:-1], strict=True):
            np.testing.assert_array_equal(value, frame[name])
            assert value.dtype == frame[name].dtype and value.flags.c_contiguous
        assert arrays[-1].shape == (4, 3) and arrays[-1].dtype == np.float32
        calls.append(arrays[-1].copy())
        return expected

    monkeypatch.setattr(export_engine, "load_programs", lambda *args: {"model": program})
    engine = ExportInferenceEngine(tmp_path, "pick", {"camera": "observation.images.camera"})
    np.testing.assert_array_equal(calls[0], noise)
    live = {
        "observation.state": state,
        "camera": np.asfortranarray(image),
        "unused.image": np.zeros(1, np.uint8),
    }
    np.testing.assert_array_equal(engine.get_action(live), state)
    np.testing.assert_array_equal(engine.get_action(live), 2 * state)
    assert len(calls) == 2
    assert not np.array_equal(calls[1], noise)


@pytest.mark.parametrize("chain", [False, True])
def test_torch_tensorrt_package_takes_host_arrays_and_returns_host_actions(tmp_path, monkeypatch, chain):
    """The AOTInductor runner hands the package tensors and returns NumPy, alone and as a chain."""
    import torch

    loaded, received = [], []

    def load_package(path):
        loaded.append(Path(path).name)

        def run(*inputs):
            received.append((Path(path).name, [type(x) for x in inputs]))
            if Path(path).name == "prefix.pt2":
                return [inputs[0] * 2]
            if Path(path).name == "step.pt2":
                return [inputs[0] + inputs[1] * 0]
            return [inputs[0] * 10]

        return run

    cudagraphs = []
    runtime = SimpleNamespace(set_cudagraphs_mode=cudagraphs.append)
    monkeypatch.setitem(sys.modules, "torch_tensorrt", SimpleNamespace(runtime=runtime))
    monkeypatch.setattr(torch._inductor, "aoti_load_package", load_package)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs: self)
    if chain:
        names = {
            "prefix": (["observation.state"], ["past_key_0"]),
            "step": (["past_key_0", "x_t", "timestep"], ["x_next"]),
            "actions": (["x_t"], ["action"]),
        }
        info = {
            "route": "torch_tensorrt",
            "inputs": ["observation.state", "noise"],
            "programs": {
                name: {"file": f"{name}.pt2", "inputs": inputs, "outputs": outputs}
                for name, (inputs, outputs) in names.items()
            },
            "steps": ["step"],
            "timesteps": [1.0],
        }
        actions = ProgramChain(load_programs(tmp_path, info), info)(
            np.ones((1, 3), np.float32), np.zeros((1, 3), np.float32)
        )
        assert loaded == ["prefix.pt2", "step.pt2", "actions.pt2"]
        np.testing.assert_allclose(actions, np.full((1, 3), 20.0))
    else:
        info = chain_info(
            {"model": (["observation.state"], ["action"])},
            "torch_tensorrt",
            ".pt2",
            inputs=["observation.state"],
        )
        actions = ProgramChain(load_programs(tmp_path, info), info)(np.full((1, 3), 2.0, np.float32))
        assert loaded == ["model.pt2"]
        np.testing.assert_array_equal(actions, np.full((1, 3), 20.0))
    assert isinstance(actions, np.ndarray)
    assert all(kind is torch.Tensor for _, kinds in received for kind in kinds), received
    assert cudagraphs and all(cudagraphs)


@pytest.fixture
def cpu_export_engine(monkeypatch):
    """Builds an export engine around `fake_program`, on the CPU."""
    from lerobot.inference.export import engine as export_engine

    monkeypatch.setattr(export_engine, "load_programs", lambda folder, info: {"model": fake_program})
    return lambda folder: export_engine.ExportInferenceEngine(folder, task="")


def test_export_dir_tells_an_exported_folder_from_a_checkpoint(tmp_path):
    from lerobot.inference.export import export_dir

    assert export_dir(tmp_path) is None
    assert export_dir(None) is None
    write_export_folder(tmp_path, [[0, 1, 2], [0, 2, 4]])
    assert export_dir(tmp_path) == tmp_path


def test_export_engine_plays_one_chunk_one_action_per_call(tmp_path, cpu_export_engine):
    import numpy as np

    write_export_folder(tmp_path, [[0, 1, 2], [0, 2, 4]])
    engine = cpu_export_engine(tmp_path)
    frame = {"observation.state": np.array([1, 2, 3], dtype=np.float32)}

    with patch.object(engine, "_run_chunk", wraps=engine._run_chunk) as run_chunk:
        actions = [engine.get_action(frame) for _ in range(3)]

    np.testing.assert_array_equal(actions[0], [1, 2, 3])
    np.testing.assert_array_equal(actions[1], [2, 4, 6])
    np.testing.assert_array_equal(actions[2], [1, 2, 3])
    assert run_chunk.call_count == 2
    assert len(engine.inference_times) == 2


def test_export_engine_refuses_a_program_that_misses_its_test_case(tmp_path, cpu_export_engine):
    write_export_folder(tmp_path, [[0, 1, 2], [0, 9, 9]])
    with pytest.raises(RuntimeError, match="does not reproduce its test case"):
        cpu_export_engine(tmp_path)


@pytest.mark.parametrize("bad", ["nan", "shape"])
def test_export_engine_refuses_nan_actions_and_a_wrong_shape(tmp_path, monkeypatch, bad):
    import numpy as np

    from lerobot.inference.export import engine as export_engine

    write_export_folder(tmp_path, [[0, 1, 2], [0, 2, 4]])

    def program(state):
        chunk = fake_program(state)
        # One step more than the test case: comparing only the overlap would match.
        return np.full_like(chunk, np.nan) if bad == "nan" else np.concatenate([chunk, chunk[:, :1]], axis=1)

    monkeypatch.setattr(export_engine, "load_programs", lambda folder, info: {"model": program})
    message = "does not reproduce its test case" if bad == "nan" else r"shaped \(2, 4\), but its test case"
    with pytest.raises(RuntimeError, match=message):
        export_engine.ExportInferenceEngine(tmp_path, task="")


@pytest.mark.parametrize("event", ["task", "reset"])
def test_export_engine_drops_the_queued_chunk_on_a_task_change_or_reset(tmp_path, cpu_export_engine, event):
    import numpy as np

    write_export_folder(tmp_path, [[0, 1, 2], [0, 2, 4]])
    engine = cpu_export_engine(tmp_path)
    frame = {"observation.state": np.array([1, 2, 3], dtype=np.float32)}
    with patch.object(engine, "_run_chunk", wraps=engine._run_chunk) as run_chunk:
        engine.get_action(frame)
        if event == "task":
            assert engine.set_task("place")
        else:
            engine.reset()
        np.testing.assert_array_equal(engine.get_action(frame), [1, 2, 3])
    assert run_chunk.call_count == 2


def test_export_engine_applies_the_rename_map_to_the_robot_frame(tmp_path, monkeypatch):
    import numpy as np

    from lerobot.inference.export import engine as export_engine

    write_export_folder(tmp_path, [[0, 1, 2], [0, 2, 4]])
    monkeypatch.setattr(export_engine, "load_programs", lambda folder, info: {"model": fake_program})
    engine = export_engine.ExportInferenceEngine(
        tmp_path, task="", rename_map={"observation.joints": "observation.state"}
    )

    action = engine.get_action({"observation.joints": np.array([1, 2, 3], dtype=np.float32)})

    np.testing.assert_array_equal(action, [1, 2, 3])


def test_export_engine_refuses_another_task_before_loading_the_programs(tmp_path, monkeypatch):
    """Loading a large chain takes long on a small board, so a wrong --task is refused first."""
    import json

    from lerobot.inference.export import engine as export_engine

    write_export_folder(tmp_path, [[0, 1, 2], [0, 2, 4]])
    info = json.loads((tmp_path / "export.json").read_text())
    (tmp_path / "export.json").write_text(json.dumps({**info, "task": "pick"}))
    monkeypatch.setattr(export_engine, "load_programs", lambda *args: pytest.fail("loaded the programs"))
    with pytest.raises(ValueError, match="only runs the task it was exported for"):
        export_engine.ExportInferenceEngine(tmp_path, task="place")


def test_export_engine_refuses_another_task_than_a_fixed_task_export(tmp_path, cpu_export_engine):
    import json

    import numpy as np

    write_export_folder(tmp_path, [[0, 1, 2], [0, 2, 4]])
    info = json.loads((tmp_path / "export.json").read_text())
    (tmp_path / "export.json").write_text(json.dumps({**info, "task": "pick"}))

    # Without --task, the rollout runs the task the folder was exported for.
    assert cpu_export_engine(tmp_path).task == "pick"

    from lerobot.inference.export import engine as export_engine

    engine = export_engine.ExportInferenceEngine(tmp_path, task="pick")
    frame = {"observation.state": np.array([1, 2, 3], dtype=np.float32)}
    np.testing.assert_array_equal(engine.get_action(frame), [1, 2, 3])
    assert "only runs the task it was exported for" in engine.text_input_error("place", instruction=True)
    assert engine.text_input_error("place") is None
    assert not engine.set_task("place")
    assert engine.task == "pick"

    from lerobot.rollout.controller import RolloutController
    from tests.interactive_helpers import _make_ctx

    ctx, strategy, *_ = _make_ctx()
    ctx.policy.inference = engine
    assert not RolloutController(strategy, ctx).set_task("place")
    assert engine.task == "pick"
    np.testing.assert_array_equal(engine.get_action(frame), [2, 4, 6])
    engine.reset()
    engine.get_action(frame)
    assert engine.task == engine.dispatched_task == "pick"


def test_export_engine_holds_the_robot_until_the_planner_replies(tmp_path, cpu_export_engine):
    import numpy as np

    write_export_folder(tmp_path, [[0, 1, 2], [0, 2, 4]])
    engine = cpu_export_engine(tmp_path)
    engine.external_text = lambda obs, query, task: "pick up the cup"
    assert engine.dispatch_allowed()
    engine.start_autosteer("clear the table", interval_s=0.0)
    assert engine.hold_for_planner()
    assert not engine.dispatch_allowed()
    engine.stop_autosteer()
    assert engine.dispatch_allowed()
    frame = {"observation.state": np.array([1, 2, 3], dtype=np.float32)}
    np.testing.assert_array_equal(engine.get_action(frame), [1, 2, 3])
