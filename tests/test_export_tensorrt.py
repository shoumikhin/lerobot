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

import ctypes
import functools
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.rollout.inference.export import tensorrt as backend


@pytest.fixture
def fake_cuda(monkeypatch):
    allocations = {}
    pending = []
    copies = []

    def malloc(size):
        memory = ctypes.create_string_buffer(size)
        pointer = ctypes.addressof(memory)
        allocations[pointer] = memory
        return 0, pointer

    def free(pointer):
        allocations.pop(pointer)
        return (0,)

    def memcpy(dst, src, size, kind, stream):
        copies.append(kind)
        pending.append(lambda: ctypes.memmove(dst, src, size))
        return (0,)

    def synchronize(stream):
        while pending:
            pending.pop(0)()
        return (0,)

    cuda = SimpleNamespace(
        cudaMalloc=malloc, cudaFree=free, cudaMemcpyAsync=memcpy,
        cudaMemcpyKind=SimpleNamespace(cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2),
        cudaStreamCreate=lambda: (0, 123), cudaStreamSynchronize=synchronize,
        cudaStreamDestroy=lambda stream: (0,), allocations=allocations, pending=pending, copies=copies,
    )
    monkeypatch.setattr(backend, "cuda_runtime", lambda: cuda)
    return cuda


@pytest.fixture
def fake_engine(monkeypatch, fake_cuda):
    contexts = []

    class Context:
        def __init__(self):
            self.addresses = {}
            contexts.append(self)

        def set_tensor_address(self, name, pointer):
            self.addresses[name] = pointer
            return True

        def execute_async_v3(self, stream):
            def execute():
                src = np.ctypeslib.as_array((ctypes.c_float * 3).from_address(self.addresses["x"]))
                dst = np.ctypeslib.as_array((ctypes.c_float * 3).from_address(self.addresses["y"]))
                dst[:] = 2 * src

            fake_cuda.pending.append(execute)
            return True

        def set_device_memory(self, pointer, size):
            self.scratch = (pointer, size)

    engine = SimpleNamespace(
        create_execution_context=lambda strategy: Context(),
        get_tensor_shape=lambda name: (1, 3), get_tensor_dtype=lambda name: "float",
        device_memory_size_v2=64,
    )
    trt = SimpleNamespace(
        nptype=lambda dtype: np.float32,
        bfloat16="bfloat16",
        ExecutionContextAllocationStrategy=SimpleNamespace(STATIC=0, USER_MANAGED=1),
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(backend, "load_engine", lambda path: engine)
    return contexts


def test_engine_copies_host_inputs_and_owns_returned_actions(tmp_path, fake_cuda, fake_engine):
    engine = backend.TensorRTEngine(tmp_path / "model.engine", ["x"], ["y"])
    values = np.arange(6, dtype=np.float64).reshape(1, 6)[:, ::2]
    first = engine(values)
    np.testing.assert_array_equal(first, [[0, 4, 8]])
    np.testing.assert_array_equal(engine(np.ones((1, 3))), [[2, 2, 2]])
    np.testing.assert_array_equal(first, [[0, 4, 8]])
    assert fake_cuda.copies == [1, 2, 1, 2]
    assert not fake_cuda.pending
    del engine
    assert not fake_cuda.allocations


def test_engines_share_cache_on_device_and_retain_scratch(tmp_path, fake_cuda, fake_engine):
    stream = backend.CudaStream()
    prefix = backend.TensorRTEngine(tmp_path / "prefix.engine", ["x"], ["y"], False, stream)
    step = backend.TensorRTEngine(tmp_path / "step.engine", ["x"], ["y"], False, stream)
    scratch = backend.CudaBuffer((64,), np.uint8)
    prefix.use_scratch(scratch)
    step.use_scratch(scratch)
    pointer = scratch.ptr
    del scratch
    cache = prefix.run_device(np.ones((1, 3), np.float32))
    np.testing.assert_array_equal(step(*cache), [[4, 4, 4]])
    assert fake_engine[1].addresses["x"] == cache[0].ptr
    assert fake_cuda.copies == [1, 2]
    assert fake_engine[0].scratch == fake_engine[1].scratch == (pointer, 64)
    assert pointer in fake_cuda.allocations


def test_engine_synchronizes_copies_after_execution_failure(tmp_path, fake_cuda, fake_engine):
    engine = backend.TensorRTEngine(tmp_path / "model.engine", ["x"], ["y"])
    fake_engine[0].execute_async_v3 = lambda stream: False
    with pytest.raises(RuntimeError, match="could not run"):
        engine(np.ones((1, 3), np.float32))
    assert not fake_cuda.pending


def test_allocator_uses_plain_malloc_and_handles_failure(monkeypatch):
    cuda = SimpleNamespace(
        cudaMalloc=MagicMock(return_value=(0, 4096)), cudaFree=MagicMock(return_value=(0,)),
        cudaStreamSynchronize=MagicMock(return_value=(0,)),
    )
    monkeypatch.setattr(backend, "cuda_runtime", lambda: cuda)
    trt = SimpleNamespace(IGpuAllocator=type("Allocator", (), {}), Runtime=lambda logger: SimpleNamespace(), Logger=MagicMock())
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setattr(backend, "runtime", functools.cache(backend.runtime.__wrapped__))
    allocator = backend.runtime().gpu_allocator
    assert allocator.allocate_async(512, 256, 0, 123) == 4096
    cuda.cudaMalloc.assert_called_once_with(512)
    assert allocator.deallocate_async(4096, 123)
    cuda.cudaFree.assert_called_once_with(4096)
    assert cuda.cudaStreamSynchronize.call_count == 2
    cuda.cudaMalloc.return_value = (2, 0)
    assert allocator.allocate(512, 256, 0) == 0


def test_bfloat16_bindings_use_extension_dtype(monkeypatch):
    ml_dtypes = pytest.importorskip("ml_dtypes")
    trt = SimpleNamespace(bfloat16="bfloat16", nptype=MagicMock(side_effect=TypeError))
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    dtype = backend.numpy_dtype(trt.bfloat16)
    assert dtype == np.dtype(ml_dtypes.bfloat16)
    assert dtype.itemsize == 2
    values = np.asarray([1.25, -2.5], dtype=dtype)
    np.testing.assert_array_equal(values.astype(np.float32), [1.25, -2.5])
    trt.nptype.assert_not_called()


def test_cuda_error_is_not_ignored():
    with pytest.raises(RuntimeError, match="CUDA runtime call failed"):
        backend.check_cuda((2, 0))


@pytest.mark.parametrize("noise_shape", [(1, 4, 3), (4, 3)])
def test_split_loop_passes_one_timestep_for_both_contracts(tmp_path, monkeypatch, noise_shape):
    times = []
    cache = object()

    class Engine:
        scratch_bytes = 64
        input_shape = (3,)

        def __init__(self, path, *args, **kwargs):
            self.name = path.stem

        def use_scratch(self, scratch):
            pass

        def run_device(self, state):
            return (cache,)

        def __call__(self, *inputs):
            if self.name == "actions":
                return inputs[0]
            saved, sample, timestep = inputs
            assert saved is cache
            assert timestep.shape == (1,) and timestep.dtype == np.float32
            times.append(timestep.item())
            return sample * timestep

    monkeypatch.setattr(backend, "TensorRTEngine", Engine)
    monkeypatch.setattr(backend, "CudaStream", lambda: None)
    monkeypatch.setattr(backend, "CudaBuffer", lambda *a: None)
    programs = {name: {"file": f"{name}.engine", "inputs": [], "outputs": []}
                for name in ("prefix", "step", "actions")}
    loop = backend.TensorRTDenoisingLoop(tmp_path, programs, 4)
    result = loop(np.ones(3), np.ones(noise_shape, dtype=np.float32))
    assert result.shape == noise_shape
    assert times == [1.0, 0.75, 0.5, 0.25]
    np.testing.assert_allclose(result, (1 - .25) * (1 - .1875) * (1 - .125) * (1 - .0625))
