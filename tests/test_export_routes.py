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

"""The route scripts: each compiles a part with its compiler, the same way for every policy."""

import contextlib
import importlib.util
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("datasets", reason="the export examples require lerobot[dataset]")
pytest.importorskip("av", reason="the export examples require lerobot[dataset]")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "export"


@pytest.fixture
def route(monkeypatch):
    """Load a route script with its compiler stubbed out, and return its `compile_part` and the stubs."""
    monkeypatch.syspath_prepend(str(EXAMPLES))

    def load(name):
        stubs = {
            "torch_tensorrt": MagicMock(),
            "executorch.exir": SimpleNamespace(ExecutorchBackendConfig=MagicMock()),
            "executorch.exir.passes.memory_planning_pass": SimpleNamespace(MemoryPlanningPass=MagicMock()),
            "executorch.exir.passes.propagate_device_pass": SimpleNamespace(
                PropagateDeviceConfig=MagicMock()
            ),
        }
        for module, stub in stubs.items():
            monkeypatch.setitem(sys.modules, module, stub)
        import build_engine

        monkeypatch.setattr(build_engine, "build_engine", MagicMock())
        stubs["build_engine"] = build_engine
        monkeypatch.setattr(torch.export, "export", MagicMock())
        monkeypatch.setattr(torch.onnx, "export", MagicMock())
        return runpy.run_path(str(EXAMPLES / f"{name}.py"))["compile_part"], stubs

    yield load
    sys.modules.pop("common", None)
    sys.modules.pop("build_engine", None)


def part():
    from common import Part

    return Part("step", torch.nn.Linear(2, 2), (torch.zeros(2),), ["x_t"], ["x_next"])


def test_executorch_route_keeps_inputs_and_outputs_on_the_gpu_and_records_cuda_graphs(route, tmp_path):
    compile_part, stubs = route("to_executorch_tensorrt")
    compile_part(part(), tmp_path / "step.pte")
    tensorrt = stubs["torch_tensorrt"]
    kwargs = tensorrt.save.call_args.kwargs
    assert tensorrt.save.call_args.args[1] == str(tmp_path / "step.pte")
    assert kwargs["output_format"] == "executorch" and kwargs["use_cuda_graphs"] is True
    stubs["executorch.exir.passes.propagate_device_pass"].PropagateDeviceConfig.assert_called_once_with(
        skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
    )
    stubs["executorch.exir.passes.memory_planning_pass"].MemoryPlanningPass.assert_called_once_with(
        alloc_graph_input=False, alloc_graph_output=False
    )


@pytest.fixture
def modelopt(monkeypatch):
    """ModelOpt stubbed out: its quantize enables one weight quantizer, its export mode records when it is on."""
    events = []

    class TensorQuantizer(torch.nn.Module):
        is_enabled = True

    def quantize(module, config, forward_loop):
        events.append(
            ("quantize", config, module.training, os.environ.get("MODELOPT_SKIP_WEIGHT_QUANT_CHECK"))
        )
        forward_loop(module)
        module.weight_quantizer = TensorQuantizer()

    @contextlib.contextmanager
    def export_torch_mode():
        events.append("enter")
        yield
        events.append("exit")

    mtq = ModuleType("modelopt.torch.quantization")
    mtq.quantize, mtq.nn = quantize, SimpleNamespace(TensorQuantizer=TensorQuantizer)
    for name in ("FP8", "INT8"):
        setattr(mtq, f"{name}_DEFAULT_CFG", {"quant_cfg": [{"quantizer_name": name}]})
    utils = ModuleType("modelopt.torch.quantization.utils")
    utils.export_torch_mode = export_torch_mode
    for name, module in {
        "modelopt": ModuleType("modelopt"),
        "modelopt.torch": ModuleType("modelopt.torch"),
        "modelopt.torch.quantization": mtq,
        "modelopt.torch.quantization.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    # Set, so that the test's teardown puts back whatever the environment had.
    monkeypatch.setenv("MODELOPT_SKIP_WEIGHT_QUANT_CHECK", "0")
    return events


@pytest.mark.parametrize("precision", ["fp8", "int8"])
def test_executorch_route_quantizes_calibrated_on_the_part_inputs_and_compiles_the_quantized_graph(
    route, tmp_path, modelopt, precision
):
    compile_part, stubs = route("to_executorch_tensorrt")
    calibrated = []
    quantized_part = part()
    quantized_part.module.register_forward_hook(lambda module, inputs, output: calibrated.append(inputs))
    torch.export.export.side_effect = lambda *args: modelopt.append("export")
    stubs["torch_tensorrt"].dynamo.compile.side_effect = lambda *args, **kwargs: modelopt.append("compile")
    compile_part(quantized_part, tmp_path / "step.pte", quantize=precision)
    # Model Optimizer's config for the precision, which also keeps GR00T's vision encoder and the inputs of
    # LayerNorm layers, calibrated in eval mode, which it then keeps for every layer, with its
    # nothing-to-quantize check off.
    keep_vision = {"quantizer_name": "*vision*", "enable": False}
    keep_layer_norm = {"quantizer_name": "*input_quantizer", "parent_class": "nn.LayerNorm", "enable": False}
    config = {"quant_cfg": [{"quantizer_name": precision.upper()}, keep_vision, keep_layer_norm]}
    assert modelopt == [("quantize", config, False, "1"), "enter", "export", "compile", "exit"]
    assert len(calibrated) == 1 and calibrated[0][0] is quantized_part.inputs[0]
    # The config was copied: the module-level preset is unchanged.
    assert sys.modules["modelopt.torch.quantization"].FP8_DEFAULT_CFG["quant_cfg"] == [
        {"quantizer_name": "FP8"}
    ]


def test_executorch_route_compiles_each_part_with_the_quantize_argument(route, monkeypatch):
    route("to_executorch_tensorrt")
    import common

    passed = {}

    def parse_args(description, route, parents=()):
        passed["options"] = {option for parser in parents for option in parser._option_string_actions}
        return SimpleNamespace(quantize="int8"), None

    monkeypatch.setattr(common, "parse_args", parse_args)
    monkeypatch.setattr(common, "run_route", lambda *args: passed.update(compile_part=args[3]))
    runpy.run_path(str(EXAMPLES / "to_executorch_tensorrt.py"), run_name="__main__")
    assert "--quantize" in passed["options"]
    assert passed["compile_part"].keywords == {"quantize": "int8"}


@pytest.mark.parametrize(
    "capability, runs",
    [
        ((8, 7), {"int8"}),
        ((8, 9), {"int8", "fp8"}),
        ((9, 0), {"int8", "fp8"}),
        ((10, 0), {"int8", "fp8", "nvfp4"}),
    ],
    ids=["orin", "ada", "hopper", "blackwell"],
)
def test_quantize_refuses_a_precision_this_gpu_cannot_run_before_the_policy_loads(
    monkeypatch, capability, runs
):
    """TensorRT would otherwise fail only at the end of the first part's build."""
    monkeypatch.syspath_prepend(str(EXAMPLES))
    from common import QUANTIZATIONS, quantize_arguments

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    try:
        for precision in QUANTIZATIONS:
            if precision in runs:
                assert quantize_arguments().parse_args([f"--quantize={precision}"]).quantize == precision
            else:
                with pytest.raises(SystemExit):
                    quantize_arguments().parse_args([f"--quantize={precision}"])
    finally:
        sys.modules.pop("common", None)


@pytest.mark.parametrize("missing", ["precision", "modelopt", "gpu"])
def test_quantize_refuses_what_this_machine_cannot_do(monkeypatch, capsys, missing):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    from common import quantize_arguments

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if missing == "modelopt" else object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: missing != "gpu")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (11, 0))
    try:
        with pytest.raises(SystemExit):
            quantize_arguments().parse_args(
                ["--quantize=int4" if missing == "precision" else "--quantize=int8"]
            )
    finally:
        sys.modules.pop("common", None)
    expected = {"precision": "choose from", "modelopt": "pip install 'nvidia-modelopt", "gpu": "NVIDIA GPU"}
    assert expected[missing] in capsys.readouterr().err


def test_onnx_route_keeps_the_onnx_file_and_builds_the_engine_beside_it(route, tmp_path):
    compile_part, stubs = route("to_onnx_tensorrt")
    compile_part(part(), tmp_path / "step.engine")
    onnx_path = torch.onnx.export.call_args.args[2]
    assert onnx_path == tmp_path / "step.onnx"
    assert torch.onnx.export.call_args.kwargs["input_names"] == ["x_t"]
    assert torch.onnx.export.call_args.kwargs["output_names"] == ["x_next"]
    build = stubs["build_engine"].build_engine
    assert build.call_args.args[:2] == (tmp_path / "step.onnx", tmp_path / "step.engine")


def test_torch_tensorrt_route_saves_an_aoti_package(route, tmp_path):
    compile_part, stubs = route("to_torch_tensorrt")
    compile_part(part(), tmp_path / "step.pt2")
    tensorrt = stubs["torch_tensorrt"]
    assert tensorrt.save.call_args.args[1] == str(tmp_path / "step.pt2")
    assert tensorrt.save.call_args.kwargs["output_format"] == "aot_inductor"


@pytest.mark.parametrize("name", ["to_executorch_tensorrt", "to_onnx_tensorrt", "to_torch_tensorrt"])
def test_routes_compile_with_the_shared_tensorrt_options(route, tmp_path, name):
    compile_part, stubs = route(name)
    from common import TENSORRT_OPTIONS

    compile_part(part(), tmp_path / "step.bin")
    if name == "to_onnx_tensorrt":
        assert TENSORRT_OPTIONS["workspace_size"] == stubs["build_engine"].WORKSPACE_BYTES
    else:
        options = stubs["torch_tensorrt"].dynamo.compile.call_args.kwargs
        assert {key: options[key] for key in TENSORRT_OPTIONS} == TENSORRT_OPTIONS


@pytest.mark.parametrize("name", ["to_executorch_tensorrt", "to_torch_tensorrt"])
def test_a_part_overrides_the_shared_tensorrt_options(route, tmp_path, name):
    compile_part, stubs = route(name)
    from common import TENSORRT_OPTIONS

    overriding = part()
    overriding.tensorrt_options = {"decompose_attention": not TENSORRT_OPTIONS["decompose_attention"]}
    compile_part(overriding, tmp_path / "step.bin")
    options = stubs["torch_tensorrt"].dynamo.compile.call_args.kwargs
    assert options == {"arg_inputs": overriding.inputs, **TENSORRT_OPTIONS, **overriding.tensorrt_options}


@pytest.mark.parametrize("checkpoint", [False, True], ids=["alone", "with_checkpoint"])
def test_help_lists_the_route_and_recipe_arguments(tmp_path, monkeypatch, capsys, checkpoint):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    from common import parse_args

    argv = ["to_executorch_tensorrt.py", "--help"]
    if checkpoint:
        (tmp_path / "config.json").write_text('{"type": "smolvla"}')
        argv.insert(1, f"--policy.path={tmp_path}")
    monkeypatch.setattr(sys, "argv", argv)
    try:
        with pytest.raises(SystemExit) as exited:
            parse_args("Export a policy.", "executorch_tensorrt")
    finally:
        for name in ("common", "smolvla"):
            sys.modules.pop(name, None)
    assert exited.value.code == 0
    text = capsys.readouterr().out
    assert "--policy.path" in text and "--output_dir" in text
    if checkpoint:
        assert "--tolerance" in text and "--dataset.repo_id" in text and "--task" in text


@pytest.mark.parametrize("parts", [["model"], ["embed", "step", "actions"]], ids=["one_part", "chain"])
def test_export_builds_each_part_in_its_own_process_then_checks_the_folder(tmp_path, monkeypatch, parts):
    """Every route fails at export time, not first on the robot, when its programs miss the test case."""
    monkeypatch.syspath_prepend(str(EXAMPLES))
    import common

    sys.modules.pop("common", None)
    events = []
    monkeypatch.setattr(common.subprocess, "run", lambda command, check: events.append(command[-1]))
    monkeypatch.setattr(common, "check_folder", lambda folder: events.append(folder))
    recipe = SimpleNamespace(__name__="recipe", part_names=lambda args: parts)
    args = SimpleNamespace(part=None, output_dir=tmp_path / "folder", route="executorch_tensorrt")
    common.run_route("script.py", args, recipe, None, ".pte")
    assert events == [*(f"--part={name}" for name in parts), tmp_path / "folder"]


def test_export_refuses_a_part_the_recipe_does_not_build(tmp_path, monkeypatch):
    """A recipe whose part names and parts disagree would otherwise write a folder with nothing compiled."""
    monkeypatch.syspath_prepend(str(EXAMPLES))
    import common

    sys.modules.pop("common", None)

    class Recipe:
        __name__ = "recipe"

        @staticmethod
        def part_names(args):
            return ["a", "b"]

        class Exporter:
            def __init__(self, args):
                pass

            def parts(self):
                for name in ("a", "b_misnamed"):
                    yield common.Part(name, torch.nn.Identity(), (torch.zeros(1),), ["x"], ["y"])

            def write(self, programs):
                raise AssertionError("nothing was compiled")

    compiled = []
    args = SimpleNamespace(part="b", output_dir=tmp_path, route="onnx_tensorrt")
    with pytest.raises(SystemExit, match="'b'"):
        common.run_route("unused", args, Recipe, lambda part, path: compiled.append(part.name), ".engine")
    assert not compiled


def test_usage_examples_print_one_option_per_line():
    """--help prints each script's docstring, where a lone backslash would join the lines of its example."""
    import ast
    import re

    for script in [*EXAMPLES.glob("*.py"), EXAMPLES / "replay_robot" / "write_frames.py"]:
        docstring = ast.get_docstring(ast.parse(script.read_text()), clean=False) or ""
        assert not re.search(r"\S {2,}--", docstring), script.name


def test_no_export_script_shadows_a_package_it_imports():
    """A script named like a package it imports hides that package, because its own folder is first on
    sys.path when it runs, as a script named torch_tensorrt.py importing torch_tensorrt would. The check is
    structural, so it catches the clash even where the package is not installed."""
    import ast

    scripts = sorted(EXAMPLES.glob("*.py"))
    filenames = {s.name for s in scripts}
    # The modules a sibling is meant to import locally: the shared driver, the engine rebuilder, the recipes,
    # and the benchmark, whose output lines the memory benchmark reads.
    local = {"common", "build_engine", "act", "smolvla", "pi05", "groot", "benchmark"}
    imported = set()
    for script in scripts:
        for node in ast.walk(ast.parse(script.read_text())):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
    shadowed = sorted(name for name in imported - local if f"{name}.py" in filenames)
    assert not shadowed, f"these scripts are named like packages they import and shadow them: {shadowed}"
