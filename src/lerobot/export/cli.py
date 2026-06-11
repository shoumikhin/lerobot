# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""lerobot-export CLI.

Usage
-----
# AOTI backend (works on the default lerobot install, torch 2.11+):
lerobot-export --policy.path=lerobot/smolvla_base --backend=aoti --output_dir=bundles/smolvla_aoti

# torch-trt / ExecuTorch backends require a separate export venv (torch 2.12 + torch-tensorrt 2.12).
# See src/lerobot/export/requirements-export-trt.txt for setup instructions.
lerobot-export --policy.path=<ckpt> --backend=torch-trt --output_dir=bundles/smolvla_trt
lerobot-export --policy.path=<ckpt> --backend=torch-trt-et --output_dir=bundles/smolvla_pte
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import draccus

from lerobot.export.adapters.registry import get_adapter
from lerobot.export.backends.aoti import run_aoti_export
from lerobot.export.backends.torch_trt import run_torch_trt_export
from lerobot.export.backends.torch_trt_et import run_torch_trt_et_export

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass (draccus-parseable, matches lerobot's --foo.bar convention)
# ---------------------------------------------------------------------------


@dataclass
class PolicyExportRef:
    """Minimal reference to a pretrained policy checkpoint."""

    path: str = ""
    """HuggingFace Hub repo ID or local directory, e.g. lerobot/smolvla_base"""

    device: str = "cuda"
    """Device to load the policy on for shape-tracing (cuda or cpu)."""


@dataclass
class ExportConfig:
    policy: PolicyExportRef = field(default_factory=PolicyExportRef)
    backend: Literal["aoti", "torch-trt", "torch-trt-et"] = "aoti"
    """
    Export backend to use:
      aoti       -- vanilla AOTInductor (.pt2) + libtorch C++ [works on default install]
      torch-trt  -- torch-tensorrt AOTI (.pt2) + TRT engines  [needs export venv]
      torch-trt-et -- torch-tensorrt -> ExecuTorch (.pte)     [needs export venv]
    """
    output_dir: Path = Path("export_bundle")
    """Directory where the bundle (artifacts + manifest.json + runner/) will be written."""

    batch_size: int = 1
    """Batch size for exported graphs (static shapes -- must match the C++ runner)."""

    emit_cpp_runner: bool = True
    """Write the reference C++ runner + CMakeLists.txt into output_dir/runner/."""

    overwrite: bool = False
    """If the output_dir already exists, overwrite without prompting."""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@draccus.wrap()
def export_main(cfg: ExportConfig) -> None:
    """Core export logic dispatched per backend."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not cfg.policy.path:
        raise ValueError(
            "No policy specified. Pass --policy.path=<hub_id_or_local_dir>, "
            "e.g. --policy.path=lerobot/smolvla_base"
        )

    output_dir = Path(cfg.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not cfg.overwrite:
        raise FileExistsError(
            f"Output directory {output_dir} already exists and is not empty. "
            "Pass --overwrite=true to overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading policy from %r ...", cfg.policy.path)
    from lerobot.policies.pretrained import PreTrainedPolicy

    # Resolve policy class via the existing lerobot registry.
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import get_policy_class

    policy_config = PreTrainedConfig.from_pretrained(cfg.policy.path)
    policy_cls = get_policy_class(policy_config.type)
    policy: PreTrainedPolicy = policy_cls.from_pretrained(cfg.policy.path)
    policy = policy.to(cfg.policy.device)
    policy.eval()
    logger.info("Policy loaded: type=%s, device=%s", policy_config.type, cfg.policy.device)

    # Look up the per-policy export adapter.
    adapter = get_adapter(policy_config.type, policy, cfg)

    logger.info("Backend: %s", cfg.backend)
    if cfg.backend == "aoti":
        run_aoti_export(adapter, cfg, output_dir)
    elif cfg.backend == "torch-trt":
        run_torch_trt_export(adapter, cfg, output_dir)
    elif cfg.backend == "torch-trt-et":
        run_torch_trt_et_export(adapter, cfg, output_dir)
    else:
        raise ValueError(f"Unknown backend: {cfg.backend!r}")

    if cfg.emit_cpp_runner:
        _emit_cpp_runner(output_dir)

    logger.info("Export complete. Bundle written to %s", output_dir)


def _emit_cpp_runner(output_dir: Path) -> None:
    """Copy the reference C++ runner into output_dir/runner/."""
    import shutil

    src = Path(__file__).parent / "runner"
    dst = output_dir / "runner"
    if src.exists():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        logger.info("C++ runner written to %s", dst)


def main() -> None:
    export_main()


if __name__ == "__main__":
    main()
