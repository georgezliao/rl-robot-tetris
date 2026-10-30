"""Record where a training or evaluation run came from.

Every run directory gets a ``provenance.json`` so a number in a results table
can be traced back to the code, game, configuration, seeds and machine that
produced it.
"""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROVENANCE_SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _run(command: list[str], cwd: Path) -> str | None:
    """Return stripped stdout, or None if the command is missing or fails."""
    try:
        completed = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def git_state(directory: Path) -> dict[str, Any]:
    """Commit hash and whether tracked or untracked files differ from it.

    Changes inside the nested ``tetris/`` repository are excluded here
    (``--ignore-submodules=all``) because that repository is recorded separately.
    Both fields are None when ``directory`` is not a Git checkout.
    """
    commit = _run(["git", "rev-parse", "HEAD"], directory)
    if commit is None:
        return {"commit": None, "dirty": None}
    status = _run(
        ["git", "status", "--porcelain", "--ignore-submodules=all"], directory
    )
    return {"commit": commit, "dirty": None if status is None else status != ""}


def java_version(repository_root: Path = REPOSITORY_ROOT) -> str | None:
    """First line of ``java -version`` for the JDK the bridge scripts select."""
    java_home = _run(
        [str(repository_root / "scripts" / "java_home.sh")], repository_root
    )
    if not java_home:
        return None
    try:
        completed = subprocess.run(
            [str(Path(java_home) / "bin" / "java"), "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # java -version writes to stderr.
    lines = (completed.stderr or completed.stdout).strip().splitlines()
    return lines[0] if lines else None


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"], Path.cwd())
        if brand:
            return brand
    return platform.processor() or platform.machine()


def device_description(device: torch.device) -> str:
    """Human-readable device name: the CUDA GPU model, or the CPU model."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"MPS on {_cpu_name()}"
    return _cpu_name()


def collect_provenance(
    *,
    config_path: Path | None,
    seeds: Mapping[str, Any],
    device: torch.device,
    extra: Mapping[str, Any] | None = None,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Return the provenance record for one run.

    ``seeds`` and ``extra`` are copied in as given, so callers decide which
    seed ranges and run-specific facts (such as a checkpoint hash) matter.
    """
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "repository": git_state(repository_root),
        "tetris": git_state(repository_root / "tetris"),
        "config_path": None if config_path is None else str(config_path),
        "seeds": dict(seeds),
        "device": {"type": device.type, "description": device_description(device)},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "java": java_version(repository_root),
        },
        **({} if extra is None else {"extra": dict(extra)}),
    }
