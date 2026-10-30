"""Small helpers shared by the training and evaluation entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


def observation_to_tensor(observation: np.ndarray, device: torch.device) -> Tensor:
    """Return one environment observation as a batch of size one on ``device``."""
    return torch.from_numpy(np.ascontiguousarray(observation)).unsqueeze(0).to(device)


def write_json(path: Path, value: object) -> None:
    """Write indented, key-sorted JSON. NaN and infinity raise ``ValueError``."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    """Append one key-sorted JSON line.

    NaN is allowed here on purpose: a diverging PPO update should still be
    logged to ``metrics.jsonl`` rather than crash the run.
    """
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
