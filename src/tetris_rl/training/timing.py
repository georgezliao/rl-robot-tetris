"""Summaries for wall-clock timings collected by the throughput probe."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Self

import numpy as np
import torch


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work so a timer measures it, not just its launch."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class Stopwatch:
    """Collect durations, in seconds, of repeated sections of code."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.durations: list[float] = []
        self._start = 0.0

    def __enter__(self) -> Self:
        synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        synchronize(self.device)
        self.durations.append(time.perf_counter() - self._start)


def summarize_durations(durations: Sequence[float]) -> dict[str, float | int]:
    """Count, median, 90th percentile and mean, in milliseconds.

    Percentiles use NumPy's default linear interpolation.
    """
    if not durations:
        raise ValueError("no durations to summarize")
    milliseconds = np.asarray(durations, dtype=np.float64) * 1000.0
    return {
        "count": int(milliseconds.size),
        "median_ms": float(np.median(milliseconds)),
        "p90_ms": float(np.percentile(milliseconds, 90)),
        "mean_ms": float(milliseconds.mean()),
    }
