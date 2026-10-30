"""Copy the small, publishable part of an evaluation run into ``results/``.

``experiments/evaluation/<run>/`` holds everything an evaluation produces,
including per-step JSONL records, and is ignored by Git. ``results/`` holds only
what a results table needs: the aggregate summary, one row per episode, and the
provenance record. Those directories are committed, and README numbers should
come only from them.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tetris_rl.training.io_utils import write_json

if TYPE_CHECKING:
    from tetris_rl.training.evaluate import EvaluationResult

# recording_directory is a local path into experiments/, so it is left out.
EPISODE_COLUMNS = (
    "episode_index",
    "seed",
    "steps",
    "terminated",
    "truncated",
    "episode_return",
    "score",
    "lines",
    "pieces",
    "episode_id",
)


def slug(text: str) -> str:
    """Lower-case ``text`` and replace runs of other characters with ``-``."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "unknown"


def results_directory_name(date: str, device: str, run_name: str) -> str:
    """``<date>-<device>-<run>``, e.g. ``2026-10-30-apple-m3-visual_reference_ppo-…``."""
    return f"{date}-{slug(device)}-{run_name}"


def export_results(
    result: EvaluationResult,
    provenance: Mapping[str, Any],
    results_root: Path,
    *,
    date: str | None = None,
) -> Path:
    """Write ``summary.json``, ``episodes.csv`` and ``provenance.json``.

    The directory name ends with the evaluation run's directory name, so the
    full records in ``experiments/evaluation/`` can be found from it. Refuses to
    overwrite an existing results directory.
    """
    date = date or datetime.now(UTC).strftime("%Y-%m-%d")
    device = str(provenance.get("device", {}).get("description", "unknown"))
    directory = results_root / results_directory_name(
        date, device, result.run_directory.name
    )
    directory.mkdir(parents=True, exist_ok=False)

    write_json(directory / "summary.json", result.summary)
    write_json(directory / "provenance.json", provenance)
    with (directory / "episodes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EPISODE_COLUMNS)
        writer.writeheader()
        for episode in result.episodes:
            row = asdict(episode)
            writer.writerow({column: row[column] for column in EPISODE_COLUMNS})
    return directory
