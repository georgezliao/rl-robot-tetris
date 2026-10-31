"""Evaluate every checkpoint of one training run to get a learning curve.

All checkpoints are evaluated the same way as ``evaluate.py`` (deterministic
actions, held-out seeds from the config). The curve is for reading training
progress. The declared headline result is still ``final.pt``, so do not pick a
checkpoint from this curve and report it as the result.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tetris_rl.envs.javafx import JavaFxControlEnv
from tetris_rl.training.config import load_visual_config
from tetris_rl.training.evaluate import (
    EvaluationResult,
    evaluate_loaded_policy,
    load_visual_checkpoint,
)
from tetris_rl.training.io_utils import write_json
from tetris_rl.training.provenance import collect_provenance
from tetris_rl.training.results import results_directory_name

CURVE_COLUMNS = (
    "checkpoint",
    "environment_steps",
    "episode_count",
    "terminated_episodes",
    "truncated_episodes",
    "mean_survival_pieces",
    "lines_per_100_pieces",
    "mean_lines",
    "evaluation_run",
)


def list_checkpoints(run_directory: Path) -> list[Path]:
    """``step-N.pt`` files in step order, then ``final.pt`` if present."""
    checkpoint_directory = run_directory / "checkpoints"
    steps = sorted(
        checkpoint_directory.glob("step-*.pt"),
        key=lambda path: int(re.fullmatch(r"step-(\d+)\.pt", path.name).group(1)),
    )
    final = checkpoint_directory / "final.pt"
    return [*steps, final] if final.exists() else steps


def curve_rows(
    checkpoints: Sequence[Path],
    evaluate_checkpoint: Callable[[Path], EvaluationResult],
) -> list[dict[str, Any]]:
    """Evaluate each checkpoint and return one learning-curve row per checkpoint."""
    rows = []
    for checkpoint in checkpoints:
        result = evaluate_checkpoint(checkpoint)
        summary = result.summary
        rows.append(
            {
                "checkpoint": checkpoint.name,
                "environment_steps": summary["checkpoint_environment_steps"],
                "episode_count": summary["episode_count"],
                "terminated_episodes": summary["terminated_episodes"],
                "truncated_episodes": summary["truncated_episodes"],
                "mean_survival_pieces": summary["mean_survival_pieces"],
                "lines_per_100_pieces": summary["lines_per_100_pieces"],
                "mean_lines": summary["mean_lines"],
                "evaluation_run": result.run_directory.name,
            }
        )
    return rows


def write_learning_curve(
    rows: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
    results_root: Path,
    training_run_name: str,
    *,
    date: str | None = None,
) -> Path:
    """Write ``learning_curve.csv`` and ``provenance.json`` to a new results dir."""
    date = date or datetime.now(UTC).strftime("%Y-%m-%d")
    device = str(provenance.get("device", {}).get("description", "unknown"))
    directory = results_root / results_directory_name(
        date, device, f"curve-{training_run_name}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "learning_curve.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=CURVE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: "" if row[column] is None else row[column]
                    for column in CURVE_COLUMNS
                }
            )
    write_json(directory / "provenance.json", provenance)
    return directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--episodes", type=int, help="bounded diagnostic override")
    parser.add_argument(
        "--max-steps-per-episode", type=int, help="bounded diagnostic override"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_visual_config(args.config)
    episode_count = (
        config.evaluation.episodes if args.episodes is None else args.episodes
    )
    max_steps = (
        config.evaluation.max_steps_per_episode
        if args.max_steps_per_episode is None
        else args.max_steps_per_episode
    )
    checkpoints = list_checkpoints(args.run_directory)
    if not checkpoints:
        raise SystemExit(f"no checkpoints under {args.run_directory}/checkpoints")

    with JavaFxControlEnv(
        observation_size=(config.observation.height, config.observation.width),
        frame_stack=config.observation.frame_stack,
    ) as environment:

        def evaluate_checkpoint(path: Path) -> EvaluationResult:
            return evaluate_loaded_policy(
                config,
                load_visual_checkpoint(path, config),
                environment,
                output_root=config.evaluation.output_directory,
                episode_count=episode_count,
                max_steps_per_episode=max_steps,
            )

        rows = curve_rows(checkpoints, evaluate_checkpoint)

    first = load_visual_checkpoint(checkpoints[0], config)
    provenance = collect_provenance(
        config_path=args.config,
        seeds={
            "evaluation_seeds": [
                config.evaluation.first_seed,
                config.evaluation.first_seed + episode_count - 1,
            ]
        },
        device=next(first.model.parameters()).device,
        extra={
            "training_run": str(args.run_directory),
            "checkpoints": [path.name for path in checkpoints],
            "max_steps_per_episode": max_steps,
        },
    )
    directory = write_learning_curve(
        rows, provenance, args.results_root, args.run_directory.name
    )
    print(json.dumps({"results_directory": str(directory), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
