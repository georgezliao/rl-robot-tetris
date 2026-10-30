"""Non-learning reference policies evaluated like a trained checkpoint.

These give the floor a learned visual policy has to beat. They use the same
environment, held-out seeds, step limit and summary metrics as
``evaluate.py``, and the same ``results/`` export.

- ``random``: uniform over the configured actions. The action generator for an
  episode is seeded with that episode's piece seed, so a rerun repeats it.
- ``noop``: never presses a key; pieces fall one row per step under gravity.
- ``hard_drop``: presses space every step, so each piece drops where it spawns.

None of them look at the observation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from tetris_rl.envs.javafx import JavaFxControlEnv
from tetris_rl.training.config import VisualExperimentConfig, load_visual_config
from tetris_rl.training.evaluate import (
    EpisodeResult,
    EvaluationEnvironment,
    EvaluationResult,
    aggregate_episodes,
    integer_metric,
)
from tetris_rl.training.io_utils import write_json
from tetris_rl.training.provenance import collect_provenance
from tetris_rl.training.results import export_results
from tetris_rl.training.visual import action_set, validate_runtime_support
from tetris_rl.types import ControlAction

# A policy is built once per episode from that episode's seed.
PolicyFactory = Callable[[tuple[ControlAction, ...], int], Callable[[], ControlAction]]


def _random(
    actions: tuple[ControlAction, ...], seed: int
) -> Callable[[], ControlAction]:
    generator = random.Random(seed)
    return lambda: generator.choice(actions)


def _constant(action: ControlAction) -> PolicyFactory:
    def factory(
        actions: tuple[ControlAction, ...], seed: int
    ) -> Callable[[], ControlAction]:
        del seed
        if action not in actions:
            raise ValueError(f"{action.name} is not in the configured action set")
        return lambda: action

    return factory


BASELINES: dict[str, PolicyFactory] = {
    "random": _random,
    "noop": _constant(ControlAction.NOOP),
    "hard_drop": _constant(ControlAction.HARD_DROP),
}


def evaluate_baseline(
    config: VisualExperimentConfig,
    name: str,
    environment: EvaluationEnvironment,
    *,
    output_root: Path,
    episode_count: int,
    max_steps_per_episode: int,
) -> EvaluationResult:
    """Play ``episode_count`` episodes with baseline ``name`` on held-out seeds.

    Writes ``episodes.json`` and ``summary.json`` to
    ``<output_root>/baseline-<name>-<time>/``. Per-step records are not written.
    """
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; choose from {sorted(BASELINES)}")
    if episode_count < 1 or max_steps_per_episode < 1:
        raise ValueError("episode_count and max_steps_per_episode must be positive")
    actions = action_set(config)
    seeds = tuple(
        range(
            config.evaluation.first_seed, config.evaluation.first_seed + episode_count
        )
    )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = output_root / f"baseline-{name}-{run_id}"
    run_directory.mkdir(parents=True)

    episodes: list[EpisodeResult] = []
    for episode_index, seed in enumerate(seeds):
        choose_action = BASELINES[name](actions, seed)
        environment.reset(seed=seed)
        episode_return = 0.0
        done = False
        info: dict[str, Any] = {}
        for step_index in range(max_steps_per_episode):
            _, raw_reward, done, info = environment.step(choose_action())
            episode_return += raw_reward + (
                config.reward.terminal_penalty if done else 0.0
            )
            if done:
                break
        episodes.append(
            EpisodeResult(
                episode_index=episode_index,
                episode_id=f"baseline-{name}-{seed}",
                seed=seed,
                steps=step_index + 1,
                terminated=done,
                truncated=not done,
                episode_return=episode_return,
                score=integer_metric(info, "score"),
                lines=integer_metric(info, "total_lines"),
                pieces=integer_metric(info, "pieces_placed"),
                recording_directory="",
            )
        )

    summary = {
        "policy": f"baseline:{name}",
        "action_labels": [action.name.lower() for action in actions],
        **aggregate_episodes(
            episodes, seeds=seeds, max_steps_per_episode=max_steps_per_episode
        ),
        "deterministic_action_selection": name != "random",
    }
    write_json(run_directory / "episodes.json", [asdict(item) for item in episodes])
    write_json(run_directory / "summary.json", summary)
    return EvaluationResult(run_directory, tuple(episodes), summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", choices=[*sorted(BASELINES), "all"], default="all")
    parser.add_argument("--episodes", type=int, help="bounded diagnostic override")
    parser.add_argument(
        "--max-steps-per-episode", type=int, help="bounded diagnostic override"
    )
    parser.add_argument("--output-root", type=Path, help="default: evaluation dir")
    parser.add_argument(
        "--results-root",
        type=Path,
        help="also export summary, episodes.csv and provenance (normally results/)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_visual_config(args.config)
    validate_runtime_support(config)
    episode_count = (
        config.evaluation.episodes if args.episodes is None else args.episodes
    )
    max_steps = (
        config.evaluation.max_steps_per_episode
        if args.max_steps_per_episode is None
        else args.max_steps_per_episode
    )
    output_root = args.output_root or config.evaluation.output_directory
    names = sorted(BASELINES) if args.policy == "all" else [args.policy]

    printed: list[dict[str, Any]] = []
    with JavaFxControlEnv(
        observation_size=(config.observation.height, config.observation.width),
        frame_stack=config.observation.frame_stack,
    ) as environment:
        for name in names:
            provenance = collect_provenance(
                config_path=args.config,
                seeds={
                    "evaluation_seeds": [
                        config.evaluation.first_seed,
                        config.evaluation.first_seed + episode_count - 1,
                    ]
                },
                device=torch.device("cpu"),
                extra={"policy": f"baseline:{name}"},
            )
            result = evaluate_baseline(
                config,
                name,
                environment,
                output_root=output_root,
                episode_count=episode_count,
                max_steps_per_episode=max_steps,
            )
            write_json(result.run_directory / "provenance.json", provenance)
            entry: dict[str, Any] = {"run_directory": str(result.run_directory)}
            if args.results_root is not None:
                entry["results_directory"] = str(
                    export_results(result, provenance, args.results_root)
                )
            printed.append({**entry, **result.summary, "seeds": "omitted"})
    print(json.dumps(printed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
