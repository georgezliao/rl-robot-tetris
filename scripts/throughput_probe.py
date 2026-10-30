"""Time the three parts of a training step on the live JavaFX bridge.

Measured separately:

- env_step: one ``JavaFxControlEnv.step`` call. That covers the key event, the
  gravity tick, the JavaFX snapshot, base64/JSON transfer, and Python-side
  decoding, resizing and frame stacking.
- env_reset: ``reset`` calls after game over, counted apart from steps.
- policy_forward: one ``VisualActorCritic.act`` call on a single observation,
  as during rollout collection.
- ppo_update: one ``update_policy`` call on a full rollout (GAE plus
  ``update_epochs`` passes of minibatch gradient steps).

The policy is untrained, so actions are close to uniform and episodes are short.
Results go to ``<results-root>/<date>-<device>-throughput-<time>/``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from tetris_rl.envs.javafx import JavaFxControlEnv
from tetris_rl.models.cnn_policy import VisualActorCritic
from tetris_rl.training.config import load_visual_config
from tetris_rl.training.io_utils import observation_to_tensor, write_json
from tetris_rl.training.provenance import collect_provenance
from tetris_rl.training.results import results_directory_name
from tetris_rl.training.timing import Stopwatch, summarize_durations
from tetris_rl.training.visual import (
    EpisodeProgress,
    action_set,
    collect_rollout,
    resolve_device,
    training_episode_seed,
    update_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=1,
        help="untimed updates first; the first MPS update includes kernel setup",
    )
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument(
        "--device", help="override training.device (cpu, mps, cuda, auto)"
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_visual_config(args.config)
    device = resolve_device(args.device or config.training.device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    actions = action_set(config)
    model = VisualActorCritic(
        config.observation.shape, len(actions), config.model.feature_dim
    ).to(device)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.algorithm.learning_rate)
    reserved = config.evaluation.seeds

    step_timer = Stopwatch(torch.device("cpu"))
    reset_timer = Stopwatch(torch.device("cpu"))
    forward_timer = Stopwatch(device)
    update_timer = Stopwatch(device)
    episodes = 0

    with JavaFxControlEnv(
        observation_size=(config.observation.height, config.observation.width),
        frame_stack=config.observation.frame_stack,
    ) as environment:
        observation = environment.reset(
            seed=training_episode_seed(config.seed, 0, reserved)
        )
        loop_start = time.perf_counter()
        for step in range(args.warmup_steps + args.steps):
            measured = step >= args.warmup_steps
            if step == args.warmup_steps:
                loop_start = time.perf_counter()
            tensor = observation_to_tensor(observation, device)
            if measured:
                with forward_timer:
                    action, _, _ = model.act(tensor)
            else:
                action, _, _ = model.act(tensor)
            index = int(action.item())
            if measured:
                with step_timer:
                    observation, _, done, _ = environment.step(actions[index])
            else:
                observation, _, done, _ = environment.step(actions[index])
            if done:
                episodes += 1
                seed = training_episode_seed(config.seed, episodes, reserved)
                with reset_timer:
                    observation = environment.reset(seed=seed)
        step_loop_seconds = time.perf_counter() - loop_start

        progress = EpisodeProgress(index=episodes)
        for update_index in range(args.warmup_updates + args.updates):
            rollout = collect_rollout(
                environment,
                model,
                actions,
                observation,
                progress,
                length=config.algorithm.rollout_steps,
                first_seed=config.seed,
                terminal_penalty=config.reward.terminal_penalty,
                steps_before=0,
                device=device,
                reserved_seeds=reserved,
            )
            observation = rollout.final_observation
            if update_index < args.warmup_updates:
                update_policy(model, optimizer, rollout, config.algorithm, device)
                continue
            with update_timer:
                update_policy(model, optimizer, rollout, config.algorithm, device)

    update = summarize_durations(update_timer.durations)
    seconds_per_step = step_loop_seconds / args.steps
    update_seconds_per_step = (
        update["median_ms"] / 1000.0 / config.algorithm.rollout_steps
    )
    report = {
        "what_was_timed": __doc__.strip().splitlines()[0],
        "config": str(args.config),
        "device": str(device),
        "measured_steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "measured_updates": args.updates,
        "warmup_updates": args.warmup_updates,
        "episodes_finished_in_step_loop": episodes,
        "rollout_steps": config.algorithm.rollout_steps,
        "update_epochs": config.algorithm.update_epochs,
        "minibatch_size": config.algorithm.minibatch_size,
        "timings": {
            "env_step": summarize_durations(step_timer.durations),
            "env_reset": (
                summarize_durations(reset_timer.durations)
                if reset_timer.durations
                else None
            ),
            "policy_forward": summarize_durations(forward_timer.durations),
            "ppo_update": update,
        },
        "step_loop_wall_clock_seconds": step_loop_seconds,
        "collection_steps_per_second": 1.0 / seconds_per_step,
        # Collection wall-clock per step plus the median update cost spread over
        # one rollout. Ignores checkpoint and logging time.
        "estimated_training_steps_per_second": 1.0
        / (seconds_per_step + update_seconds_per_step),
        "estimated_hours_per_million_steps": (
            1e6 * (seconds_per_step + update_seconds_per_step) / 3600.0
        ),
    }
    provenance = collect_provenance(
        config_path=args.config,
        seeds={"training_first_seed": config.seed},
        device=device,
    )
    run_name = "throughput-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = args.results_root / results_directory_name(
        datetime.now(UTC).strftime("%Y-%m-%d"),
        device.type
        if device.type != "cpu"
        else "cpu " + provenance["device"]["description"],
        run_name,
    )
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "throughput.json", report)
    write_json(directory / "provenance.json", provenance)
    print(json.dumps({"results_directory": str(directory), **report}, indent=2))


if __name__ == "__main__":
    main()
