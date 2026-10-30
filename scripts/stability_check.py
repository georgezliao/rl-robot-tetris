"""Exercise determinism and lifecycle properties of the visual Tetris environment."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tetris_rl.envs.javafx import JavaFxControlEnv
from tetris_rl.events.observations import observation_sha256
from tetris_rl.types import ControlAction, Observation


@dataclass(frozen=True, slots=True)
class TransitionSignature:
    action: int
    reward: float
    done: bool
    info: dict[str, Any]
    observation_digest: str


def validate_observation(observation: Observation) -> None:
    if observation.shape != (12, 128, 96):
        raise AssertionError(f"Unexpected observation shape {observation.shape}")
    if observation.dtype != np.float32:
        raise AssertionError(f"Unexpected observation dtype {observation.dtype}")
    if not np.isfinite(observation).all():
        raise AssertionError("Observation contains a non-finite value")
    minimum = float(observation.min())
    maximum = float(observation.max())
    if minimum < 0.0 or maximum > 1.0:
        raise AssertionError(
            f"Observation range [{minimum}, {maximum}] lies outside [0, 1]"
        )


def run_trace(
    environment: JavaFxControlEnv,
    *,
    seed: int,
    actions: list[ControlAction],
) -> tuple[str, tuple[TransitionSignature, ...]]:
    observation = environment.reset(seed=seed)
    validate_observation(observation)
    reset_digest = observation_sha256(observation)
    transitions: list[TransitionSignature] = []
    for action in actions:
        observation, reward, done, info = environment.step(action)
        validate_observation(observation)
        transitions.append(
            TransitionSignature(
                action=int(action),
                reward=reward,
                done=done,
                info=dict(info),
                observation_digest=observation_sha256(observation),
            )
        )
        if done:
            break
    return reset_digest, tuple(transitions)


def write_failure(
    directory: Path,
    *,
    seed: int,
    actions: list[ControlAction],
    error: BaseException,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "stability_failure.json"
    payload = {
        "seed": seed,
        "actions": [int(action) for action in actions],
        "action_labels": [action.name.lower() for action in actions],
        "error_type": type(error).__name__,
        "error": str(error),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--determinism-steps", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--failure-dir", type=Path, default=Path("experiments/stability")
    )
    args = parser.parse_args()
    for name in ("determinism_steps", "episodes", "max_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    rng = random.Random(args.seed)
    action_space = list(ControlAction)
    deterministic_actions = [
        rng.choice(action_space) for _ in range(args.determinism_steps)
    ]
    checks: dict[str, str] = {}

    try:
        with JavaFxControlEnv() as environment:
            for action in action_space:
                initial = environment.reset(seed=args.seed)
                observation, _, _, _ = environment.step(action)
                validate_observation(observation)
                if observation_sha256(initial) == observation_sha256(observation):
                    raise AssertionError(
                        f"Action {action.name} produced no visible state change"
                    )
            checks["ui_action_acceptance"] = "passed"

            first = run_trace(
                environment, seed=args.seed, actions=deterministic_actions
            )
            second = run_trace(
                environment, seed=args.seed, actions=deterministic_actions
            )
            if first != second:
                raise AssertionError(
                    "Same-process replay differed for an identical seed and action trace"
                )
            checks["same_process_deterministic_replay"] = "passed"
            checks["observation_contract"] = "passed"

            total_steps = 0
            completed_episodes = 0
            for episode_index in range(args.episodes):
                episode_seed = args.seed + episode_index
                actions = [rng.choice(action_space) for _ in range(args.max_steps)]
                _, transitions = run_trace(
                    environment, seed=episode_seed, actions=actions
                )
                total_steps += len(transitions)
                if transitions and transitions[-1].done:
                    completed_episodes += 1
                    try:
                        environment.step(ControlAction.NOOP)
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError(
                            "Environment accepted a post-terminal action"
                        )
            checks["repeated_reset_and_step"] = "passed"
            checks["post_terminal_action_rejection"] = "passed"
        checks["bridge_close"] = "passed"
    except BaseException as error:
        failure_path = write_failure(
            args.failure_dir,
            seed=args.seed,
            actions=deterministic_actions,
            error=error,
        )
        print(json.dumps({"checks": checks, "failure": str(failure_path)}, indent=2))
        raise

    report = {
        "checks": checks,
        "episodes_requested": args.episodes,
        "episodes_reaching_done": completed_episodes,
        "soak_steps": total_steps,
        "not_checked": [
            "resident-memory growth",
            "cross-operating-system pixel equality",
            "training convergence",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
