"""Record one random-policy visual Tetris episode as a generic JSONL run."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from tetris_rl.envs.javafx import JavaFxControlEnv
from tetris_rl.events.observations import observation_sha256
from tetris_rl.events.recorder import EpisodeRecorder
from tetris_rl.events.schema import EpisodeManifest, JsonScalar, StepEvent
from tetris_rl.types import ControlAction

ENVIRONMENT_VERSION = "tetris-javafx-visual-v1"
ACTION_SPEC = tuple(action.name.lower() for action in ControlAction)


def scalar_metrics(info: dict[str, object]) -> dict[str, JsonScalar]:
    """Retain JSON scalar metrics and reject unsupported structured values."""
    metrics: dict[str, JsonScalar] = {}
    for key, value in info.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            metrics[key] = value
        else:
            raise TypeError(f"Metric {key!r} is not a JSON scalar: {type(value)}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/records"))
    parser.add_argument("--seed", type=int, default=42, help="Tetris piece seed")
    parser.add_argument(
        "--action-seed", type=int, default=42, help="Random-policy action seed"
    )
    parser.add_argument("--max-steps", type=int, default=10_000)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")

    manifest = EpisodeManifest.create(
        game_id="tetris",
        environment_version=ENVIRONMENT_VERSION,
        seed=args.seed,
        action_spec=ACTION_SPEC,
        observation_spec={
            "kind": "stacked_rendered_rgb",
            "shape": "12x128x96",
            "dtype": "float32",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        configuration={
            "frame_stack": 4,
            "observation_height": 128,
            "observation_width": 96,
            "gravity_ticks_per_step": 1,
            "policy": "uniform_random",
            "action_seed": args.action_seed,
        },
        model_id=f"uniform-random:{args.action_seed}",
    )
    output_directory = args.output_root / manifest.episode_id
    rng = random.Random(args.action_seed)
    actions = list(ControlAction)

    with (
        JavaFxControlEnv() as environment,
        EpisodeRecorder(output_directory, manifest) as recorder,
    ):
        observation = environment.reset(seed=args.seed)
        if observation.shape != (12, 128, 96) or observation.dtype != np.float32:
            raise RuntimeError(
                f"Unexpected reset observation: {observation.shape}, {observation.dtype}"
            )

        cumulative_reward = 0.0
        done = False
        final_info: dict[str, object] = {}
        for step_index in range(args.max_steps):
            action = rng.choice(actions)
            observation, reward, done, final_info = environment.step(action)
            cumulative_reward += reward
            recorder.record(
                StepEvent(
                    episode_id=manifest.episode_id,
                    step_index=step_index,
                    action=int(action),
                    action_label=action.name.lower(),
                    reward=reward,
                    cumulative_reward=cumulative_reward,
                    done=done,
                    metrics=scalar_metrics(final_info),
                    observation_digest=observation_sha256(observation),
                )
            )
            if done:
                break

    print(f"recording={output_directory}")
    print(
        f"steps={step_index + 1} reward={cumulative_reward:.6g} "
        f"done={done} metrics={final_info}"
    )


if __name__ == "__main__":
    main()
