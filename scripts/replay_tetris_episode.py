"""Replay a recorded Tetris action sequence and report the first divergence."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from tetris_rl.envs.javafx import JavaFxControlEnv
from tetris_rl.events.observations import observation_sha256
from tetris_rl.events.recorder import read_episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument(
        "--ignore-pixels",
        action="store_true",
        help="Compare rewards, termination, and metrics without frame hashes",
    )
    args = parser.parse_args()
    recording = read_episode(args.recording)
    if recording.manifest.game_id != "tetris":
        raise SystemExit(
            f"Expected game_id 'tetris', got {recording.manifest.game_id!r}"
        )

    with JavaFxControlEnv() as environment:
        environment.reset(seed=recording.manifest.seed)
        for expected in recording.steps:
            observation, reward, done, info = environment.step(expected.action)
            differences: list[str] = []
            if not math.isclose(reward, expected.reward, rel_tol=1e-9, abs_tol=1e-12):
                differences.append(f"reward {reward} != {expected.reward}")
            if done != expected.done:
                differences.append(f"done {done} != {expected.done}")
            if info != expected.metrics:
                differences.append(f"metrics {info!r} != {expected.metrics!r}")
            if not args.ignore_pixels:
                actual_digest = observation_sha256(observation)
                if actual_digest != expected.observation_digest:
                    differences.append(
                        "observation digest "
                        f"{actual_digest} != {expected.observation_digest}"
                    )
            if differences:
                detail = "; ".join(differences)
                raise SystemExit(f"diverged at step {expected.step_index}: {detail}")

    print(
        f"replay matched {len(recording.steps)} steps "
        f"for episode {recording.manifest.episode_id}"
    )


if __name__ == "__main__":
    main()
