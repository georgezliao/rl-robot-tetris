from pathlib import Path

import numpy as np
import pytest

from tetris_rl.training.baselines import BASELINES, evaluate_baseline
from tetris_rl.training.config import load_visual_config
from tetris_rl.types import ControlAction

REFERENCE_CONFIG = Path("configs/visual/reference_ppo.yaml")


class CountingEnv:
    """Game over after `length` steps; one line per hard drop."""

    def __init__(self, length: int = 3) -> None:
        self.length = length
        self.seeds: list[int | None] = []
        self.actions: list[ControlAction] = []
        self.steps = 0
        self.lines = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.seeds.append(seed)
        self.steps = 0
        self.lines = 0
        return self._observation()

    def step(self, action: ControlAction | int) -> tuple:
        self.actions.append(ControlAction(int(action)))
        self.steps += 1
        reward = 1.0 if action == ControlAction.HARD_DROP else 0.0
        self.lines += int(reward)
        info = {
            "score": 10 * self.lines,
            "total_lines": self.lines,
            "pieces_placed": self.steps,
        }
        return self._observation(), reward, self.steps == self.length, info

    @staticmethod
    def _observation() -> np.ndarray:
        return np.zeros((12, 128, 96), dtype=np.float32)


def test_hard_drop_baseline_uses_held_out_seeds_and_summary_format(
    tmp_path: Path,
) -> None:
    config = load_visual_config(REFERENCE_CONFIG)
    environment = CountingEnv()

    result = evaluate_baseline(
        config,
        "hard_drop",
        environment,
        output_root=tmp_path,
        episode_count=2,
        max_steps_per_episode=10,
    )

    assert environment.seeds == [10000, 10001]
    assert set(environment.actions) == {ControlAction.HARD_DROP}
    assert result.summary["policy"] == "baseline:hard_drop"
    assert result.summary["total_lines"] == 6
    assert result.summary["total_pieces"] == 6
    assert result.summary["lines_per_100_pieces"] == 100.0
    assert result.summary["terminated_episodes"] == 2
    assert (result.run_directory / "summary.json").exists()


def test_step_limit_truncates_a_baseline_episode(tmp_path: Path) -> None:
    config = load_visual_config(REFERENCE_CONFIG)

    result = evaluate_baseline(
        config,
        "noop",
        CountingEnv(length=50),
        output_root=tmp_path,
        episode_count=1,
        max_steps_per_episode=4,
    )

    assert result.episodes[0].steps == 4
    assert result.episodes[0].truncated is True


def test_random_baseline_repeats_for_the_same_seed() -> None:
    actions = tuple(ControlAction)
    first = BASELINES["random"](actions, 10000)
    second = BASELINES["random"](actions, 10000)

    assert [first() for _ in range(20)] == [second() for _ in range(20)]


def test_unknown_baseline_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown baseline"):
        evaluate_baseline(
            load_visual_config(REFERENCE_CONFIG),
            "greedy",
            CountingEnv(),
            output_root=tmp_path,
            episode_count=1,
            max_steps_per_episode=1,
        )
