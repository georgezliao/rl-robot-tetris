"""Sparse and shaped reward policies."""

from dataclasses import dataclass

from tetris_rl.observations.features import BoardFeatures
from tetris_rl.types import Transition


@dataclass(frozen=True, slots=True)
class RewardWeights:
    line_clear: float = 1.0
    holes: float = 0.0
    aggregate_height: float = 0.0
    bumpiness: float = 0.0
    game_over: float = -1.0


def shaped_reward(
    transition: Transition,
    before: BoardFeatures,
    after: BoardFeatures,
    weights: RewardWeights,
) -> float:
    """Reward changes in board quality; zero weights produce sparse rewards."""
    return (
        weights.line_clear * transition.lines_cleared
        + weights.holes * (after.holes - before.holes)
        + weights.aggregate_height * (after.aggregate_height - before.aggregate_height)
        + weights.bumpiness * (after.bumpiness - before.bumpiness)
        + weights.game_over * float(transition.game_over)
    )
