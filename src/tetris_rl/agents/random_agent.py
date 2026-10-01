"""Random baseline and environment smoke-test policy."""

import random
from collections.abc import Sequence
from typing import Generic, TypeVar

from tetris_rl.types import Observation

ActionT = TypeVar("ActionT")


class RandomAgent(Generic[ActionT]):
    def __init__(self, actions: Sequence[ActionT], seed: int | None = None) -> None:
        if not actions:
            raise ValueError("actions must not be empty")
        self.actions = actions
        self.rng = random.Random(seed)

    def act(self, observation: Observation, *, explore: bool = False) -> ActionT:
        del observation, explore
        return self.rng.choice(self.actions)
