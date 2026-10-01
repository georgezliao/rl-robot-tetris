"""Common policy interface."""

from typing import Protocol, TypeVar

from tetris_rl.types import Observation

ActionT = TypeVar("ActionT", covariant=True)


class Agent(Protocol[ActionT]):
    def act(self, observation: Observation, *, explore: bool = False) -> ActionT: ...
