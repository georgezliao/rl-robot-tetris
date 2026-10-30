"""Common policy interface."""

from typing import Protocol, TypeVar

from tetris_rl.types import Observation

ActionT_co = TypeVar("ActionT_co", covariant=True)


class Agent(Protocol[ActionT_co]):
    def act(self, observation: Observation, *, explore: bool = False) -> ActionT_co: ...
