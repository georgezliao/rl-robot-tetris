"""Environment protocol used by training code."""

from typing import Generic, Protocol, TypeVar

from tetris_rl.types import Observation, StepResult

ActionT = TypeVar("ActionT", contravariant=True)


class TetrisEnv(Protocol, Generic[ActionT]):
    """Minimal four-value API, intentionally independent of an RL library."""

    def reset(self, seed: int | None = None) -> Observation: ...

    def step(self, action: ActionT) -> StepResult: ...
