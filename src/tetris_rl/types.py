"""Shared public types for the game and RL layers."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

Observation: TypeAlias = npt.NDArray[np.float32]
Info: TypeAlias = dict[str, Any]
StepResult: TypeAlias = tuple[Observation, float, bool, Info]


class ControlAction(IntEnum):
    """One input applied during a single game tick."""

    NOOP = 0
    LEFT = 1
    RIGHT = 2
    ROTATE_CLOCKWISE = 3
    ROTATE_COUNTERCLOCKWISE = 4
    SOFT_DROP = 5
    HARD_DROP = 6
    HOLD = 7


@dataclass(frozen=True, slots=True)
class PlacementAction:
    """A final rotation and column for a hard-dropped piece."""

    rotation: int
    column: int


@dataclass(frozen=True, slots=True)
class Transition:
    """Facts produced by the game engine after an action."""

    lines_cleared: int = 0
    piece_locked: bool = False
    hard_drop_distance: int = 0
    game_over: bool = False
