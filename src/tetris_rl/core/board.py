"""Board representation and board-level operations."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class Board:
    rows: int = 20
    columns: int = 10
    cells: npt.NDArray[np.uint8] = field(init=False)

    def __post_init__(self) -> None:
        self.cells = np.zeros((self.rows, self.columns), dtype=np.uint8)

    def copy_grid(self) -> npt.NDArray[np.uint8]:
        """Return a copy so observations cannot mutate the game."""
        return self.cells.copy()

    def clear_full_lines(self) -> int:
        """Remove complete rows, insert empty rows, and return their count."""
        complete = np.all(self.cells != 0, axis=1)
        count = int(complete.sum())
        if count:
            remaining = self.cells[~complete]
            empty = np.zeros((count, self.columns), dtype=np.uint8)
            self.cells = np.vstack((empty, remaining))
        return count
