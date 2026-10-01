"""Engineered board measurements for feature-based agents."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class BoardFeatures:
    aggregate_height: int
    max_height: int
    holes: int
    bumpiness: int
    complete_lines: int

    def as_array(self) -> npt.NDArray[np.float32]:
        return np.asarray(
            (
                self.aggregate_height,
                self.max_height,
                self.holes,
                self.bumpiness,
                self.complete_lines,
            ),
            dtype=np.float32,
        )


def extract_board_features(grid: npt.NDArray[np.uint8]) -> BoardFeatures:
    """Measure a settled board without assigning hand-tuned feature weights."""
    occupied = grid != 0
    rows = grid.shape[0]
    any_in_column = occupied.any(axis=0)
    first_occupied = np.argmax(occupied, axis=0)
    heights = np.where(any_in_column, rows - first_occupied, 0)

    holes = 0
    for column, height in enumerate(heights):
        if height:
            holes += int((~occupied[rows - height :, column]).sum())

    return BoardFeatures(
        aggregate_height=int(heights.sum()),
        max_height=int(heights.max(initial=0)),
        holes=holes,
        bumpiness=int(np.abs(np.diff(heights)).sum()),
        complete_lines=int(np.all(occupied, axis=1).sum()),
    )
