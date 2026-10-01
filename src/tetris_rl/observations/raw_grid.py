"""Raw board encoder for convolutional agents."""

import numpy as np

from tetris_rl.core.game import GameState
from tetris_rl.types import Observation


def encode_raw_grid(state: GameState) -> Observation:
    """Encode occupied cells as a single normalized channel: (1, rows, columns)."""
    occupied = (state.board.cells != 0).astype(np.float32)
    return occupied[np.newaxis, :, :]
