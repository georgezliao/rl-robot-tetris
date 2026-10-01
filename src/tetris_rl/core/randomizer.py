"""Seedable piece generation."""

import random
from collections.abc import Iterator

from tetris_rl.core.pieces import Tetromino


def seven_bag(seed: int | None = None) -> Iterator[Tetromino]:
    """Yield fair, reproducible seven-piece bags indefinitely."""
    rng = random.Random(seed)
    pieces = list(Tetromino)
    while True:
        rng.shuffle(pieces)
        yield from pieces
