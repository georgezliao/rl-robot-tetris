"""Tetromino identities and rotation definitions."""

from enum import IntEnum


class Tetromino(IntEnum):
    I = 1
    O = 2
    T = 3
    S = 4
    Z = 5
    J = 6
    L = 7


SPAWN_SHAPES: dict[Tetromino, tuple[tuple[int, int], ...]] = {
    Tetromino.I: ((1, 0), (1, 1), (1, 2), (1, 3)),
    Tetromino.O: ((0, 1), (0, 2), (1, 1), (1, 2)),
    Tetromino.T: ((0, 1), (1, 0), (1, 1), (1, 2)),
    Tetromino.S: ((0, 1), (0, 2), (1, 0), (1, 1)),
    Tetromino.Z: ((0, 0), (0, 1), (1, 1), (1, 2)),
    Tetromino.J: ((0, 0), (1, 0), (1, 1), (1, 2)),
    Tetromino.L: ((0, 2), (1, 0), (1, 1), (1, 2)),
}
