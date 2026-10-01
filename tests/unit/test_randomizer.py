from itertools import islice

from tetris_rl.core.pieces import Tetromino
from tetris_rl.core.randomizer import seven_bag


def test_seven_bag_is_seeded_and_contains_every_piece() -> None:
    first = list(islice(seven_bag(seed=7), 14))
    second = list(islice(seven_bag(seed=7), 14))

    assert first == second
    assert set(first[:7]) == set(Tetromino)
    assert set(first[7:]) == set(Tetromino)
