import numpy as np

from tetris_rl.core.board import Board


def test_clear_full_lines_compacts_board_downward() -> None:
    board = Board(rows=4, columns=3)
    board.cells[-1] = 1
    board.cells[-2, 0] = 1

    assert board.clear_full_lines() == 1
    np.testing.assert_array_equal(board.cells[-1], [1, 0, 0])
    np.testing.assert_array_equal(board.cells[0], [0, 0, 0])
