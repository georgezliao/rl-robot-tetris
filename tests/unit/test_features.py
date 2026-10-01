import numpy as np

from tetris_rl.observations.features import extract_board_features


def test_extract_board_features_counts_holes_and_heights() -> None:
    grid = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )

    features = extract_board_features(grid)

    assert features.aggregate_height == 6
    assert features.max_height == 3
    assert features.holes == 1
    assert features.bumpiness == 3
    assert features.complete_lines == 1
