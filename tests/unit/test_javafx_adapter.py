import numpy as np
import pytest

from tetris_rl.envs.javafx import BridgeError, JavaFxControlEnv


def test_observation_converts_bridge_grid_to_cnn_channel() -> None:
    payload = {"observation": [[0] * 10 for _ in range(20)]}

    observation = JavaFxControlEnv._observation(payload)

    assert observation.shape == (1, 20, 10)
    assert observation.dtype == np.float32


def test_observation_rejects_wrong_board_shape() -> None:
    with pytest.raises(BridgeError):
        JavaFxControlEnv._observation({"observation": [[0]]})
