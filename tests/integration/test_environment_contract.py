import numpy as np

from tetris_rl.envs.control import ControlEnv


def test_reset_returns_raw_grid_observation() -> None:
    observation = ControlEnv().reset(seed=42)

    assert observation.shape == (1, 20, 10)
    assert observation.dtype == np.float32
