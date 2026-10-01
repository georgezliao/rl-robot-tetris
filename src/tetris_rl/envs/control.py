"""Environment where the agent supplies one controller input per tick."""

from tetris_rl.core.game import TetrisGame
from tetris_rl.observations.raw_grid import encode_raw_grid
from tetris_rl.types import ControlAction, Observation, StepResult


class ControlEnv:
    def __init__(self, game: TetrisGame | None = None) -> None:
        self.game = game or TetrisGame()

    def reset(self, seed: int | None = None) -> Observation:
        state = self.game.reset(seed)
        return encode_raw_grid(state)

    def step(self, action: ControlAction) -> StepResult:
        transition = self.game.step_control(action)
        observation = encode_raw_grid(self.game.state)
        info = {"lines_cleared": transition.lines_cleared}
        return observation, float(transition.lines_cleared), transition.game_over, info
