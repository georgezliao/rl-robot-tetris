"""Environment where the agent selects a rotation and landing column."""

from tetris_rl.core.game import TetrisGame
from tetris_rl.observations.raw_grid import encode_raw_grid
from tetris_rl.types import Observation, PlacementAction, StepResult


class PlacementEnv:
    def __init__(self, game: TetrisGame | None = None) -> None:
        self.game = game or TetrisGame()

    def reset(self, seed: int | None = None) -> Observation:
        state = self.game.reset(seed)
        return encode_raw_grid(state)

    def step(self, action: PlacementAction) -> StepResult:
        transition = self.game.step_placement(action)
        observation = encode_raw_grid(self.game.state)
        info = {"lines_cleared": transition.lines_cleared}
        return observation, float(transition.lines_cleared), transition.game_over, info
