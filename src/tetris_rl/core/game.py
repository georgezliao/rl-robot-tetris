"""Game state and the boundary for the rules engine."""

from dataclasses import dataclass, field

from tetris_rl.core.board import Board
from tetris_rl.core.pieces import Tetromino
from tetris_rl.types import ControlAction, PlacementAction, Transition


@dataclass(slots=True)
class GameState:
    board: Board = field(default_factory=Board)
    active_piece: Tetromino | None = None
    next_pieces: tuple[Tetromino, ...] = ()
    held_piece: Tetromino | None = None
    score: int = 0
    lines: int = 0
    pieces_placed: int = 0
    game_over: bool = False


class TetrisGame:
    """Deterministic engine shared by every environment and baseline."""

    def __init__(self, rows: int = 20, columns: int = 10) -> None:
        self.rows = rows
        self.columns = columns
        self.state = GameState(board=Board(rows, columns))

    def reset(self, seed: int | None = None) -> GameState:
        """Reset the game. Piece queue initialization is the next milestone."""
        del seed
        self.state = GameState(board=Board(self.rows, self.columns))
        return self.state

    def step_control(self, action: ControlAction) -> Transition:
        del action
        raise NotImplementedError("Implement movement, rotation, gravity, and locking")

    def step_placement(self, action: PlacementAction) -> Transition:
        del action
        raise NotImplementedError("Implement legal-placement enumeration and hard drop")
