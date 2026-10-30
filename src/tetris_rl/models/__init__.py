"""Neural-network components for learned Tetris agents."""

from tetris_rl.models.cnn_policy import VisualActorCritic
from tetris_rl.models.encoders import SmallCnnEncoder

__all__ = ["SmallCnnEncoder", "VisualActorCritic"]
