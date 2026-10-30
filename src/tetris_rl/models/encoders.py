"""Convolutional encoders for rendered visual observations."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SmallCnnEncoder(nn.Module):
    """Encode a channel-first image using three convolutions and one projection."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        feature_dim: int = 512,
    ) -> None:
        super().__init__()
        channels, height, width = observation_shape
        if channels < 1 or height < 1 or width < 1:
            raise ValueError("observation_shape dimensions must be positive")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")

        self.observation_shape = observation_shape
        self.feature_dim = feature_dim
        self.convolutions = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        try:
            with torch.no_grad():
                convolution_output = self.convolutions(
                    torch.zeros(1, channels, height, width)
                )
        except RuntimeError as exception:
            raise ValueError(
                f"observation_shape {observation_shape} is too small for small_cnn"
            ) from exception
        flattened_dim = math.prod(convolution_output.shape[1:])
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, feature_dim),
            nn.ReLU(),
        )

    def forward(self, observation: Tensor) -> Tensor:
        if observation.ndim != 4:
            raise ValueError(
                "observation must have shape (batch, channels, height, width)"
            )
        if tuple(observation.shape[1:]) != self.observation_shape:
            raise ValueError(
                f"expected observation shape {self.observation_shape}, "
                f"received {tuple(observation.shape[1:])}"
            )
        return self.projection(self.convolutions(observation))
