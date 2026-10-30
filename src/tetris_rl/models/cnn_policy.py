"""Actor-critic policy for rendered Tetris observations."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from tetris_rl.models.encoders import SmallCnnEncoder


class VisualActorCritic(nn.Module):
    """Shared CNN encoder with categorical-policy and scalar-value heads."""

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        action_count: int,
        feature_dim: int = 512,
    ) -> None:
        super().__init__()
        if action_count < 2:
            raise ValueError("action_count must be at least two")
        self.observation_shape = observation_shape
        self.action_count = action_count
        self.encoder = SmallCnnEncoder(observation_shape, feature_dim)
        self.policy_head = nn.Linear(feature_dim, action_count)
        self.value_head = nn.Linear(feature_dim, 1)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encoder(observation)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values

    def distribution_and_value(self, observation: Tensor) -> tuple[Categorical, Tensor]:
        logits, values = self(observation)
        return Categorical(logits=logits), values

    @torch.no_grad()
    def act(
        self, observation: Tensor, *, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution, values = self.distribution_and_value(observation)
        actions = (
            distribution.logits.argmax(dim=-1)
            if deterministic
            else distribution.sample()
        )
        log_probabilities = distribution.log_prob(actions)
        return actions, log_probabilities, values

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
