"""Proximal-policy-optimization calculations for a discrete action policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tetris_rl.models.cnn_policy import VisualActorCritic
from tetris_rl.training.config import AlgorithmConfig


@dataclass(frozen=True, slots=True)
class PpoMetrics:
    """Mean losses and diagnostics over one PPO update."""

    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float


def generalized_advantage_estimates(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    next_value: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Return advantages and value targets for a single ordered rollout.

    ``dones[t]`` is one when transition ``t`` terminates an episode. Therefore,
    both the one-step bootstrap and the recursive advantage are masked at that
    transition. The returned tensors have the same one-dimensional shape as the
    rewards.
    """
    if rewards.ndim != 1 or values.ndim != 1 or dones.ndim != 1:
        raise ValueError("rewards, values, and dones must be one-dimensional")
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must have matching shapes")
    if next_value.numel() != 1:
        raise ValueError("next_value must contain one scalar")

    advantages = torch.zeros_like(rewards)
    running_advantage = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    following_value = next_value.reshape(()).to(
        dtype=rewards.dtype, device=rewards.device
    )
    for index in range(rewards.numel() - 1, -1, -1):
        not_terminal = 1.0 - dones[index].to(rewards.dtype)
        temporal_difference = (
            rewards[index] + gamma * not_terminal * following_value - values[index]
        )
        running_advantage = (
            temporal_difference + gamma * gae_lambda * not_terminal * running_advantage
        )
        advantages[index] = running_advantage
        following_value = values[index]
    return advantages, advantages + values


def ppo_update(
    model: VisualActorCritic,
    optimizer: torch.optim.Optimizer,
    observations: Tensor,
    actions: Tensor,
    old_log_probabilities: Tensor,
    advantages: Tensor,
    returns: Tensor,
    config: AlgorithmConfig,
    *,
    generator: torch.Generator | None = None,
) -> PpoMetrics:
    """Perform one multi-epoch PPO update on a fixed rollout."""
    batch_size = observations.shape[0]
    if batch_size < 1:
        raise ValueError("a PPO update requires at least one transition")
    expected = (batch_size,)
    for name, tensor in {
        "actions": actions,
        "old_log_probabilities": old_log_probabilities,
        "advantages": advantages,
        "returns": returns,
    }.items():
        if tensor.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")

    normalized_advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    totals = torch.zeros(5, dtype=torch.float64)
    minibatch_count = 0
    minibatch_size = min(config.minibatch_size, batch_size)

    model.train()
    for _ in range(config.update_epochs):
        permutation = torch.randperm(
            batch_size, generator=generator, device=observations.device
        )
        for start in range(0, batch_size, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            distribution, predicted_values = model.distribution_and_value(
                observations[indices]
            )
            new_log_probabilities = distribution.log_prob(actions[indices])
            log_ratio = new_log_probabilities - old_log_probabilities[indices]
            probability_ratio = log_ratio.exp()
            selected_advantages = normalized_advantages[indices]
            unclipped = probability_ratio * selected_advantages
            clipped = (
                probability_ratio.clamp(
                    1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
                )
                * selected_advantages
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(predicted_values, returns[indices])
            entropy = distribution.entropy().mean()
            total_loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
            optimizer.step()

            with torch.no_grad():
                approximate_kl = ((probability_ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    ((probability_ratio - 1.0).abs() > config.clip_epsilon)
                    .float()
                    .mean()
                )
                totals += torch.tensor(
                    [
                        policy_loss.item(),
                        value_loss.item(),
                        entropy.item(),
                        approximate_kl.item(),
                        clip_fraction.item(),
                    ],
                    dtype=torch.float64,
                )
            minibatch_count += 1

    means = totals / minibatch_count
    return PpoMetrics(*[float(value) for value in means])
