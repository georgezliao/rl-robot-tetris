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


@dataclass(frozen=True, slots=True)
class PpoLosses:
    """Loss terms for one minibatch.

    ``total`` keeps its autograd graph; every other field is detached.
    """

    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor


def ppo_losses(
    new_log_probabilities: Tensor,
    old_log_probabilities: Tensor,
    advantages: Tensor,
    predicted_values: Tensor,
    returns: Tensor,
    entropies: Tensor,
    config: AlgorithmConfig,
) -> PpoLosses:
    """Return the PPO loss terms for one minibatch of sampled actions.

    With probability ratio ``rho = exp(new_log_prob - old_log_prob)`` and
    clip range ``epsilon``:

    - policy loss: ``-mean(min(rho * A, clip(rho, 1 - epsilon, 1 + epsilon) * A))``.
      Taking the minimum means a sample stops contributing gradient once its
      ratio has moved past the clip boundary in the direction that helps it.
    - value loss: ``mean((V - R) ** 2)``.
    - total: ``policy + value_coefficient * value - entropy_coefficient * entropy``.

    ``approximate_kl`` is the ``mean((rho - 1) - log rho)`` estimator of
    KL(old || new), and ``clip_fraction`` is the share of samples with
    ``|rho - 1| > epsilon``. Both are diagnostics only.
    """
    log_ratio = new_log_probabilities - old_log_probabilities
    probability_ratio = log_ratio.exp()
    unclipped = probability_ratio * advantages
    clipped = (
        probability_ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
        * advantages
    )
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = nn.functional.mse_loss(predicted_values, returns)
    entropy = entropies.mean()
    total_loss = (
        policy_loss
        + config.value_coefficient * value_loss
        - config.entropy_coefficient * entropy
    )
    with torch.no_grad():
        approximate_kl = ((probability_ratio - 1.0) - log_ratio).mean()
        clip_fraction = (
            ((probability_ratio - 1.0).abs() > config.clip_epsilon).float().mean()
        )
    return PpoLosses(
        total=total_loss,
        policy=policy_loss.detach(),
        value=value_loss.detach(),
        entropy=entropy.detach(),
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
    )


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

    Working backward from the last transition::

        delta_t = r_t + gamma * (1 - d_t) * V_(t+1) - V_t
        A_t     = delta_t + gamma * lambda * (1 - d_t) * A_(t+1)
        R_t     = A_t + V_t

    where ``V_(t+1)`` is ``values[t + 1]``, or ``next_value`` for the last row.

    - ``dones[t] == 1`` ends an episode at ``t``. Nothing after ``t`` (neither
      the next value nor the next advantage) flows back into ``t``, so several
      episodes can share one rollout.
    - If the rollout stops mid-episode (last ``dones`` is zero), ``next_value``
      stands in for the unseen rest of the episode. If the last step is
      terminal, ``next_value`` is ignored.
    - ``gae_lambda = 0`` gives one-step TD errors (``A_t = delta_t``);
      ``gae_lambda = 1`` gives the discounted return to the episode end (or
      bootstrap) minus ``V_t``. ``gamma = 0`` reduces both to ``r_t - V_t``.

    All tensors are one-dimensional with one entry per transition.
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
            losses = ppo_losses(
                distribution.log_prob(actions[indices]),
                old_log_probabilities[indices],
                normalized_advantages[indices],
                predicted_values,
                returns[indices],
                distribution.entropy(),
                config,
            )

            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)
            optimizer.step()

            totals += torch.tensor(
                [
                    losses.policy.item(),
                    losses.value.item(),
                    losses.entropy.item(),
                    losses.approximate_kl.item(),
                    losses.clip_fraction.item(),
                ],
                dtype=torch.float64,
            )
            minibatch_count += 1

    means = totals / minibatch_count
    return PpoMetrics(*[float(value) for value in means])
