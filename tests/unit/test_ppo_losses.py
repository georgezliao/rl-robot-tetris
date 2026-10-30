"""PPO loss terms checked against values worked out by hand.

Four samples, all with old log-probability 0, clip epsilon 0.2:

| sample | ratio | advantage | unclipped | clipped | min  | case                  |
|--------|-------|-----------|-----------|---------|------|-----------------------|
| a      | 1.5   | +1        | 1.5       | 1.2     | 1.2  | clipped above         |
| b      | 0.5   | -1        | -0.5      | -0.8    | -0.8 | clipped below         |
| c      | 1.5   | -1        | -1.5      | -1.2    | -1.5 | outside, not clipped  |
| d      | 1.0   | +2        | 2.0       | 2.0     | 2.0  | inside the clip range |

policy loss = -(1.2 - 0.8 - 1.5 + 2.0) / 4 = -0.225
value loss  = mean((0-1)^2, (1-1)^2, (2-0)^2, (3-3)^2) = 5 / 4 = 1.25
entropy     = mean(0.5, 1.0, 1.5, 1.0) = 1.0
total       = -0.225 + 0.5 * 1.25 - 0.01 * 1.0 = 0.39
"""

import math

import torch

from tetris_rl.training.config import AlgorithmConfig
from tetris_rl.training.ppo import ppo_losses

RATIOS = [1.5, 0.5, 1.5, 1.0]
TOLERANCE = {"atol": 1e-6, "rtol": 0.0}


def config() -> AlgorithmConfig:
    return AlgorithmConfig(
        kind="ppo",
        learning_rate=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        rollout_steps=4,
        update_epochs=1,
        minibatch_size=4,
        max_gradient_norm=0.5,
    )


def losses_and_log_probabilities():
    new_log_probabilities = torch.tensor(
        [math.log(ratio) for ratio in RATIOS], requires_grad=True
    )
    losses = ppo_losses(
        new_log_probabilities,
        old_log_probabilities=torch.zeros(4),
        advantages=torch.tensor([1.0, -1.0, -1.0, 2.0]),
        predicted_values=torch.tensor([0.0, 1.0, 2.0, 3.0]),
        returns=torch.tensor([1.0, 1.0, 0.0, 3.0]),
        entropies=torch.tensor([0.5, 1.0, 1.5, 1.0]),
        config=config(),
    )
    return losses, new_log_probabilities


def test_loss_terms_match_hand_computed_values() -> None:
    losses, _ = losses_and_log_probabilities()

    torch.testing.assert_close(losses.policy, torch.tensor(-0.225), **TOLERANCE)
    torch.testing.assert_close(losses.value, torch.tensor(1.25), **TOLERANCE)
    torch.testing.assert_close(losses.entropy, torch.tensor(1.0), **TOLERANCE)
    torch.testing.assert_close(losses.total, torch.tensor(0.39), **TOLERANCE)


def test_diagnostics_match_hand_computed_values() -> None:
    losses, _ = losses_and_log_probabilities()
    expected_kl = sum((ratio - 1.0) - math.log(ratio) for ratio in RATIOS) / 4

    torch.testing.assert_close(
        losses.approximate_kl, torch.tensor(expected_kl), **TOLERANCE
    )
    # Samples a, b and c have |ratio - 1| > 0.2.
    torch.testing.assert_close(losses.clip_fraction, torch.tensor(0.75), **TOLERANCE)


def test_clipped_samples_receive_no_policy_gradient() -> None:
    losses, new_log_probabilities = losses_and_log_probabilities()

    # Only the policy term depends on the log-probabilities, so the gradient of
    # the total is the policy gradient. For an unclipped sample,
    # d(-ratio * A / 4) / d(log ratio) = -ratio * A / 4.
    losses.total.backward()

    torch.testing.assert_close(
        new_log_probabilities.grad,
        torch.tensor([0.0, 0.0, 1.5 / 4, -2.0 / 4]),
        **TOLERANCE,
    )
