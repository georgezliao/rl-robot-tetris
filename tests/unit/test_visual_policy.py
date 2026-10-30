import math

import torch

from tetris_rl.models.cnn_policy import VisualActorCritic
from tetris_rl.training.config import AlgorithmConfig
from tetris_rl.training.ppo import (
    generalized_advantage_estimates,
    ppo_update,
)


def algorithm_config() -> AlgorithmConfig:
    return AlgorithmConfig(
        kind="ppo",
        learning_rate=0.001,
        gamma=0.9,
        gae_lambda=0.8,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        rollout_steps=4,
        update_epochs=2,
        minibatch_size=2,
        max_gradient_norm=0.5,
    )


def test_visual_actor_critic_output_shapes_are_finite() -> None:
    model = VisualActorCritic((3, 84, 84), action_count=7, feature_dim=32)

    logits, values = model(torch.zeros(2, 3, 84, 84))

    assert logits.shape == (2, 7)
    assert values.shape == (2,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()


def test_generalized_advantage_estimates_mask_terminal_bootstrap() -> None:
    advantages, returns = generalized_advantage_estimates(
        rewards=torch.tensor([1.0, 2.0]),
        values=torch.tensor([0.5, 0.25]),
        dones=torch.tensor([0.0, 1.0]),
        next_value=torch.tensor(99.0),
        gamma=0.9,
        gae_lambda=0.8,
    )

    torch.testing.assert_close(advantages, torch.tensor([1.985, 1.75]))
    torch.testing.assert_close(returns, torch.tensor([2.485, 2.0]))


def test_ppo_update_accepts_a_synthetic_rollout_and_changes_parameters() -> None:
    torch.manual_seed(7)
    model = VisualActorCritic((3, 84, 84), action_count=3, feature_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    observations = torch.rand(4, 3, 84, 84)
    with torch.no_grad():
        distribution, values = model.distribution_and_value(observations)
        actions = distribution.sample()
        old_log_probabilities = distribution.log_prob(actions)
    before = model.policy_head.weight.detach().clone()

    metrics = ppo_update(
        model,
        optimizer,
        observations,
        actions,
        old_log_probabilities,
        torch.tensor([1.0, -0.5, 0.25, 0.75]),
        values + torch.tensor([0.5, -0.25, 0.1, 0.4]),
        algorithm_config(),
    )

    assert not torch.equal(before, model.policy_head.weight)
    assert all(
        math.isfinite(getattr(metrics, field)) for field in metrics.__dataclass_fields__
    )
