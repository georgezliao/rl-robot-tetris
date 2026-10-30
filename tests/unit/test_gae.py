"""Edge cases for generalized_advantage_estimates, each worked out by hand."""

import torch

from tetris_rl.training.ppo import generalized_advantage_estimates

TOLERANCE = {"atol": 1e-6, "rtol": 0.0}


def gae(rewards, values, dones, next_value, *, gamma, gae_lambda):
    return generalized_advantage_estimates(
        torch.tensor(rewards),
        torch.tensor(values),
        torch.tensor(dones),
        torch.tensor(next_value),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )


def test_rollout_cut_mid_episode_bootstraps_from_next_value() -> None:
    # lambda = 1 with zero value estimates: the return is the discounted reward
    # sum plus the discounted bootstrap. R_0 = 1 + 0.5 * 1 + 0.25 * 10 = 4.
    advantages, returns = gae(
        [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], 10.0, gamma=0.5, gae_lambda=1.0
    )

    torch.testing.assert_close(returns, torch.tensor([4.0, 6.0]), **TOLERANCE)
    torch.testing.assert_close(advantages, torch.tensor([4.0, 6.0]), **TOLERANCE)


def test_two_episodes_in_one_rollout_do_not_leak_into_each_other() -> None:
    # Episode one is steps 0-1 (terminal at 1); episode two is steps 2-3 and is
    # cut off, so it bootstraps from next_value = 2. With gamma = lambda = 1 the
    # return is the plain reward sum to the end of each episode.
    advantages, returns = gae(
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 0.5, 0.5, 0.5],
        [0.0, 1.0, 0.0, 0.0],
        2.0,
        gamma=1.0,
        gae_lambda=1.0,
    )

    expected_returns = torch.tensor([1.0 + 2.0, 2.0, 3.0 + 4.0 + 2.0, 4.0 + 2.0])
    torch.testing.assert_close(returns, expected_returns, **TOLERANCE)
    torch.testing.assert_close(advantages, expected_returns - 0.5, **TOLERANCE)


def test_terminal_last_step_ignores_next_value() -> None:
    advantages, _ = gae([1.0], [0.25], [1.0], 99.0, gamma=0.9, gae_lambda=0.9)

    torch.testing.assert_close(advantages, torch.tensor([0.75]), **TOLERANCE)


def test_lambda_zero_gives_one_step_td_errors() -> None:
    # delta_1 = 2 + 0.9 * 5 - 3 = 3.5 and delta_0 = 1 + 0.9 * 3 - 1 = 2.7.
    advantages, _ = gae(
        [1.0, 2.0], [1.0, 3.0], [0.0, 0.0], 5.0, gamma=0.9, gae_lambda=0.0
    )

    torch.testing.assert_close(advantages, torch.tensor([2.7, 3.5]), **TOLERANCE)


def test_gamma_zero_gives_reward_minus_value() -> None:
    advantages, returns = gae(
        [1.0, 2.0], [0.5, 0.25], [0.0, 0.0], 99.0, gamma=0.0, gae_lambda=0.9
    )

    torch.testing.assert_close(advantages, torch.tensor([0.5, 1.75]), **TOLERANCE)
    torch.testing.assert_close(returns, torch.tensor([1.0, 2.0]), **TOLERANCE)
