"""Training orchestration for a CNN policy controlling the JavaFX UI."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from tetris_rl.envs.base import TetrisEnv
from tetris_rl.envs.javafx import JavaFxControlEnv
from tetris_rl.models.cnn_policy import VisualActorCritic
from tetris_rl.training.config import AlgorithmConfig, VisualExperimentConfig
from tetris_rl.training.io_utils import (
    append_jsonl,
    observation_to_tensor,
    write_json,
)
from tetris_rl.training.ppo import (
    PpoMetrics,
    generalized_advantage_estimates,
    ppo_update,
)
from tetris_rl.types import ControlAction


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Locations and counters produced by one bounded training run."""

    run_directory: Path
    environment_steps: int
    completed_episodes: int
    updates: int


def action_set(config: VisualExperimentConfig) -> tuple[ControlAction, ...]:
    """Return policy-index actions in a stable order."""
    actions = tuple(ControlAction)
    if not config.actions.allow_hold:
        actions = tuple(
            action for action in actions if action is not ControlAction.HOLD
        )
    return actions


def validate_runtime_support(config: VisualExperimentConfig) -> None:
    """Reject valid experimental settings not yet implemented by this trainer."""
    unsupported: list[str] = []
    if config.observation.region != "full_ui":
        unsupported.append("only observation.region=full_ui is implemented")
    if config.observation.color != "rgb":
        unsupported.append("only observation.color=rgb is implemented")
    if config.actions.gravity_ticks_per_step != 1:
        unsupported.append("only one gravity tick per step is implemented")
    if config.actions.macro_actions:
        unsupported.append("macro actions are not implemented")
    if config.reward.kind != "sparse_lines":
        unsupported.append("only reward.kind=sparse_lines is implemented")
    if unsupported:
        raise ValueError("; ".join(unsupported))


def resolve_device(name: str) -> torch.device:
    """Resolve a requested accelerator and fail rather than silently changing it."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)


def experiment_summary(config: VisualExperimentConfig) -> dict[str, Any]:
    """Describe the resolved model and data contract without starting the game."""
    validate_runtime_support(config)
    actions = action_set(config)
    model = VisualActorCritic(
        config.observation.shape, len(actions), config.model.feature_dim
    )
    device = resolve_device(config.training.device)
    return {
        "name": config.name,
        "observation_shape": list(config.observation.shape),
        "actions": [action.name for action in actions],
        "device": str(device),
        "model_parameters": model.parameter_count,
        "environment_steps": config.training.environment_steps,
        "status": "validated_without_environment_execution",
    }


@dataclass(slots=True)
class EpisodeProgress:
    """The episode in progress, carried from one rollout into the next.

    Episode ``index`` is played on piece seed ``config.seed + index``.
    """

    index: int = 0
    length: int = 0
    total_return: float = 0.0


@dataclass(frozen=True, slots=True)
class Rollout:
    """A fixed-length batch of consecutive transitions from one environment.

    Row ``t`` of every tensor describes the same transition: the observation the
    policy saw, the action it sampled, that action's log-probability and value
    estimate under the rollout policy, the reward, and whether the step ended an
    episode. ``final_observation`` is the observation after the last row, used
    to bootstrap the value when the rollout stops mid-episode.
    """

    observations: Tensor
    actions: Tensor
    log_probabilities: Tensor
    values: Tensor
    rewards: Tensor
    dones: Tensor
    final_observation: np.ndarray
    finished_episodes: tuple[dict[str, Any], ...]

    def __len__(self) -> int:
        return int(self.rewards.shape[0])


def collect_rollout(
    environment: TetrisEnv[ControlAction],
    model: VisualActorCritic,
    actions: tuple[ControlAction, ...],
    observation: np.ndarray,
    progress: EpisodeProgress,
    *,
    length: int,
    first_seed: int,
    terminal_penalty: float,
    steps_before: int,
    device: torch.device,
) -> Rollout:
    """Play ``length`` policy steps starting from ``observation``.

    When an episode ends, its summary is added to ``finished_episodes`` and the
    environment is reset on the next seed, so a rollout may span several
    episodes. ``progress`` is updated in place.
    """
    observations: list[Tensor] = []
    selected_actions: list[Tensor] = []
    log_probabilities: list[Tensor] = []
    values: list[Tensor] = []
    rewards: list[float] = []
    dones: list[bool] = []
    finished_episodes: list[dict[str, Any]] = []

    model.eval()
    for step in range(length):
        observation_tensor = observation_to_tensor(observation, device)
        policy_action, log_probability, value = model.act(observation_tensor)
        next_observation, raw_reward, done, info = environment.step(
            actions[int(policy_action.item())]
        )
        reward = raw_reward + (terminal_penalty if done else 0.0)

        observations.append(observation_tensor.squeeze(0))
        selected_actions.append(policy_action.squeeze(0))
        log_probabilities.append(log_probability.squeeze(0))
        values.append(value.squeeze(0))
        rewards.append(reward)
        dones.append(done)
        progress.length += 1
        progress.total_return += reward

        if done:
            finished_episodes.append(
                {
                    "type": "episode",
                    "episode": progress.index,
                    "seed": first_seed + progress.index,
                    "environment_steps": steps_before + step + 1,
                    "length": progress.length,
                    "return": progress.total_return,
                    "score": info.get("score"),
                    "total_lines": info.get("total_lines"),
                    "pieces_placed": info.get("pieces_placed"),
                }
            )
            progress.index += 1
            progress.length = 0
            progress.total_return = 0.0
            next_observation = environment.reset(seed=first_seed + progress.index)
        observation = next_observation

    return Rollout(
        observations=torch.stack(observations),
        actions=torch.stack(selected_actions).long(),
        log_probabilities=torch.stack(log_probabilities),
        values=torch.stack(values),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        dones=torch.tensor(dones, dtype=torch.float32, device=device),
        final_observation=observation,
        finished_episodes=tuple(finished_episodes),
    )


def bootstrap_value(
    model: VisualActorCritic, rollout: Rollout, device: torch.device
) -> Tensor:
    """Value of the state after the rollout; zero if the last step was terminal."""
    if rollout.dones[-1]:
        return torch.zeros((), device=device)
    with torch.no_grad():
        _, next_values = model(observation_to_tensor(rollout.final_observation, device))
    return next_values.squeeze(0)


def update_policy(
    model: VisualActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    config: AlgorithmConfig,
    device: torch.device,
) -> PpoMetrics:
    """Compute GAE advantages for ``rollout`` and run one PPO update on it."""
    advantages, returns = generalized_advantage_estimates(
        rollout.rewards,
        rollout.values,
        rollout.dones,
        bootstrap_value(model, rollout, device),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    return ppo_update(
        model,
        optimizer,
        rollout.observations,
        rollout.actions,
        rollout.log_probabilities,
        advantages,
        returns,
        config,
    )


def _checkpoint(
    path: Path,
    model: VisualActorCritic,
    optimizer: torch.optim.Optimizer,
    environment_steps: int,
    config: VisualExperimentConfig,
) -> None:
    torch.save(
        {
            "schema_version": 1,
            "environment_steps": environment_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
        },
        path,
    )


def _create_run_directory(
    config: VisualExperimentConfig,
    *,
    requested_steps: int,
    actions: tuple[ControlAction, ...],
    device: torch.device,
    model: VisualActorCritic,
) -> Path:
    """Create ``<output>/<name>-<UTC time>/checkpoints`` and the resolved config."""
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_directory = config.training.output_directory / f"{config.name}-{run_id}"
    (run_directory / "checkpoints").mkdir(parents=True)
    write_json(
        run_directory / "resolved_config.json",
        {
            **config.to_dict(),
            "executed_environment_steps": requested_steps,
            "action_labels": [action.name for action in actions],
            "resolved_device": str(device),
            "model_parameters": model.parameter_count,
        },
    )
    return run_directory


def _update_record(
    update: int, environment_steps: int, metrics: PpoMetrics
) -> dict[str, Any]:
    return {
        "type": "update",
        "update": update,
        "environment_steps": environment_steps,
        **{field: getattr(metrics, field) for field in metrics.__dataclass_fields__},
    }


def train_visual_agent(
    config: VisualExperimentConfig,
    *,
    environment_steps_override: int | None = None,
) -> TrainingResult:
    """Train the reference PPO policy for a finite number of UI interactions.

    Each iteration collects one rollout, updates the policy on it, and logs the
    finished episodes and update metrics to ``metrics.jsonl``. A checkpoint is
    saved after the first update at or past each ``checkpoint_interval``
    boundary, and ``final.pt`` at the end.
    """
    validate_runtime_support(config)
    requested_steps = (
        config.training.environment_steps
        if environment_steps_override is None
        else environment_steps_override
    )
    if requested_steps < 1:
        raise ValueError("environment steps must be positive")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.training.device)
    actions = action_set(config)
    model = VisualActorCritic(
        config.observation.shape, len(actions), config.model.feature_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.algorithm.learning_rate)

    run_directory = _create_run_directory(
        config,
        requested_steps=requested_steps,
        actions=actions,
        device=device,
        model=model,
    )
    checkpoint_directory = run_directory / "checkpoints"
    metrics_path = run_directory / "metrics.jsonl"

    steps_done = 0
    completed_episodes = 0
    updates = 0
    progress = EpisodeProgress()
    next_checkpoint = config.training.checkpoint_interval

    with JavaFxControlEnv(
        observation_size=(config.observation.height, config.observation.width),
        frame_stack=config.observation.frame_stack,
    ) as environment:
        observation = environment.reset(seed=config.seed)
        while steps_done < requested_steps:
            rollout = collect_rollout(
                environment,
                model,
                actions,
                observation,
                progress,
                length=min(
                    config.algorithm.rollout_steps, requested_steps - steps_done
                ),
                first_seed=config.seed,
                terminal_penalty=config.reward.terminal_penalty,
                steps_before=steps_done,
                device=device,
            )
            steps_done += len(rollout)
            observation = rollout.final_observation
            for episode in rollout.finished_episodes:
                append_jsonl(metrics_path, episode)
            completed_episodes += len(rollout.finished_episodes)

            update_metrics = update_policy(
                model, optimizer, rollout, config.algorithm, device
            )
            updates += 1
            append_jsonl(
                metrics_path, _update_record(updates, steps_done, update_metrics)
            )

            if steps_done >= next_checkpoint:
                _checkpoint(
                    checkpoint_directory / f"step-{steps_done}.pt",
                    model,
                    optimizer,
                    steps_done,
                    config,
                )
                while next_checkpoint <= steps_done:
                    next_checkpoint += config.training.checkpoint_interval

    _checkpoint(checkpoint_directory / "final.pt", model, optimizer, steps_done, config)
    write_json(
        run_directory / "summary.json",
        {
            "environment_steps": steps_done,
            "completed_episodes": completed_episodes,
            "updates": updates,
            "unfinished_episode_length": progress.length,
            "unfinished_episode_return": progress.total_return,
        },
    )
    return TrainingResult(run_directory, steps_done, completed_episodes, updates)
